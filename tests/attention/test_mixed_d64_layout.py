# SPDX-License-Identifier: Apache-2.0
"""Run actual producer address arithmetic on CPU with instrumented async loads."""

from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).parents[2]


def function(source, name):
    match = re.search(r"\b" + name + r"\s*\(", source)
    assert match
    start = source.rfind("template <", 0, match.start())
    brace = source.index("{", match.end())
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_actual_paged_producer_byte_coverage(tmp_path):
    compiler = shutil.which("g++")
    assert compiler, "CPU address qualification requires a C++ compiler"
    source = (ROOT / "include/flashinfer/attention/prefill.cuh").read_text()
    permuted = (
        (ROOT / "include/flashinfer/permuted_smem.cuh")
        .read_text()
        .split("struct smem_t", 1)[1]
    )
    methods = "\n".join(
        function(permuted, name)
        for name in [
            "get_permuted_offset",
            "advance_offset_by_column",
            "advance_offset_by_row",
        ]
    )
    traits = source[
        source.index("  static constexpr SwizzleMode SWIZZLE_MODE_KV =") : source.index(
            "  static constexpr PosEncodingMode POS_ENCODING_MODE ="
        )
    ]
    for marker in [
        "warp_idx * V_THR_LAYOUT_ROW + lane_idx / V_THR_LAYOUT_COL",
        "size_t thr_local_v_offset[NUM_MMA_KV * V_THR_LAYOUT_COL / 2 / NUM_WARPS_Q]",
    ]:
        assert marker in source
    actual = (
        function(source, "page_source_offsets")
        + "\n"
        + function(source, "page_produce_kv")
    )
    program = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <unordered_map>
#include <vector>
#define __device__
#define __forceinline__ inline
using std::size_t;
enum class SwizzleMode {k128B,k64B};
enum class SharedMemFillMode {kNoFill,kFillZero};
template<class T> constexpr bool is_fp4_type_v = false;
template<class T> constexpr int upcast_size() {return 16/sizeof(T);}
struct Record {size_t bytes; std::vector<int> writes;};
std::unordered_map<void*, Record> records;
template<SwizzleMode swizzle_mode> struct smem_t {
  uint8_t* base;
  template<class T> smem_t(T* p):base(reinterpret_cast<uint8_t*>(p)) {}
  METHODS
  template<SharedMemFillMode Fill, class T>
  void load_128b_async(uint32_t offset, const T* src, bool predicate) {
    auto& rec=records.at(base);
    assert(offset*16+16<=rec.bytes);
    if(predicate || Fill==SharedMemFillMode::kFillZero) {
      if(predicate) std::memcpy(base+offset*16,src,16);
      else std::memset(base+offset*16,0,16);
      for(int j=0;j<16;++j) ++rec.writes[offset*16+j];
    }
  }
  template<SharedMemFillMode Fill,class T>
  void load_64b_async(uint32_t,const T*,bool) {assert(false);}
};
template<int B> struct Blob {uint8_t bytes[B];};
template<int HD,int KB,int VB,int WQ,int WK,int MM> struct Traits {
  using DTypeKV_=Blob<KB>; using DTypeV_=Blob<VB>;
  using DTypeKV=DTypeKV_; using DTypeK=DTypeKV_; using DTypeV=DTypeV_;
  using IdType=int32_t;
  static constexpr int HEAD_DIM_VO=HD;
  TRAITS
  static constexpr int NUM_WARPS_Q=WQ,NUM_WARPS_KV=WK,NUM_WARPS=WQ*WK;
  static constexpr int NUM_MMA_KV=MM,CTA_TILE_KV=16*MM*WK;
  static constexpr int NUM_MMA_D_QK=HD/16,NUM_MMA_D_VO=HD/16;
  static constexpr int UPCAST_STRIDE_K=HD*KB/16,UPCAST_STRIDE_V=HD*VB/16;
  static constexpr bool USE_KV_SHARED_SMEM=false;
};
struct PageSize {void divmod(uint32_t x,uint32_t& p,uint32_t& e)const {p=x/16;e=x%16;}};
struct Pages {
  PageSize page_size; int dimension; std::vector<int> ids{3,0,2};
  size_t protective_get_kv_offset(uint32_t p,uint32_t h,uint32_t e,uint32_t col,int32_t bound)const {
    return p<uint32_t(bound) ? (ids[p]*16+e)*dimension+col : 0;
  }
};
ACTUAL

template<class T,bool V> void verify(int live) {
  constexpr int B=sizeof(std::conditional_t<V,typename T::DTypeV,typename T::DTypeK>);
  constexpr int HD=T::HEAD_DIM_VO,ROWS=T::CTA_TILE_KV;
  using E=Blob<B>;
  struct Storage {typename T::DTypeK k_smem[ROWS*HD];typename T::DTypeV v_smem[ROWS*HD];} storage;
  std::memset(&storage,0xa5,sizeof(storage));
  E* dest;
  if constexpr(V) dest=reinterpret_cast<E*>(storage.v_smem);
  else dest=reinterpret_cast<E*>(storage.k_smem);
  auto bytes=reinterpret_cast<uint8_t*>(dest);
  records[bytes]={ROWS*HD*B,std::vector<int>(ROWS*HD*B,0)};
  std::vector<E> input(4*16*HD);
  for(size_t i=0;i<input.size()*B;++i) reinterpret_cast<uint8_t*>(input.data())[i]=(i*7+19)%251;
  constexpr auto SW=V?T::SWIZZLE_MODE_V:T::SWIZZLE_MODE_KV;
  constexpr int ROW=SW==SwizzleMode::k128B?4:8,COL=32/ROW;
  constexpr int STRIDE=HD*B/16;
  Pages pages{{},HD};
  for(int warp=0;warp<T::NUM_WARPS;++warp) for(int lane=0;lane<32;++lane) {
    size_t offsets[T::NUM_MMA_KV*COL/2/T::NUM_WARPS_Q];
    page_source_offsets<V,T>(pages,0,0,3,warp,lane,offsets);
    uint32_t dst=smem_t<SW>::template get_permuted_offset<STRIDE>(warp*ROW+lane/COL,lane%COL);
    page_produce_kv<V,T>(&storage,&dst,input.data(),0,offsets,live,warp,lane);
  }
  for(int row=0;row<ROWS;++row) for(int col=0;col<HD*B;++col) {
    int dst=smem_t<SW>::template get_permuted_offset<STRIDE>(row,col/16)*16+col%16;
    assert(dst>=0 && dst<ROWS*HD*B);
    if(row<live) {
      size_t src=(pages.ids[row/16]*16+row%16)*HD*B+col;
      assert(bytes[dst]==reinterpret_cast<uint8_t*>(input.data())[src]);
      assert(records.at(bytes).writes[dst]==1);
    } else {
      assert(bytes[dst]==(V?0:0xa5));
      assert(records.at(bytes).writes[dst]==(V?1:0));
    }
  }
  records.clear();
}
template<int HD,int KB,int VB,int WQ,int WK,int MM> void cases() {
  for(int live:{1,16,17,32,33,47}) {
    verify<Traits<HD,KB,VB,WQ,WK,MM>,false>(live);
    verify<Traits<HD,KB,VB,WQ,WK,MM>,true>(live);
  }
}
int main() {
  cases<64,2,1,4,1,4>();cases<64,2,1,1,4,4>();cases<64,2,1,4,1,8>();
  cases<128,2,1,4,1,4>();cases<128,2,1,1,4,4>();
  cases<64,2,2,4,1,4>();cases<64,2,2,1,4,4>();
  cases<64,1,1,4,1,4>();cases<64,1,1,1,4,4>();
  std::cout<<"108 producer-side cases passed\n";
}
"""
    for token, value in [("METHODS", methods), ("TRAITS", traits), ("ACTUAL", actual)]:
        program = program.replace(token, value)
    cpp = tmp_path / "coverage.cpp"
    cpp.write_text(program)
    binary = tmp_path / "coverage"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-O0", "-Werror", str(cpp), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "108 producer-side cases passed" in result.stdout
