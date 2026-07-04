/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_MAINLOOP_CUH_
#define FLASHINFER_ATTENTION_HOPPER_MAINLOOP_CUH_

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "../../math.cuh"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "mainloop_mma.cuh"
#include "named_barrier.cuh"
#include "utils.cuh"

namespace flashinfer {

using namespace cute;

template <typename AdditionalParams, typename Ktraits, bool CAUSAL>
struct CollectiveMainloop {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  // Value-side dtype. For asymmetric K/V (16-bit K, FP8 V) DTypeV != DTypeKV: V
  // is stored FP8 in gmem but dequantized to DTypeKV (16-bit) in the producer so
  // the smem V tile and the PV WGMMA stay 16-bit (mirrors the paged path).
  using DTypeV = typename Ktraits::DTypeV;
  static constexpr bool IS_ASYM = Ktraits::IS_ASYM;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;
  static constexpr int CTA_Q = get<0>(TileShape_QKD{});
  static constexpr int CTA_KV = get<1>(TileShape_QKD{});

  static constexpr int NUM_STAGES = Ktraits::NUM_STAGES;
  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  static constexpr int HEAD_DIM_QK = Ktraits::HEAD_DIM_QK;
  static constexpr int HEAD_DIM_VO = Ktraits::HEAD_DIM_VO;

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV = cute::SM90_TMA_LOAD;

  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;

  // Asymmetric K/V: cp.async copy machinery for loading FP8 V from gmem (the TMA
  // V path cannot dtype-convert). Mirrors the paged SparseCollectiveMainloop. The
  // atom is used only to *partition* the V tile into per-thread fragments (dest
  // smem + predication coords); the actual load is a manual FP8 LDG + convert +
  // 16-bit STS in load(). Unused (but well-formed) for symmetric K/V.
  static constexpr int NUM_COPY_THREADS = cutlass::NumThreadsPerWarpGroup;
  static constexpr auto AlignmentKV = 128 / cutlass::sizeof_bits<DTypeKV>::value;
  using AlignmentTypeKV = cute::uint_byte_t<static_cast<int>(sizeof(DTypeKV)) * AlignmentKV>;
  using GmemCopyAtomV = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<AlignmentTypeKV>, DTypeKV>;
  using GmemTiledCopyV =
      decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
               GmemCopyAtomV, NUM_COPY_THREADS, AlignmentKV,
               cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>,
               decltype(cute::get<2>(TileShape_PDV{})), decltype(cute::get<1>(TileShape_PDV{}))>());

  using ShapeT = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideT = cute::Shape<int64_t, _1, int64_t>;  // (N, D, H)
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using ShapeLseT = cute::Shape<int32_t, int32_t>;
  using StrideLseT = cute::Shape<_1, int64_t>;
  using LayoutLseT = cute::Layout<ShapeLseT, StrideLseT>;

  using TMA_Q = decltype(make_tma_copy(
      GmemTiledCopyQ{},
      make_tensor(make_gmem_ptr(static_cast<DTypeQ const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)), StrideT{}),
      SmemLayoutQ{}, select<0, 2>(TileShape_QKD{}), _1{}));  // no mcast for Q

  using TMA_K = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<DTypeKV const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)), StrideT{}),
      take<0, 2>(SmemLayoutK{}), select<1, 2>(TileShape_QKD{}), _1{}));  // no mcast

  using TMA_V = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<DTypeKV const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)), StrideT{}),
      take<0, 2>(SmemLayoutV{}), select<2, 1>(TileShape_PDV{}), _1{}));  // no mcast

  static constexpr bool USE_TMA_LOAD_KV = true;
  // K always loads via TMA. V loads via TMA too, except for asymmetric K/V
  // (16-bit K, FP8 V): TMA cannot dtype-convert FP8->16-bit, so V is loaded via
  // cp.async and dequantized in the producer (MainloopPipelineV = PipelineAsync
  // for asym, PipelineTmaAsync for symmetric). See kernel_traits.cuh.
  using MainloopPipelineK = typename Ktraits::MainloopPipelineK;
  using MainloopPipelineV = typename Ktraits::MainloopPipelineV;
  using MainloopPipeline = MainloopPipelineK;  // back-compat alias
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename Ktraits::PipelineState;

  // Set the bytes transferred in this TMA transaction (may involve multiple issues)
  static constexpr uint32_t TmaTransactionBytesQ =
      static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<DTypeQ> / 8);
  static constexpr uint32_t TmaTransactionBytesK =
      static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<DTypeKV> / 8);
  static constexpr uint32_t TmaTransactionBytesV =
      static_cast<uint32_t>(size(take<0, 2>(SmemLayoutV{})) * cutlass::sizeof_bits_v<DTypeKV> / 8);

  // Whether use scheduler barrier or hardware warp scheduler, using heuristic based on data type
  // and head dim
  static constexpr bool UseSchedulerBarrier =
      cutlass::sizeof_bits_v<DTypeQ> == 8 ? HEAD_DIM_VO >= 128 : HEAD_DIM_VO <= 128;
  using WarpScheduler = WarpScheduler<Ktraits, UseSchedulerBarrier>;

  // Host side kernel arguments
  struct Arguments {
    DTypeQ const* Q_ptr;
    LayoutT layout_Q;
    DTypeKV const* K_ptr;
    LayoutT layout_K;
    DTypeV const* V_ptr;
    LayoutT layout_V;
    int window_left;
    AdditionalParams additional_params;
  };

  // Device side kernel params
  struct Params {
    LayoutT layout_Q;
    LayoutT layout_K;
    LayoutT layout_V;
    TMA_Q tma_load_Q;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    // Raw FP8 V base pointer for the asymmetric cp.async dequant path (unused for
    // symmetric K/V, which loads V via tma_load_V).
    DTypeV const* V_ptr;
    int window_left;
    AdditionalParams additional_params;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.Q_ptr), args.layout_Q);
    TMA_Q tma_load_Q = make_tma_copy(GmemTiledCopyQ{}, mQ, SmemLayoutQ{},
                                     select<0, 2>(TileShape_QKD{}), _1{});  // no mcast for Q
    Tensor mK = make_tensor(make_gmem_ptr(args.K_ptr), args.layout_K);
    TMA_K tma_load_K = make_tma_copy(GmemTiledCopyKV{}, mK, SmemLayoutK{}(_, _, _0{}),
                                     select<1, 2>(TileShape_QKD{}), _1{});  // no mcast
    // V TMA descriptor: only used for symmetric K/V. For asymmetric K/V, V is FP8
    // and loaded via cp.async (dequantized in the producer), so build the
    // descriptor over a null 16-bit pointer purely so the Params struct is
    // well-formed; it is never used (see load()).
    TMA_V tma_load_V = [&] {
      if constexpr (!IS_ASYM) {
        Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<DTypeKV const*>(args.V_ptr)),
                                args.layout_V);
        return make_tma_copy(GmemTiledCopyKV{}, mV, SmemLayoutV{}(_, _, _0{}),
                             select<2, 1>(TileShape_PDV{}), _1{});  // no mcast
      } else {
        Tensor mV = make_tensor(make_gmem_ptr(static_cast<DTypeKV const*>(nullptr)), args.layout_V);
        return make_tma_copy(GmemTiledCopyKV{}, mV, SmemLayoutV{}(_, _, _0{}),
                             select<2, 1>(TileShape_PDV{}), _1{});
      }
    }();
    return {args.layout_Q, args.layout_K, args.layout_V, tma_load_Q,
            tma_load_K,    tma_load_V,    args.V_ptr,     args.window_left,
            args.additional_params};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_V.get_tma_descriptor());
  }

  CUTLASS_DEVICE
  int get_num_kv_tiles(Params const& mainloop_params, int q_tile_idx, const int qo_len,
                       const int kv_len) {
    static constexpr int CTA_Q = get<0>(TileShape_QKD{});
    static constexpr int CTA_KV = get<1>(TileShape_QKD{});
    int num_kv_tiles = cute::ceil_div(kv_len, CTA_KV);
    if constexpr (CAUSAL) {
      num_kv_tiles = std::min(num_kv_tiles,
                              cute::ceil_div((q_tile_idx + 1) * CTA_Q + kv_len - qo_len, CTA_KV));
    }

    return num_kv_tiles;
  }

  template <bool LEFT_SLIDING_WINDOW, typename BlockCoord, typename Scheduler,
            typename SharedStorage>
  CUTLASS_DEVICE void load(Params const& mainloop_params, MainloopPipelineK pipeline_k,
                           MainloopPipelineV pipeline_v, PipelineState& smem_pipe_write_k,
                           PipelineState& smem_pipe_write_v, SharedStorage& shared_storage,
                           Scheduler& scheduler, typename Scheduler::Params const& scheduler_params,
                           typename Scheduler::WorkTileInfo& work_tile_info,
                           BlockCoord const& block_coord, int work_idx) {
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());
    Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.layout_K.shape());
    Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(mainloop_params.layout_V.shape());

    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;

    // Prepare the TMA loads
    Tensor gQ = get_local_tile_tensor(mQ, select<0, 2>(TileShape_QKD{}), qo_head_idx, qo_indptr,
                                      qo_len)(_, _, q_tile_idx);  // (Q, D)
    Tensor gK = get_local_tile_tensor(mK, select<1, 2>(TileShape_QKD{}), kv_head_idx, kv_indptr,
                                      kv_len);  // (K, D, _)
    Tensor gV = get_local_tile_tensor(mV, select<2, 1>(TileShape_PDV{}), kv_head_idx, kv_indptr,
                                      kv_len);  // (K, D, _)

    Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
    Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
    auto [tQgQ, tQsQ] =
        tma_partition(mainloop_params.tma_load_Q, _0{}, Layout<_1>{}, group_modes<0, 2>(sQ_x),
                      group_modes<0, 2>(gQ_x));  // (TMA), (TMA)
    auto [tKgK, tKsK] =
        tma_partition(mainloop_params.tma_load_K, _0{}, Layout<_1>{}, group_modes<0, 2>(sK),
                      group_modes<0, 2>(gK));  // (TMA, k), (TMA, PIPE)
    auto [tVgV, tVsV] =
        tma_partition(mainloop_params.tma_load_V, _0{}, Layout<_1>{}, group_modes<0, 2>(sV),
                      group_modes<0, 2>(gV));  // (TMA, k), (TMA, PIPE)

    int num_kv_tiles = get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
    int kv_tile_idx = num_kv_tiles - 1;
    int swa_begin_kv_tile_idx = 0;
    if constexpr (LEFT_SLIDING_WINDOW) {
      swa_begin_kv_tile_idx = get_swa_begin_kv_tile_idx<CTA_Q, CTA_KV>(mainloop_params.window_left,
                                                                       q_tile_idx, qo_len, kv_len);
    }

    int lane_predicate = cute::elect_one_sync();

    if constexpr (!IS_ASYM) {
      // ================= Symmetric K/V: K and V both via TMA =================
      if (lane_predicate) {
        pipeline_k.producer_acquire(smem_pipe_write_k);
        copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                             /*mcast_mask=*/0),
             tKgK(_, kv_tile_idx), tKsK(_, smem_pipe_write_k.index()));
        ++smem_pipe_write_k;
      }

      // Wait for the MMA warpgroups to say that smem_q is ready
      cutlass::arch::NamedBarrier::sync(NUM_MMA_THREADS + Ktraits::NUM_PRODUCER_THREADS,
                                        static_cast<int>(NamedBarriers::kQueryEmpty));

      if (lane_predicate) {
        shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
        copy(mainloop_params.tma_load_Q.with(
                 reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                     shared_storage.barrier_Q),
                 /*mcast_mask=*/0),
             tQgQ, tQsQ);
      }

      // Wait for warp 1 to signal that smem_v are ready and V can be copied from gmem
      // Need ClusterBarrier, not just NamedBarrier. Otherwise we might have CTA 0 finishing the
      // TMA store on O first, call TMA multicast load on V, before CTA 1 can finishing TMA store on
      // O.
      shared_storage.barrier_O.wait((work_idx + 1) % 2);

      if (lane_predicate) {
#pragma unroll 2
        for (; kv_tile_idx > swa_begin_kv_tile_idx; --kv_tile_idx) {
          pipeline_k.producer_acquire(smem_pipe_write_k);
          copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                               /*mcast_mask=*/0),
               tKgK(_, kv_tile_idx - 1), tKsK(_, smem_pipe_write_k.index()));
          ++smem_pipe_write_k;
          pipeline_v.producer_acquire(smem_pipe_write_v);
          copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v),
                                               /*mcast_mask=*/0),
               tVgV(_, kv_tile_idx), tVsV(_, smem_pipe_write_v.index()));
          ++smem_pipe_write_v;
        }
      }
      scheduler.prefetch_next_work(scheduler_params, work_tile_info);
      if (lane_predicate) {
        pipeline_v.producer_acquire(smem_pipe_write_v);
        copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v),
                                             /*mcast_mask=*/0),
             tVgV(_, kv_tile_idx), tVsV(_, smem_pipe_write_v.index()));
        ++smem_pipe_write_v;
      }
    } else {
      // ============= Asymmetric K/V: K via TMA, V via cp.async dequant =========
      // K keeps its single-warp TMA fast path; V (FP8) is loaded and dequantized
      // to 16-bit by the full producer warpgroup, mirroring the paged path but
      // with contiguous (ragged/single) addressing instead of page-table gather.
      int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
      bool is_leader = (warp_idx_in_warpgroup == 0) && lane_predicate;  // one thread: K/Q TMA

      // V dequant setup: contiguous FP8 base (fold in the sequence and head
      // offsets), plus the SIMT partition of the 16-bit smem V tile.
      int64_t v_stride_n = stride<0>(mainloop_params.layout_V);
      int64_t v_stride_h = stride<2>(mainloop_params.layout_V);
      DTypeV const* V_base = mainloop_params.V_ptr +
                             static_cast<int64_t>(kv_indptr) * v_stride_n +
                             static_cast<int64_t>(kv_head_idx) * v_stride_h;
      constexpr int HEAD_DIM_VO_c = get<1>(TileShape_PDV{});
      Tensor gV_virt = make_tensor(make_gmem_ptr(static_cast<DTypeKV*>(nullptr)),
                                   make_shape(CTA_KV, HEAD_DIM_VO_c),
                                   make_stride(HEAD_DIM_VO_c, _1{}));  // (KV, D) col-major
      Tensor gV_tiled = local_tile(gV_virt, select<2, 1>(TileShape_PDV{}), make_coord(_, _0{}));
      Tensor cV = cute::make_identity_tensor(gV_tiled.shape());
      GmemTiledCopyV gmem_tiled_copy_v;
      auto gmem_thr_copy_v = gmem_tiled_copy_v.get_slice(threadIdx.x);
      Tensor tVsV_dq = gmem_thr_copy_v.partition_D(sV);  // (CPY, CPY_KV, CPY_D, PIPE)
      Tensor tVcV_dq = gmem_thr_copy_v.partition_D(cV);  // (CPY, CPY_KV, CPY_D, kv)

      auto load_v_dequant = [&](int tile, int stage_idx, bool use_predicate) {
        using VecOut = AlignmentTypeKV;                            // 16 bytes == 8 x DTypeKV
        constexpr int VecSize = sizeof(VecOut) / sizeof(DTypeKV);  // 8 elements per vector
        int kv_base_idx = tile * CTA_KV;
        auto dst = recast<VecOut>(flatten(tVsV_dq(_, _, _, stage_idx)));  // 16-bit smem dest
        auto c = flatten(tVcV_dq(_, _, _, tile));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(dst); ++i) {
          auto coord = c(VecSize * i);
          int kv_offset = get<0>(coord);
          int d_idx = get<1>(coord);
          int kv_idx = kv_base_idx + kv_offset;
          bool guard = !use_predicate || kv_idx < kv_len;
          int64_t off = static_cast<int64_t>(kv_idx) * v_stride_n + static_cast<int64_t>(d_idx);
          // Load VecSize FP8 elements (== 8 bytes) as one aligned uint2; zero-fill
          // out-of-bound rows (softmax masks them out anyway; zero avoids NaN*0).
          uint2 packed = make_uint2(0u, 0u);
          if (guard) {
            packed = *reinterpret_cast<const uint2*>(V_base + off);
          }
          const DTypeV* src_reg = reinterpret_cast<const DTypeV*>(&packed);
          VecOut out_vec;
          DTypeKV* dst_reg = reinterpret_cast<DTypeKV*>(&out_vec);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < VecSize; ++j) {
            dst_reg[j] = static_cast<DTypeKV>(static_cast<float>(src_reg[j]));
          }
          dst(i) = out_vec;
        }
      };

      // load last k-tile (TMA, single thread)
      if (is_leader) {
        pipeline_k.producer_acquire(smem_pipe_write_k);
        copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                             /*mcast_mask=*/0),
             tKgK(_, kv_tile_idx), tKsK(_, smem_pipe_write_k.index()));
        ++smem_pipe_write_k;
      }

      // All producer threads sync on kQueryEmpty before loading Q
      cutlass::arch::NamedBarrier::sync(NUM_MMA_THREADS + Ktraits::NUM_PRODUCER_THREADS,
                                        static_cast<int>(NamedBarriers::kQueryEmpty));

      if (is_leader) {
        shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
        copy(mainloop_params.tma_load_Q.with(
                 reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                     shared_storage.barrier_Q),
                 /*mcast_mask=*/0),
             tQgQ, tQsQ);
      }

      shared_storage.barrier_O.wait((work_idx + 1) % 2);

      // Main loop: K(kv_tile_idx-1) via TMA (leader), V(kv_tile_idx) via full-WG
      // dequant. The V producer_commit is a plain (non-cp.async) barrier arrive:
      // the STS above are ordinary stores, released by the mbarrier arrive.
#pragma unroll 2
      for (; kv_tile_idx > swa_begin_kv_tile_idx; --kv_tile_idx) {
        if (is_leader) {
          pipeline_k.producer_acquire(smem_pipe_write_k);
          copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                               /*mcast_mask=*/0),
               tKgK(_, kv_tile_idx - 1), tKsK(_, smem_pipe_write_k.index()));
          ++smem_pipe_write_k;
        }
        pipeline_v.producer_acquire(smem_pipe_write_v);
        load_v_dequant(kv_tile_idx, smem_pipe_write_v.index(), /*use_predicate=*/true);
        pipeline_v.producer_commit(smem_pipe_write_v);
        ++smem_pipe_write_v;
      }
      scheduler.prefetch_next_work(scheduler_params, work_tile_info);
      // load first v tile
      pipeline_v.producer_acquire(smem_pipe_write_v);
      load_v_dequant(kv_tile_idx, smem_pipe_write_v.index(), /*use_predicate=*/true);
      pipeline_v.producer_commit(smem_pipe_write_v);
      ++smem_pipe_write_v;
    }
    scheduler.broadcast_next_work(work_tile_info);
  }

  CUTLASS_DEVICE void load_tail(MainloopPipelineK pipeline_k, MainloopPipelineV pipeline_v,
                                PipelineState& smem_pipe_write_k,
                                PipelineState& smem_pipe_write_v) {
    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {
      pipeline_k.producer_tail(smem_pipe_write_k);
      if constexpr (!IS_ASYM) {
        pipeline_v.producer_tail(smem_pipe_write_v);
      }
    }
    if constexpr (IS_ASYM) {
      // V is a cp.async (PipelineAsync) pipeline: its producer_tail must be called
      // by the whole producer warpgroup, not just the leader thread.
      pipeline_v.producer_tail(smem_pipe_write_v);
    }
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_MAINLOOP_CUH_
