/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_ATTENTION_POST_SCALE_CUH_
#define FLASHINFER_ATTENTION_POST_SCALE_CUH_
namespace flashinfer {
template <typename T>
__host__ __device__ auto post_scale_value(const T& p, int) -> decltype(p.post_scale) {
  return p.post_scale;
}
template <typename T>
__host__ __device__ auto post_scale_value(const T& p, long)
    -> decltype(p.additional_params.post_scale) {
  return p.additional_params.post_scale;
}
template <typename T>
__host__ __device__ float post_scale_value(const T&, ...) {
  return 1.f;
}
template <typename T>
__host__ __device__ float get_post_scale(const T& p) {
  return post_scale_value(p, 0);
}
template <typename Out>
__device__ __forceinline__ Out rounded_post_scale(float value, float scale) {
  // Preserve the existing store-to-output rounding before the scale multiply.
  Out rounded = static_cast<Out>(value);
  return static_cast<Out>(static_cast<float>(rounded) * scale);
}
}  // namespace flashinfer
#endif
