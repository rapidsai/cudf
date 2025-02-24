/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/climits>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cstddef>

// clang-format off
#include "transform/jit/operation-udf.hpp"
// clang-format on

namespace cudf {
namespace transformation {
namespace jit {

using scale_rep = cuda::std::underlying_type_t<numeric::scale_type>;

/// @brief This class supports striding into columns of data as either scalars or actual
/// columns at no runtime cost. Although it implies the kernel will be recompiled if scalar and
/// column inputs are interchanged.
template <int32_t id, int32_t multiplier, typename T>
struct strided {
  using type = T;
  using rep  = T;

  T data;

  __device__ T get(int64_t index, scale_rep const* __restrict__ scales) const
  {
    return (&data)[index * multiplier];
  }

  __device__ T const& get_rep(int64_t index) const { return (&data)[index * multiplier]; }

  __device__ T& get_rep(int64_t index) { return (&data)[index * multiplier]; }
};

template <int32_t id, int32_t multiplier, typename Rep, numeric::Radix rad>
struct strided<id, multiplier, numeric::fixed_point<Rep, rad>> {
  using type = numeric::fixed_point<Rep, rad>;
  using rep  = Rep;

  rep data;

  __device__ type get(int64_t index, scale_rep const* __restrict__ scales) const
  {
    return type{
      numeric::scaled_integer<rep>{(&data)[index * multiplier], numeric::scale_type{scales[id]}}};
  }

  __device__ rep const& get_rep(int64_t index) const { return (&data)[index * multiplier]; }

  __device__ rep& get_rep(int64_t index) { return (&data)[index * multiplier]; }
};

template <typename Out, typename... In>
CUDF_KERNEL void kernel(cudf::size_type size,
                        scale_rep const* __restrict__ scales,
                        Out* __restrict__ out,
                        In const* __restrict__... ins)
{
  // cannot use global_thread_id utility due to a JIT build issue by including
  // the `cudf/detail/utilities/cuda.cuh` header
  auto const block_size          = static_cast<thread_index_type>(blockDim.x);
  thread_index_type const start  = threadIdx.x + blockIdx.x * block_size;
  thread_index_type const stride = block_size * gridDim.x;

  for (auto i = start; i < static_cast<thread_index_type>(size); i += stride) {
    GENERIC_TRANSFORM_OP(&out->get_rep(i), ins->get(i, scales)...);
  }
}

template <typename Out, typename... In>
CUDF_KERNEL void fixed_point_kernel(cudf::size_type size,
                                    scale_rep const* __restrict__ scales,
                                    Out* __restrict__ out,
                                    In const* __restrict__... ins)
{
  // cannot use global_thread_id utility due to a JIT build issue by including
  // the `cudf/detail/utilities/cuda.cuh` header
  auto const block_size          = static_cast<thread_index_type>(blockDim.x);
  thread_index_type const start  = threadIdx.x + blockIdx.x * block_size;
  thread_index_type const stride = block_size * gridDim.x;

  for (auto i = start; i < static_cast<thread_index_type>(size); i += stride) {
    typename Out::type value{
      numeric::scaled_integer<typename Out::rep>{0, numeric::scale_type{scales[0]}}};
    GENERIC_TRANSFORM_OP(&value, ins->get(i, scales)...);
    out->get_rep(i) = value.value();
  }
}

}  // namespace jit
}  // namespace transformation
}  // namespace cudf
