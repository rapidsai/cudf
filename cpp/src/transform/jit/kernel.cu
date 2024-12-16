/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

template <typename TypeOut, typename TypeIn>
CUDF_KERNEL void kernel(cudf::size_type size, TypeOut* out_data, TypeIn* in_data)
{
  // cannot use global_thread_id utility due to a JIT build issue by including
  // the `cudf/detail/utilities/cuda.cuh` header
  auto const block_size          = static_cast<thread_index_type>(blockDim.x);
  thread_index_type const start  = threadIdx.x + blockIdx.x * block_size;
  thread_index_type const stride = block_size * gridDim.x;

  for (auto i = start; i < static_cast<thread_index_type>(size); i += stride) {
    GENERIC_UNARY_OP(&out_data[i], in_data[i]);
  }
}

}  // namespace jit
}  // namespace transformation
}  // namespace cudf
