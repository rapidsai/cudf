/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

// Include Jitify's cstddef header first
#include <cstddef>

#include <cuda/std/climits>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <transform/jit/operation-udf.hpp>

#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

namespace cudf {
namespace transformation {
namespace jit {

template <typename TypeOut, typename TypeIn>
__global__ void kernel(cudf::size_type size, TypeOut* out_data, TypeIn* in_data)
{
  int tid    = threadIdx.x;
  int blkid  = blockIdx.x;
  int blksz  = blockDim.x;
  int gridsz = gridDim.x;

  int start = tid + blkid * blksz;
  int step  = blksz * gridsz;

  for (cudf::size_type i = start; i < size; i += step) {
    GENERIC_UNARY_OP(&out_data[i], in_data[i]);
  }
}

}  // namespace jit
}  // namespace transformation
}  // namespace cudf
