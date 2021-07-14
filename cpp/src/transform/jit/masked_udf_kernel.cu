/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cstddef>
#include <cstdint>
#include <transform/jit/operation-udf.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <cuda/std/climits>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

namespace cudf {
namespace transformation {
namespace jit {

template <typename T>
struct Masked {
  T value;
  bool valid;
};

template <typename TypeIn, typename MaskType, typename OffsetType>
__device__ auto make_args(cudf::size_type id, TypeIn in_ptr, MaskType in_mask, OffsetType in_offset)
{
  bool valid = in_mask ? cudf::bit_is_set(in_mask, in_offset + id) : true;
  return cuda::std::make_tuple(in_ptr[id], valid);
}

template <typename InType, typename MaskType, typename OffsetType, typename... Arguments>
__device__ auto make_args(cudf::size_type id,
                          InType in_ptr,
                          MaskType in_mask,      // in practice, always cudf::bitmask_type const*
                          OffsetType in_offset,  // in practice, always cudf::size_type
                          Arguments... args)
{
  bool valid = in_mask ? cudf::bit_is_set(in_mask, in_offset + id) : true;
  return cuda::std::tuple_cat(cuda::std::make_tuple(in_ptr[id], valid), make_args(id, args...));
}

template <typename TypeOut, typename... Arguments>
__global__ void generic_udf_kernel(cudf::size_type size,
                                   TypeOut* out_data,
                                   bool* out_mask,
                                   Arguments... args)
{
  int const tid    = threadIdx.x;
  int const blkid  = blockIdx.x;
  int const blksz  = blockDim.x;
  int const gridsz = gridDim.x;
  int const start  = tid + blkid * blksz;
  int const step   = blksz * gridsz;

  Masked<TypeOut> output;
  for (cudf::size_type i = start; i < size; i += step) {
    auto func_args = cuda::std::tuple_cat(
      cuda::std::make_tuple(&output.value),
      make_args(i, args...)  // passed int64*, bool*, int64, int64*, bool*, int64
    );
    cuda::std::apply(GENERIC_OP, func_args);
    out_data[i] = output.value;
    out_mask[i] = output.valid;
  }
}

}  // namespace jit
}  // namespace transformation
}  // namespace cudf
