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
#include <cudf/utilities/bit.hpp>

#include <tuple>
#include <cuda/std/tuple>

namespace cudf {
namespace transformation {
namespace jit {

struct Masked {
  int64_t value;
  bool valid;
};
/*
template <typename TypeOut, typename TypeLhs, typename TypeRhs>
__global__
void binop_kernel(cudf::size_type size,
                  TypeOut* out_data, 
                  TypeLhs* lhs_data,
                  TypeRhs* rhs_data,
                  bool* out_mask,
                  cudf::bitmask_type const* lhs_mask,
                  cudf::size_type lhs_offset,
                  cudf::bitmask_type const* rhs_mask,
                  cudf::size_type rhs_offset
) {
    int tid = threadIdx.x;
    int blkid = blockIdx.x;
    int blksz = blockDim.x;
    int gridsz = gridDim.x;
    int start = tid + blkid * blksz;
    int step = blksz * gridsz;

    Masked output;
    char l_valid;
    char r_valid;
    long int l_data;
    long int r_data;

    for (cudf::size_type i=start; i<size; i+=step) {
      l_valid = lhs_mask ? cudf::bit_is_set(lhs_mask, lhs_offset + i) : true;
      r_valid = rhs_mask ? cudf::bit_is_set(rhs_mask, rhs_offset + i) : true;
      l_data = lhs_data[i];
      r_data = rhs_data[i];
      GENERIC_BINARY_OP(&output.value, lhs_data[i], l_valid, rhs_data[i], r_valid);
      out_data[i] = output.value;
      out_mask[i] = output.valid;
    }
}
*/
template <typename TypeIn, typename MaskType, typename OffsetType>
__device__ auto make_args(cudf::size_type id,
                          TypeIn in_ptr,
                          MaskType in_mask,
                          OffsetType in_offset) 
{
    bool valid = in_mask ? cudf::bit_is_set(in_mask, in_offset + id) : true;
    return cuda::std::make_tuple(in_ptr[id], valid);
}

template <typename InType, typename MaskType, typename OffsetType, typename ... Arguments>
__device__ auto make_args(cudf::size_type id, 
                          InType in_ptr, 
                          MaskType in_mask,     // in practice, always cudf::bitmask_type const* 
                          OffsetType in_offset,  // in practice, always cudf::size_type
                          Arguments ... args) {

    bool valid = in_mask ? cudf::bit_is_set(in_mask, in_offset + id) : true;
    return cuda::std::tuple_cat(
        cuda::std::make_tuple(in_ptr[id], valid),
        make_args(id, args...)
    );
}

template <typename TypeOut, typename ... Arguments>
__global__
void generic_udf_kernel(cudf::size_type size, 
                        TypeOut* out_data, 
                        bool* out_mask, 
                        Arguments ... args)
{   

    int tid = threadIdx.x;
    int blkid = blockIdx.x;
    int blksz = blockDim.x;
    int gridsz = gridDim.x;
    int start = tid + blkid * blksz;
    int step = blksz * gridsz;

    Masked output;
    for (cudf::size_type i=start; i<size; i+=step) {
      auto func_args = cuda::std::tuple_cat(
          cuda::std::make_tuple(&output.value),
          make_args(i, args...) // passed int64*, bool*, int64, int64*, bool*, int64
      );
      cuda::std::apply(GENERIC_OP, func_args);
      out_data[i] = output.value;
      out_mask[i] = output.valid;
    }

}


}  // namespace jit
}  // namespace transformation
}  // namespace cudf
