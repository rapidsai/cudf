/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <filling.hpp>
#include <utilities/error_utils.hpp>
#include <utilities/type_dispatcher.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/fill.h>

namespace {

template <typename T>
__global__ 
void fill_kernel(T *data, bit_mask::bit_mask_t *bitmask, 
                 gdf_size_type *null_count,
                 gdf_index_type begin, gdf_index_type end,
                 T value, bool value_is_valid)
{
  gdf_index_type tid = threadIdx.x + blockIdx.x * blockDim.x;

  int index = tid + begin;

  while (index < end) {
    data[index] = value;

    // bitmask cases:
    // 1. If only a single value in this word is set
    //    read old value, set new value, null_count += (int{new_is_null}-int{old_is_null})
    // 2. If range completely covers this bitmask word
    //    read old word, count nulls. new_null_count = value_is_valid ? 32 : 0
    //    Set new bitmask word to all 0 or all 1
    //    null_count += (int{new_null_count}-int{old_null_count})
    // 3. If range partially covers this bitmask word
    //    read old word, mask it. count nulls. 
    //    set new bitmask word to all 0 or all 1 masked by range. Count bits.
    //    null_count += (int{new_null_count}-int{old_null_count});

    index += blockDim.x * gridDim.x;
  }
}

struct fill_dispatch {
  template <typename T>
  void operator()(gdf_column *column, gdf_scalar const& value, 
                  gdf_index_type begin, gdf_index_type end,
                  cudaStream_t stream = 0)
  {
    T * __restrict__ data = static_cast<T*>(column->data);
    T const val = *reinterpret_cast<T const*>(&value.data);
    thrust::fill(rmm::exec_policy(stream)->on(stream),
                 data + begin, data + end, val);
    CHECK_STREAM(stream);

    // now set nulls
  }
};

}; // namespace anonymous

namespace cudf {

void fill(gdf_column *column, gdf_scalar const& value, 
          gdf_index_type begin, gdf_index_type end)
{
  CUDF_EXPECTS(column != nullptr, "Column is null");
  CUDF_EXPECTS(column->data != nullptr, "Data pointer is null");
  CUDF_EXPECTS(column->dtype == value.dtype, "Data type mismatch");

  cudf::type_dispatcher(column->dtype,
                        fill_dispatch{},
                        column, value, begin, end);
}

}; // namespace cudf