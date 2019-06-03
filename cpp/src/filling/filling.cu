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