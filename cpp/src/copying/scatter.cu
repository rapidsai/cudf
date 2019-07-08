/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/cudf.h>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/cudf_utils.h>
#include <cudf/table.hpp>

#include <copying/gather.hpp>

namespace cudf {
namespace detail {
  
__global__ void invert_map(gdf_index_type gather_map[], const gdf_size_type destination_rows,
                            gdf_index_type const scatter_map[], const gdf_size_type source_rows){
  gdf_index_type source_row = threadIdx.x + blockIdx.x * blockDim.x;
  if(source_row < source_rows){
    gdf_index_type destination_row = scatter_map[source_row];
    if(destination_row < destination_rows){
      gather_map[destination_row] = source_row;
    }
  }
}

void scatter(table const* source_table, gdf_index_type const scatter_map[],
            table* destination_table) {
  const gdf_size_type num_source_rows = source_table->num_rows();
  const gdf_size_type num_destination_rows = destination_table->num_rows();
  // Turn the scatter_map[] into a gather_map[] and then call gather(...).
  // We are initializing the result gather_map with `num_destination_rows`
  // so if at the end the value is not modified we know the original 
  // scatter map does not map to this row, and we should keep whatever is 
  // there originally
  rmm::device_vector<gdf_index_type> v_gather_map(num_destination_rows, num_destination_rows);
 
  constexpr int block_size = 256;

  const gdf_size_type invert_grid_size =
    (destination_table->num_rows() + block_size - 1) / block_size;

  detail::invert_map<<<invert_grid_size, block_size>>>(v_gather_map.data().get(), num_destination_rows, scatter_map, num_source_rows);
  
  // We want to check bounds for scatter since it is possible that
  // some elements of the destination column are not modified. 
  detail::gather(source_table, v_gather_map.data().get(), destination_table, true);    
}

}  // namespace detail

void scatter(table const* source_table, gdf_index_type const scatter_map[],
             table* destination_table) {
  detail::scatter(source_table, scatter_map, destination_table);
}

}  // namespace cudf
