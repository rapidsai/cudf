/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Christian Noboa Mardini <christian@blazingdb.com>
 *     Copyright 2019 William Scott Malpica <william@blazingdb.com>
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
#include "slice.hpp"
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/error_utils.hpp>
#include <rmm/thrust_rmm_allocator.h>

namespace cudf {

std::vector<gdf_column*> split(gdf_column const &         input_column,
                               gdf_index_type const*      indices,
                               gdf_size_type              num_indices) {

    if (num_indices == 0 || indices == nullptr){
      return std::vector<gdf_column*>();
    } else {
      // Get indexes on host side
      std::vector<gdf_size_type> host_indices(num_indices);
      CUDA_TRY( cudaMemcpy(host_indices.data(), indices, num_indices * sizeof(gdf_size_type), cudaMemcpyDeviceToHost) );

      // Convert to slice indices
      std::vector<gdf_size_type> host_slice_indices((num_indices + 1) * 2);
      host_slice_indices[0] = 0;
      for (gdf_size_type i = 0; i < num_indices; i++){
        host_slice_indices[2*i + 1] = host_indices[i];
        host_slice_indices[2*i + 2] = host_indices[i];
      }
      host_slice_indices[host_slice_indices.size()-1] = input_column.size;
      rmm::device_vector<gdf_index_type> slice_indices = host_slice_indices; // copy to device happens automatically
      gdf_size_type slice_num_indices = slice_indices.size();

      return cudf::detail::slice(input_column, slice_indices.data().get(), slice_num_indices);
    }
}

} // namespace cudf
