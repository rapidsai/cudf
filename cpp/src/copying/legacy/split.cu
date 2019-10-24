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
#include <cudf/utilities/error.hpp>
#include <rmm/thrust_rmm_allocator.h>

namespace cudf {


rmm::device_vector<cudf::size_type> splits_to_slice_indices(
                               cudf::size_type const*  splits,
                               cudf::size_type const    num_splits,
                               cudf::size_type const   split_end) {
    rmm::device_vector<cudf::size_type> slice_indices((num_splits + 1) * 2);
    slice_indices[0] = 0;
    slice_indices[slice_indices.size()-1] = split_end;
    thrust::tabulate( slice_indices.begin()+1,
        slice_indices.end()-1,
        [splits] __device__ (auto i) { return splits[i/2]; });
    return slice_indices;
}

std::vector<gdf_column*> split(gdf_column const &         input_column,
                               cudf::size_type const*      splits,
                               cudf::size_type              num_splits) {

    if (num_splits == 0 || splits== nullptr){
      return std::vector<gdf_column*>();
    } else {
      rmm::device_vector<cudf::size_type> slice_indices =
        splits_to_slice_indices(splits, num_splits, input_column.size); 
      return cudf::detail::slice(input_column, slice_indices.data().get(),
          slice_indices.size());
    }
}

} // namespace cudf
