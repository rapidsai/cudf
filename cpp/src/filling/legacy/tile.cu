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

#include <cudf/utilities/error.hpp>
#include <cudf/legacy/copying.hpp>

#include <cudf/cudf.h>
#include <cudf/types.h>
#include <rmm/thrust_rmm_allocator.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>

namespace cudf {

namespace detail {

cudf::table tile(const cudf::table &in, gdf_size_type count, 
                 cudaStream_t stream = 0)
{
  CUDF_EXPECTS(count >= 0, "Count cannot be negative");

  gdf_size_type num_rows = in.num_rows();

  if (num_rows == 0 or count == 0) {
    return cudf::empty_like(in);
  }
  

  gdf_size_type out_num_rows = count * num_rows;

  // make tile iterator
  auto counting_it = thrust::make_counting_iterator(0);
  auto tiled_it = thrust::make_transform_iterator(counting_it, 
    [=] __device__ (auto i) -> gdf_size_type { return i % num_rows; });

  // make gather map
  rmm::device_vector<gdf_size_type> gather_map(out_num_rows);
  thrust::copy(rmm::exec_policy(stream)->on(stream), tiled_it, tiled_it + out_num_rows, gather_map.begin());

  // Allocate `output` with out_num_rows elements
  cudf::table output = cudf::allocate_like(in, out_num_rows);
  cudf::gather(&in, gather_map.data().get(), &output);

  return output;
}

} // namespace detail


cudf::table tile(const cudf::table &in, gdf_size_type count) {
  return detail::tile(in, count);
}

} // namespace cudf
