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

#include <utilities/error_utils.hpp>
#include <utilities/column_utils.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <cudf/copying.hpp>

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

cudf::table repeat(const cudf::table &in, const gdf_column& count, cudaStream_t stream = 0) {
  CUDF_EXPECTS(count.dtype == gdf_dtype_of<gdf_size_type>(),
    "Count column should be of index type");
  CUDF_EXPECTS(in.num_rows() == count.size, "in and count must have equal size");
  CUDF_EXPECTS(not has_nulls(count), "count cannot contain nulls");

  if (in.num_rows() == 0) {
    return cudf::empty_like(in);
  }
  
  auto exec_policy = rmm::exec_policy(stream)->on(stream);
  rmm::device_vector<gdf_size_type> offset(count.size);
  auto count_data = static_cast <gdf_size_type*> (count.data);
  
  thrust::inclusive_scan(exec_policy, count_data, count_data + count.size, offset.begin());

  gdf_size_type output_size = offset.back();

  rmm::device_vector<gdf_size_type> indices(output_size);
  thrust::upper_bound(exec_policy,
                      offset.begin(), offset.end(),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(output_size),
                      indices.begin());

  cudf::table output = cudf::allocate_like(in, output_size, RETAIN, stream);

  cudf::gather(&in, indices.data().get(), &output);

  return output;
}

cudf::table repeat(const cudf::table &in, const gdf_scalar& count, cudaStream_t stream = 0) {
  CUDF_EXPECTS(count.dtype == gdf_dtype_of<gdf_size_type>(),
    "Count value should be of index type");
  CUDF_EXPECTS(count.is_valid, "count cannot be null");

  if (in.num_rows() == 0) {
    return cudf::empty_like(in);
  }
  
  gdf_size_type stride = count.data.si32;

  gdf_size_type output_size = stride * in.num_rows();
  auto offset = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    [stride] __device__ (auto i) { return (i+1) * stride; }
  );

  rmm::device_vector<gdf_size_type> indices(output_size);
  thrust::upper_bound(rmm::exec_policy(stream)->on(stream),
                      offset, offset + in.num_rows(),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(output_size),
                      indices.begin());

  cudf::table output = cudf::allocate_like(in, output_size, RETAIN, stream);

  cudf::gather(&in, indices.data().get(), &output);

  return output;
}

} // namespace detail


cudf::table repeat(const cudf::table &in, const gdf_column& count) {
  return detail::repeat(in, count);
}

cudf::table repeat(const cudf::table &in, const gdf_scalar& count) {
  return detail::repeat(in, count);
}

} // namespace cudf
