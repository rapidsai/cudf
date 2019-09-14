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
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <cudf/copying.hpp>

#include <cudf/cudf.h>
#include <cudf/types.h>
#include <rmm/thrust_rmm_allocator.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>

template <typename T>
void print(rmm::device_vector<T> const& d_vec, std::string label = "") {
  thrust::host_vector<T> h_vec = d_vec;
  printf("%s \t", label.c_str());
  for (auto &&i : h_vec)  std::cout << i << " ";
  printf("\n");
}

struct printer
{
  template <typename T>
  void operator()(gdf_column const& col, std::string label = "") {
    auto col_data = reinterpret_cast<T*>(col.data);
    auto d_vec = rmm::device_vector<T>(col_data, col_data+col.size);
    print(d_vec, label);
  }
};

void print(gdf_column const& col, std::string label = "") {
  cudf::type_dispatcher(col.dtype, printer{}, col, label);
}

namespace cudf {

namespace detail {

gdf_column repeat(const gdf_column &in, const gdf_column& count, cudaStream_t stream = 0) {
  CUDF_EXPECTS(count.dtype == gdf_dtype_of<gdf_size_type>(),
    "Count column should be of index type");
  CUDF_EXPECTS(count.null_count == 0, "Couns cannot be null");
  
  auto exec_policy = rmm::exec_policy(stream)->on(stream);
  rmm::device_vector<gdf_size_type> offset(count.size);
  auto count_data = static_cast <gdf_size_type*> (count.data);
  
  thrust::inclusive_scan(exec_policy, count_data, count_data + count.size, offset.begin());
  // print(offset);

  gdf_size_type output_size = offset[count.size - 1];

  rmm::device_vector<gdf_size_type> indices(output_size);
  thrust::upper_bound(exec_policy,
                      offset.begin(), offset.end(),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(output_size),
                      indices.begin());
  // print(indices);

  // Allocate `output` with output_size elements
  gdf_column output = cudf::allocate_like(in, output_size);

  cudf::table in_table{const_cast<gdf_column*>(&in)};
  cudf::table out_table{&output};
  cudf::gather(&in_table, indices.data().get(), &out_table);
  // print(output);

  return output;
}

} // namespace detail


gdf_column repeat(const gdf_column &in, const gdf_column& count) {
  return detail::repeat(in, count);
}

} // namespace cudf
