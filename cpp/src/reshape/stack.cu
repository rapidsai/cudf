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

#include <copying/scatter.hpp>
#include <utilities/column_utils.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <nvstrings/NVCategory.h>

#include <algorithm>

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

#include <cudf/utilities/legacy/type_dispatcher.hpp>
void print(gdf_column const& col, std::string label = "") {
  cudf::type_dispatcher(col.dtype, printer{}, col, label);
}

namespace cudf {

namespace detail {
  
gdf_column stack(const cudf::table &values, cudaStream_t stream = 0)
{
  gdf_dtype dtype = values.get_column(0)->dtype;
  gdf_size_type num_cols = values.num_columns();
  gdf_size_type num_rows = values.num_rows();

  for (auto &&col : values) {
    CUDF_EXPECTS(col->dtype == dtype, "All columns must have the same type");
  }

  bool input_is_nullable = std::any_of(values.begin(), values.end(),
    [](gdf_column const* col){ return is_nullable(*col); });

  if (values.num_rows() == 0) {
    return cudf::allocate_column(dtype, 0, input_is_nullable);
  }

  // Allocate output
  gdf_column output = allocate_like(*values.get_column(0),
                                    num_cols * num_rows);

  if (dtype == GDF_STRING_CATEGORY) {
    auto categories = static_cast<NVCategory *>(values.get_column(0)->dtype_info.category);
    // We need to initialize data in case of string category because 
    CUDA_TRY( cudaMemset(output.data, 0, output.size * sizeof(cudf::nvstring_category)) );
    output.dtype_info.category = 
      categories->gather(static_cast<int*>(output.data), output.size);
  }

  // Allocate scatter map
  rmm::device_vector<gdf_size_type> scatter_map(values.num_rows());
  auto counting_it = thrust::make_counting_iterator(0);
  auto strided_it = thrust::make_transform_iterator(counting_it,
    [num_cols] __device__ (auto i){ return num_cols * i; });
  thrust::copy(strided_it, strided_it + num_rows, scatter_map.begin());

  cudf::table output_table{&output};
  for (auto &&col : values) {
    cudf::table single_col_table = { const_cast<gdf_column*>(col) };
    detail::scatter(&single_col_table, scatter_map.data().get(), &output_table);
    thrust::transform(scatter_map.begin(), scatter_map.end(), scatter_map.begin(),
      [] __device__ (auto i) { return ++i; });
  }

  return output;
}

} // namespace detail

gdf_column stack(const cudf::table &values) {
  return detail::stack(values);
}

} // namespace cudf