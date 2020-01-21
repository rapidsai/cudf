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

#include <copying/legacy/scatter.hpp>
#include <utilities/legacy/column_utils.hpp>
#include <cudf/legacy/copying.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <nvstrings/NVCategory.h>

#include <algorithm>

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
  // This needs to be done because the output is unnamed in pandas
  free(output.col_name);
  output.col_name = nullptr;

  // PLAN:
  // Sync column categories if they were GDF_STRING_CATEGORY and convert temporary 
  // columns to GDF_INT32 (using gdf_dtype_of). Then normal path till after scatter.
  // Finally, do a NVCategory->gather on the result column.

  std::vector<gdf_column *> temp_values;
  if (dtype == GDF_STRING_CATEGORY) {
    std::transform(values.begin(), values.end(), std::back_inserter(temp_values),
      [] (const gdf_column *c) { return new gdf_column(allocate_like(*c)); } );

    sync_column_categories(values.begin(), temp_values.data(), values.num_columns());

    std::for_each(temp_values.begin(), temp_values.end(),
      [] (gdf_column* c) { c->dtype = gdf_dtype_of<gdf_nvstring_category>(); });
    
    output.dtype = gdf_dtype_of<gdf_nvstring_category>();
  } else {
    std::transform(values.begin(), values.end(), std::back_inserter(temp_values),
      [] (const gdf_column *c) { return const_cast<gdf_column *>(c); } );
  }

  // Allocate scatter map
  rmm::device_vector<gdf_size_type> scatter_map(values.num_rows());
  auto counting_it = thrust::make_counting_iterator(0);
  auto strided_it = thrust::make_transform_iterator(counting_it,
    [num_cols] __device__ (auto i){ return num_cols * i; });
  thrust::copy(strided_it, strided_it + num_rows, scatter_map.begin());

  cudf::table output_table{&output};
  for (auto &&col : temp_values) {
    cudf::table single_col_table = { const_cast<gdf_column*>(col) };
    detail::scatter(&single_col_table, scatter_map.data().get(), &output_table);
    thrust::transform(scatter_map.begin(), scatter_map.end(), scatter_map.begin(),
      [] __device__ (auto i) { return ++i; });
  }

  if (dtype == GDF_STRING_CATEGORY)
  {
    output.dtype = GDF_STRING_CATEGORY;
    nvcategory_gather(&output, static_cast<NVCategory*>(temp_values[0]->dtype_info.category));
    std::for_each(temp_values.begin(), temp_values.end(),
      [] (gdf_column* c) { gdf_column_free(c); });
  }

  return output;
}

} // namespace detail

gdf_column stack(const cudf::table &values) {
  return detail::stack(values);
}

} // namespace cudf
