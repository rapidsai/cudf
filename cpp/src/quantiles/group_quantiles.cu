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

#include <quantiles/quantiles.hpp>
#include <utilities/type_dispatcher.hpp>

#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <cudf/groupby.hpp>
#include <cudf/legacy/column.hpp>

#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>

namespace cudf {

namespace {

struct quantiles_functor {

  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, void >
  operator()(gdf_column const& values_col, gdf_column const& group_indices,
             gdf_column& quants_col, double quantile,
             gdf_quantile_method interpolation)
  {
    // prepare args to be used by lambda below
    auto quants = reinterpret_cast<double*>(quants_col.data);
    auto values = reinterpret_cast<T*>(values_col.data);
    auto grp_id = reinterpret_cast<gdf_size_type*>(group_indices.data);
    auto num_vals = values_col.size;
    auto num_grps = group_indices.size;

    // For each group, calculate quantile
    thrust::for_each_n(thrust::device,
      thrust::make_counting_iterator(0),
      num_grps,
      [=] __device__ (gdf_size_type i) {
        gdf_size_type upper_limit = (i < num_grps-1)
                                  ? grp_id[i + 1]
                                  : num_vals;
        gdf_size_type segment_size = (upper_limit - grp_id[i]);
        quants[i] = detail::select_quantile(values + grp_id[i], segment_size,
                                            quantile, interpolation);
      }
    );
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_arithmetic<T>::value, void >
  operator()(Args&&... args) {
    CUDF_FAIL("Only arithmetic types are supported in quantile");
  }
};

} // namespace anonymous

namespace detail {

// TODO: optimize this so that it doesn't have to generate the sorted table
auto group_values_and_indices(cudf::table const& input_table,
                              gdf_context const& context)
{
  // Sort and groupby the input table
  cudf::table grouped_table;
  gdf_column discard_indices;
  std::vector<gdf_index_type> key_col_indices(input_table.num_columns());
  std::iota(key_col_indices.begin(), key_col_indices.end(), 0);
  
  std::tie(grouped_table, discard_indices) = 
    gdf_group_by_without_aggregations(input_table, input_table.num_columns(),
                                      key_col_indices.data(),
                                      const_cast<gdf_context*>(&context));

  gdf_column_free(&discard_indices);

  // discard_indices now contains the starting location of all the groups
  // but we don't want that. WE want indices when the last column is NOT included
  // So make a table without it and get the indices for it
  std::vector<gdf_column*> key_cols(input_table.num_columns() - 1);
  std::transform(key_col_indices.begin(), key_col_indices.end() - 1,
                 key_cols.begin(),
                 [&grouped_table](gdf_index_type const index) {
                   return grouped_table.get_column(index);
                 });
  cudf::table key_table(key_cols);

  // And get the group indices again, this time excluding the last (values) column
  auto group_indices = gdf_unique_indices(key_table, context);
  key_table.destroy();
  auto grouped_values = **(grouped_table.end() - 1);
  return std::make_pair(grouped_values, group_indices);
}

} // namespace detail

// TODO: add optional check for is_sorted. Use context.flag_sorted
gdf_column group_quantiles(cudf::table const& input_table,
                           double quantile,
                           gdf_quantile_method interpolation,
                           gdf_context const& context)
{
  gdf_column grouped_values, group_indices;
  std::tie(grouped_values, group_indices) = 
    detail::group_values_and_indices(input_table, context);

  // TODO: currently ignoring nulls
  gdf_size_type num_grps = group_indices.size;
  auto quants_col = cudf::allocate_column(GDF_FLOAT64, num_grps, false);

  type_dispatcher(grouped_values.dtype, quantiles_functor{},
                  grouped_values, group_indices, quants_col, quantile,
                  interpolation);

  gdf_column_free(&grouped_values);
  gdf_column_free(&group_indices);

  return quants_col;
}
    
} // namespace cudf


