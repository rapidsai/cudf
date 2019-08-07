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
#include <cudf/copying.hpp>
#include <cudf/legacy/column.hpp>

#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <tuple>

namespace cudf {

namespace {

struct quantiles_functor {

  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, void >
  operator()(gdf_column const& values_col, gdf_column const& group_indices,
             gdf_column& result_col, rmm::device_vector<double> const& quantile,
             gdf_quantile_method interpolation)
  {
    // prepare args to be used by lambda below
    auto result = reinterpret_cast<double*>(result_col.data);
    auto values = reinterpret_cast<T*>(values_col.data);
    auto grp_id = reinterpret_cast<gdf_size_type*>(group_indices.data);
    auto d_quants = quantile.data().get();
    auto num_qnts = quantile.size();
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

        for (gdf_size_type j = 0; j < num_qnts; j++) {
          gdf_size_type k = i * num_qnts + j;
          result[k] = detail::select_quantile(values + grp_id[i], segment_size,
                                              d_quants[j], interpolation);
        }
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
// But that needs a cudf::gather that can take a transformed iterator
auto group_values_and_indices(cudf::table const& key_table,
                              gdf_context const& context)
{
  // Sort and groupby the input table
  cudf::table sorted_key_table;
  gdf_column group_indices;
  std::vector<gdf_index_type> key_col_indices(key_table.num_columns());
  std::iota(key_col_indices.begin(), key_col_indices.end(), 0);
  
  std::tie(sorted_key_table, group_indices) =
    gdf_group_by_without_aggregations(key_table, key_table.num_columns(),
                                      key_col_indices.data(),
                                      const_cast<gdf_context*>(&context));
  
  // Get output_keys using group_indices and input table
  gdf_size_type num_grps = group_indices.size;
  auto out_key_table = cudf::allocate_like_of_size(key_table, num_grps);
  cudf::gather(&sorted_key_table,
               reinterpret_cast<gdf_index_type*>(group_indices.data),
               &out_key_table);

  // Temporary guest. Not needed because we'll sort again including value cols
  sorted_key_table.destroy();

  return std::make_pair(out_key_table, group_indices);
}

} // namespace detail

// TODO: add optional check for is_sorted. Use context.flag_sorted
std::pair<cudf::table, gdf_column>
group_quantiles(cudf::table const& key_table,
                gdf_column const& values,
                std::vector<double> const& quantiles,
                gdf_quantile_method interpolation,
                gdf_context const& context)
{
  // Get the group indices and table of unique keys. Latter is simply returned
  cudf::table out_key_table;
  gdf_column group_indices;
  std::tie(out_key_table, group_indices) = 
    detail::group_values_and_indices(key_table, context);

  rmm::device_vector<double> dv_quantiles(quantiles);

  // Per column =============================
  // Merge the key_table and values column because we want to sort them together
  std::vector<gdf_column*> input_columns(key_table.get_columns());
  input_columns.push_back(const_cast<gdf_column*>(&values));
  auto combined_table = cudf::table(input_columns);

  // Get sorted indices
  auto idx_col = allocate_column(gdf_dtype_of<gdf_index_type>(),
                                 combined_table.num_rows(),
                                 false);
  gdf_order_by(combined_table.begin(), nullptr,
               combined_table.num_columns(), &idx_col,
               const_cast<gdf_context*>(&context));

  // Sort the values column
  auto sorted_values = allocate_like(values);
  auto val_table = cudf::table{const_cast<gdf_column*>(&values)};
  auto sorted_val_table = cudf::table{&sorted_values};
  cudf::gather(&val_table, 
               reinterpret_cast<gdf_index_type*>(idx_col.data),
               &sorted_val_table);

  // Go forth and calculate the quantiles
  // TODO: currently ignoring nulls
  auto result_col = cudf::allocate_column(GDF_FLOAT64,
                                          group_indices.size * quantiles.size(),
                                          false);

  type_dispatcher(sorted_values.dtype, quantiles_functor{},
                  sorted_values, group_indices, result_col, dv_quantiles,
                  interpolation);

  gdf_column_free(&idx_col);
  gdf_column_free(&sorted_values);
  // Per column =============================

  gdf_column_free(&group_indices);

  return std::make_pair(out_key_table, result_col);
}
    
} // namespace cudf


