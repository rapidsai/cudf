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
#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>
#include <bitmask/legacy/bit_mask.cuh>
#include <utilities/type_dispatcher.hpp>

#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <cudf/groupby.hpp>
#include <cudf/copying.hpp>
#include <cudf/legacy/column.hpp>

#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include <thrust/for_each.h>
#include <thrust/adjacent_difference.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <algorithm>
#include <tuple>
#include <numeric>

namespace cudf {

namespace {

struct quantiles_functor {

  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, void >
  operator()(gdf_column const& values_col, gdf_column const& group_indices,
             rmm::device_vector<gdf_size_type> const& group_sizes,
             gdf_column& result_col, rmm::device_vector<double> const& quantile,
             gdf_quantile_method interpolation)
  {
    // prepare args to be used by lambda below
    auto result = reinterpret_cast<double*>(result_col.data);
    auto values = reinterpret_cast<T*>(values_col.data);
    auto grp_id = reinterpret_cast<gdf_size_type*>(group_indices.data);
    auto grp_size = group_sizes.data().get();
    auto d_quants = quantile.data().get();
    auto num_qnts = quantile.size();
    auto num_vals = values_col.size;
    auto num_grps = group_indices.size;

    // For each group, calculate quantile
    thrust::for_each_n(thrust::device,
      thrust::make_counting_iterator(0),
      num_grps,
      [=] __device__ (gdf_size_type i) {
        gdf_size_type segment_size = grp_size[i];

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
std::tuple<cudf::table, gdf_column, std::vector<gdf_column*>,
  std::vector<rmm::device_vector<gdf_size_type> > >
group_values_and_indices(cudf::table const& key_table,
                              cudf::table const& val_table,
                              gdf_context const& context)
{
  // Sort and groupby the input table
  cudf::table sorted_table;
  gdf_column group_indices;
  std::vector<gdf_index_type> key_col_indices(key_table.num_columns());
  std::iota(key_col_indices.begin(), key_col_indices.end(), 0);

  // Combine key and val tables. We'll just segmentize the vals right now
  std::vector<gdf_column*> key_cols(key_table.get_columns());
  std::vector<gdf_column*> val_cols(val_table.get_columns());
  auto all_cols = key_cols;
  all_cols.insert(all_cols.end(), val_cols.begin(), val_cols.end());
  auto combined_table = cudf::table(all_cols);

  std::tie(sorted_table, group_indices) =
    gdf_group_by_without_aggregations(combined_table,
                                      key_table.num_columns(),
                                      key_col_indices.data(),
                                      const_cast<gdf_context*>(&context));

  // Get group labels for future use in segmented sorting
  gdf_size_type nrows = sorted_table.num_rows();
  rmm::device_vector<gdf_size_type> group_labels(nrows);
  thrust::fill(group_labels.begin(), group_labels.end(), 0);
  auto group_labels_ptr = group_labels.data().get();
  auto group_indices_ptr = reinterpret_cast<gdf_size_type*>(group_indices.data);
  thrust::for_each_n(thrust::make_counting_iterator(1),
                     group_indices.size - 1,
                     [=] __device__ (gdf_size_type i) { 
                       group_labels_ptr[group_indices_ptr[i]] = 1;
                     });
  thrust::inclusive_scan(thrust::device,
                        group_labels.begin(),
                        group_labels.end(),
                        group_labels.begin());

  // Sort individual value columns group wise
  auto seg_val_cols =
    std::vector<gdf_column*>(sorted_table.begin() + key_table.num_columns(),
                             sorted_table.end());

  auto idx_col = allocate_column(gdf_dtype_of<gdf_index_type>(),
                                 sorted_table.num_rows(),
                                 false);
  gdf_column group_labels_col;
  gdf_column_view(&group_labels_col, group_labels.data().get(), nullptr,
    group_labels.size(), gdf_dtype_of<gdf_size_type>());
  for (auto seg_val_col : seg_val_cols) {
    auto seg_table = cudf::table{&group_labels_col, seg_val_col};
    gdf_order_by(seg_table.begin(),
                 nullptr,
                 seg_table.num_columns(), // always 2
                 &idx_col,
                 const_cast<gdf_context*>(&context));

    cudf::table seg_val_col_table{seg_val_col};
    cudf::gather(&seg_val_col_table,
                 reinterpret_cast<gdf_size_type*>(idx_col.data),
                 &seg_val_col_table);
  }

  // Get number of valid values in each group
  std::vector<rmm::device_vector<gdf_size_type> > vals_group_sizes;
  for (auto seg_val_col : seg_val_cols)
  {
    rmm::device_vector<gdf_size_type> val_group_sizes(group_indices.size);
    auto col_valid = reinterpret_cast<bit_mask::bit_mask_t*>(seg_val_col->valid);

    rmm::device_vector<gdf_size_type> d_bools(seg_val_col->size);
    thrust::transform(
        thrust::make_counting_iterator(static_cast<gdf_size_type>(0)),
        thrust::make_counting_iterator(seg_val_col->size), d_bools.begin(),
        [col_valid] __device__ (gdf_size_type i) { return bit_mask::is_valid(col_valid, i); });

    thrust::reduce_by_key(
                          group_labels.begin(),
                          group_labels.end(),
                          d_bools.begin(),
                          thrust::make_discard_iterator(),
                          val_group_sizes.begin());
    vals_group_sizes.push_back(val_group_sizes);
  }

  // Get output_keys using group_indices and sorted_key_table
  // Separate key and value cols
  auto sorted_key_cols =
    std::vector<gdf_column*>(sorted_table.begin(),
                             sorted_table.begin() + key_table.num_columns());
  auto sorted_key_table = cudf::table(sorted_key_cols);

  gdf_size_type num_grps = group_indices.size;
  auto out_key_table = cudf::allocate_like_of_size(key_table, num_grps);
  cudf::gather(&sorted_key_table,
               reinterpret_cast<gdf_index_type*>(group_indices.data),
               &out_key_table);

  // No longer need sorted key columns
  sorted_key_table.destroy();

  return std::make_tuple(out_key_table, group_indices, seg_val_cols,
    vals_group_sizes);
}

} // namespace detail

// TODO: add optional check for is_sorted. Use context.flag_sorted
std::pair<cudf::table, cudf::table>
group_quantiles(cudf::table const& key_table,
                cudf::table const& val_table,
                std::vector<double> const& quantiles,
                gdf_quantile_method interpolation,
                gdf_context const& context)
{
  // Get the group indices and table of unique keys. Latter is simply returned
  cudf::table out_key_table;
  gdf_column group_indices;
  std::vector <gdf_column*> sorted_val_cols;
  std::vector<rmm::device_vector<gdf_size_type> > vals_group_sizes;
  std::tie(out_key_table, group_indices, sorted_val_cols, vals_group_sizes) = 
    detail::group_values_and_indices(key_table, val_table, context);


  rmm::device_vector<double> dv_quantiles(quantiles);

  cudf::table result_table(group_indices.size * quantiles.size(),
                           std::vector<gdf_dtype>(val_table.num_columns(),
                                                  GDF_FLOAT64));

  for (size_t i = 0; i < sorted_val_cols.size(); i++)
  {
    auto& sorted_values = *(sorted_val_cols[i]);
    auto& result_col = *(result_table.get_column(i));
    auto& group_sizes = vals_group_sizes[i];

    // Go forth and calculate the quantiles
    // TODO: currently ignoring nulls
    type_dispatcher(sorted_values.dtype, quantiles_functor{},
                    sorted_values, group_indices, group_sizes, result_col, dv_quantiles,
                    interpolation);

    gdf_column_free(&sorted_values);
  }
  
  gdf_column_free(&group_indices);

  return std::make_pair(out_key_table, result_table);
}
    
} // namespace cudf


