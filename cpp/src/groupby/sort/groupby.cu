/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Alexander Ocsa <alexander@blazingdb.com>
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

#include <algorithm>
#include <cassert>
#include <thrust/fill.h>
#include <tuple>


#include <cudf/cudf.h>
#include <bitmask/legacy/bit_mask.cuh>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/legacy/bitmask.hpp>
#include <cudf/legacy/table.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>
#include <utilities/column_utils.hpp>
#include <utilities/cuda_utils.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
 
 #include "../common/aggregation_requests.hpp"
#include "../common/util.hpp"
#include "groupby.hpp"
#include "groupby_kernels.cuh"  

using namespace cudf::groupby::common;

namespace cudf {
namespace groupby {
namespace sort {

namespace {   

cudf::table compose_inputs(cudf::table input_table, gdf_column* col) {
  std::vector<gdf_column*> output(input_table.num_columns());  
  std::transform(input_table.begin(), input_table.end(), output.begin(), [](const gdf_column *item){
    return (gdf_column *)item;
  }); 
  output.push_back(col);

  gdf_column **group_by_input_key = output.data();
  return cudf::table{group_by_input_key, input_table.num_columns() + 1};
}

cudf::table compose_output_keys(cudf::table input_table) {
  std::vector<gdf_column*> output(input_table.num_columns() - 1);  
  std::transform(input_table.begin(), input_table.end() - 1, output.begin(), [](const gdf_column *item){
    return (gdf_column *)item;
  }); 
  return cudf::table {output};
}

rmm::device_vector<gdf_size_type> get_last_column (cudf::table current_table) {
  auto num_column = current_table.num_columns();
  gdf_column * sorted_column = current_table.get_column(num_column - 1);
  rmm::device_vector<gdf_size_type> returned_vector(current_table.num_rows());
  cudaMemcpy(returned_vector.data().get(), sorted_column->data, sorted_column->size * sizeof(gdf_size_type), cudaMemcpyDeviceToDevice); 
  return returned_vector;
}

std::pair<cudf::table, gdf_column> compute_sort_groupby_wo_agg(cudf::table const& input_keys, 
                            Options options,
                            rmm::device_vector<gdf_size_type> &d_sorted_indices,
                            cudaStream_t stream) {
  gdf_context context;
  auto ignore_null_keys = options.ignore_null_keys;
  if (not ignore_null_keys) { // SQL
    context.flag_groupby_include_nulls = true;
    context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;
  } else { // PANDAS
    context.flag_groupby_include_nulls = false;
    context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;
  }

  std::vector<int> groupby_col_indices;
  for (gdf_size_type i = 0; i < input_keys.num_columns(); i++)
    groupby_col_indices.push_back(i);

  cudf::table sorted_keys_table;
  gdf_column group_indices_col;
  
  auto nrows = input_keys.num_rows();
  rmm::device_vector<gdf_size_type> d_seq_indices_values(nrows);
  thrust::sequence(d_seq_indices_values.begin(), d_seq_indices_values.end(), 0, 1);

  gdf_column seq_indices_col{};
  CUDF_TRY(gdf_column_view(&seq_indices_col,
                           (void *)(d_seq_indices_values.data().get()), nullptr,
                           nrows, GDF_INT32));

  auto input_table = compose_inputs(input_keys, &seq_indices_col);
  std::tie(sorted_keys_table,
                        group_indices_col) = gdf_group_by_without_aggregations(input_table,
                                                                          groupby_col_indices.size(),
                                                                          groupby_col_indices.data(),
                                                                          &context);
  cudf::table output_keys = compose_output_keys(sorted_keys_table);
  d_sorted_indices = get_last_column(sorted_keys_table); 
  return std::make_pair(output_keys, group_indices_col);
}

template <bool keys_have_nulls, bool values_have_nulls>
auto compute_sort_groupby(cudf::table const& input_keys, cudf::table const& input_values,
                          std::vector<operators> const& input_ops, Options options,
                          cudaStream_t stream) {
  cudf::table sorted_keys_table;
  gdf_column group_indices_col;

  rmm::device_vector<gdf_size_type> d_sorted_indices;
  std::tie(sorted_keys_table,
                          group_indices_col) = compute_sort_groupby_wo_agg(input_keys, options, d_sorted_indices, stream);

  if (sorted_keys_table.num_rows() == 0) {
    return std::make_pair(
        cudf::empty_like(input_keys),
        cudf::table(0, target_dtypes(column_dtypes(input_values), input_ops), column_dtype_infos(input_values)));
  }

  std::vector<AggRequestType> original_requests(input_values.num_columns());
  std::transform(input_values.begin(), input_values.end(), input_ops.begin(),
                 original_requests.begin(),
                 [](gdf_column const* col, operators op) {
                   return std::make_pair(const_cast<gdf_column*>(col), op);
                 });

  std::vector<SimpleAggRequestCounter> simple_requests =
      compound_to_simple(original_requests);

  std::vector<gdf_column*> simple_values_columns;
  std::vector<operators> simple_operators;
  for (auto const& p : simple_requests) {
    const AggRequestType& agg_req_type = p.first;
    simple_values_columns.push_back(
        const_cast<gdf_column*>(agg_req_type.first));
    simple_operators.push_back(agg_req_type.second);
  }
  cudf::table simple_values_table{simple_values_columns};

  cudf::table simple_output_values{
      group_indices_col.size, target_dtypes(column_dtypes(simple_values_table), simple_operators),
      column_dtype_infos(simple_values_table), values_have_nulls, false, stream};

  initialize_with_identity(simple_output_values, simple_operators, stream);

  auto d_input_keys = device_table::create(sorted_keys_table);
  auto d_input_values = device_table::create(simple_values_table);
  auto d_output_values = device_table::create(simple_output_values, stream);
  rmm::device_vector<operators> d_ops(simple_operators);
 
  auto row_bitmask = cudf::row_bitmask(sorted_keys_table, stream);

  cudf::util::cuda::grid_config_1d grid_params{sorted_keys_table.num_rows(), 256};

  cudf::groupby::sort::aggregate_all_rows<keys_have_nulls, values_have_nulls><<<
      grid_params.num_blocks, grid_params.num_threads_per_block, 0, stream>>>(
      *d_input_keys, *d_input_values, *d_output_values, d_sorted_indices.data().get(), 
      (gdf_index_type *)group_indices_col.data, group_indices_col.size,
      d_ops.data().get(), row_bitmask.data().get());

  cudf::table destination_table(group_indices_col.size,
                                cudf::column_dtypes(sorted_keys_table),
                                cudf::column_dtype_infos(sorted_keys_table),
                                keys_have_nulls);
  
  cudf::gather(&sorted_keys_table, (gdf_index_type *)group_indices_col.data,
               &destination_table); 

  cudf::table final_output_values = compute_original_requests(
      original_requests, simple_requests, simple_output_values, stream);

  // FIX: destroy temporal tables, and temporal gdf_columns! 
  return std::make_pair(destination_table, final_output_values);
}

/**---------------------------------------------------------------------------*
 * @brief Returns appropriate callable instantiation of `compute_sort_groupby`
 * based on presence of null values in keys and values.
 *
 * @param keys The groupby key columns
 * @param values The groupby value columns
 * @return Instantiated callable of compute_sort_groupby
 *---------------------------------------------------------------------------**/
auto groupby_null_specialization(table const& keys, table const& values) {
  if (cudf::has_nulls(keys)) {
    if (cudf::has_nulls(values)) {
      return compute_sort_groupby<true, true>;
    } else {
      return compute_sort_groupby<true, false>;
    }
  } else {
    if (cudf::has_nulls(values)) {
      return compute_sort_groupby<false, true>;
    } else {
      return compute_sort_groupby<false, false>;
    }
  }
}
} // anonymous namespace

namespace detail {

std::pair<cudf::table, cudf::table> groupby(cudf::table const &keys,
                                            cudf::table const &values,
                                            std::vector<operators> const &ops,
                                            Options options,
                                            cudaStream_t stream) {
  CUDF_EXPECTS(keys.num_rows() == values.num_rows(),
               "Size mismatch between number of rows in keys and values.");

  verify_operators(values, ops);

  // Empty inputs
  if (keys.num_rows() == 0) {
    return std::make_pair(
        cudf::empty_like(keys),
        cudf::table(0, target_dtypes(column_dtypes(values), ops), column_dtype_infos(values)));
  }

 auto compute_groupby = groupby_null_specialization(keys, values);

  cudf::table output_keys;
  cudf::table output_values;
  std::tie(output_keys, output_values) =
      compute_groupby(keys, values, ops, options, stream);

  update_nvcategories(keys, output_keys, values, output_values);
  return std::make_pair(output_keys, output_values);
}

} // namespace detail

std::pair<cudf::table, cudf::table> groupby(cudf::table const &keys,
                                            cudf::table const &values,
                                            std::vector<operators> const &ops,
                                            Options options) {
  return detail::groupby(keys, values, ops, options);
}

} // END: namespace sort
} // END: namespace groupby
} // END: namespace cudf
