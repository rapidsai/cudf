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

// TODO: replace this quantiles includes and wait until they are merged  
#include "quantiles/groupby.hpp"

using namespace cudf::groupby::common;

namespace cudf {
namespace groupby {
namespace sort {

namespace {

using index_vector = rmm::device_vector<gdf_size_type>;

struct median_result_type {
  template <typename SourceType>
  gdf_dtype operator()() {
    return cudf::gdf_dtype_of<target_type_t<SourceType, MEDIAN>>();  
  }
};
  
cudf::table compute_remain_stats_requests(
    cudf::table current_output_values,
    std::vector<AggRequestType> const& original_requests,
    table input_keys,
    const gdf_column& group_indices_col,
    rmm::device_vector<gdf_size_type> key_sorted_order,
    Options options,
    cudaStream_t stream) {

  std::vector<gdf_column*> final_value_columns(original_requests.size());
  for (gdf_size_type i = 0; i < current_output_values.num_columns(); i++) {
    final_value_columns[i] = current_output_values.get_column(i);
  }

  for (size_t i = 0; i < original_requests.size(); ++i) {
    auto const& element = original_requests[i];
    if (element.second == MEDIAN) {
      // gdf_column * value_col = element.first;
      // auto nrows = input_keys.num_rows();
      

      // rmm::device_vector<gdf_size_type> value_col_sorted_indices_vector(thrust::make_counting_iterator(int(0)),
      //                                   thrust::make_counting_iterator(int(nrows)));

      // gdf_column value_col_sorted_indices{};
      // CUDF_TRY(gdf_column_view(&value_col_sorted_indices,
      //                         (void*)(value_col_sorted_indices_vector.data().get()), nullptr, nrows,
      //                         GDF_INT32));
      // // gdf_order_by_groups(group_indices_col, input_keys, &value_col, nullptr, 1, &value_col_sorted_indices, key_sorted_order, options, stream);
      // final_value_columns[i] = compute_median(group_indices_col, *value_col, value_col_sorted_indices_vector, key_sorted_order, stream);
    }
  }
  return cudf::table{final_value_columns};
}

template <bool keys_have_nulls, bool values_have_nulls>
auto compute_sort_groupby(cudf::table const& input_keys, cudf::table const& input_values,
                          std::vector<operators> const& input_ops, Options options,
                          cudaStream_t stream) {
  auto include_nulls = not options.ignore_null_keys;
  auto groupby = cudf::sort::groupby(input_keys, include_nulls);
  index_vector group_indices = groupby.group_indices();

  if (group_indices.size() == 0) {
    return std::make_pair(
        cudf::empty_like(input_keys),
        cudf::table(0, target_dtypes(column_dtypes(input_values), input_ops), column_dtype_infos(input_values)));
  }
  gdf_column key_sorted_order = groupby.key_sorted_order();
  index_vector group_labels = groupby.group_labels();
  gdf_size_type num_groups = (gdf_size_type)group_indices.size();

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
  cudf::table current_output_values{};
  // process simple columns
  if (simple_values_columns.size() > 0) {
    cudf::table simple_values_table{simple_values_columns};

    cudf::table simple_output_values{
        num_groups, target_dtypes(column_dtypes(simple_values_table), simple_operators),
        column_dtype_infos(simple_values_table), values_have_nulls, false, stream};

    initialize_with_identity(simple_output_values, simple_operators, stream);

    auto d_input_values = device_table::create(simple_values_table);
    auto d_output_values = device_table::create(simple_output_values, stream);
    rmm::device_vector<operators> d_ops(simple_operators);

    auto row_bitmask = cudf::row_bitmask(input_keys, stream);

    cudf::util::cuda::grid_config_1d grid_params{input_keys.num_rows(), 256};

    cudf::groupby::sort::aggregate_all_rows<keys_have_nulls, values_have_nulls><<<
        grid_params.num_blocks, grid_params.num_threads_per_block, 0, stream>>>(
        input_keys.num_rows(), *d_input_values, *d_output_values, (gdf_index_type *)key_sorted_order.data,
        group_labels.data().get(),
        d_ops.data().get(), row_bitmask.data().get());

    current_output_values = compute_original_requests(
        original_requests, simple_requests, simple_output_values, stream);
  }
  // cudf::table final_output_values = compute_remain_stats_requests(current_output_values, original_requests,  input_keys, group_indices_col, key_sorted_order, options, stream);
  return std::make_pair(groupby.unique_keys(), current_output_values);
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
