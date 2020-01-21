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

#include <bitmask/legacy/bit_mask.cuh>
#include <cudf/legacy/copying.hpp>
#include <cudf/cudf.h>
#include <cudf/legacy/groupby.hpp>
#include <cudf/legacy/bitmask.hpp>
#include <cudf/legacy/table.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>
#include <utilities/legacy/column_utils.hpp>
#include <utilities/legacy/cuda_utils.hpp>

#include "groupby/common/legacy/aggregation_requests.hpp"
#include "groupby/common/legacy/type_info.hpp"
#include "groupby/common/legacy/utils.hpp"
#include "groupby_kernels.cuh"
#include "sort_helper.hpp"

#include <quantiles/legacy/group_quantiles.hpp>
#include <reductions/legacy/group_reductions.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>

namespace cudf {
namespace groupby {
namespace sort {

using index_vector = rmm::device_vector<cudf::size_type>;

namespace {

/**---------------------------------------------------------------------------*
 * @brief Computes the ordered aggregation requests which were skipped 
 * in a previous process (`compound_to_simple`). These ordered aggregations
 * were skipped because they can't be compound to simple aggregation.
 * 
 * Then combine these results with  the set of output aggregation columns 
 * corresponding to not ordered aggregation requests.
 *
 * @param groupby[in] The object for computing sort-based groupby
 * @param original_requests[in] The original set of potentially ordered
 * aggregation requests
 * @param input_ops_args[in] The list of arguments fot each of the previous ordered
 * aggregation requests
 * @param current_output_values[in] Set of output aggregation columns corresponding to
 * not ordered aggregation requests
 * @param stream[in] CUDA stream on which to execute
 * @return vector of columns satisfying each of the original aggregation requests
 *---------------------------------------------------------------------------**/
std::vector<gdf_column*>  compute_ordered_aggregations(
    detail::helper &groupby,
    std::vector<AggRequestType> const& original_requests,
    std::vector<operation_args*> const& input_ops_args,
    cudf::table& current_output_values,
    cudaStream_t stream) {

  std::vector<gdf_column*> output_value(original_requests.size());
  std::copy(current_output_values.begin(), current_output_values.end(), output_value.begin());

  for (size_t i = 0; i < original_requests.size(); ++i) {
    auto const& element = original_requests[i];
    if (is_ordered(element.second)) {
      gdf_column * value_col = element.first;
      gdf_column sorted_values;
      rmm::device_vector<cudf::size_type> group_sizes;

      std::tie(sorted_values, group_sizes) = groupby.sort_values(*value_col);
      auto result_col = new gdf_column;

      switch (element.second) {
      case MEDIAN: {
        *result_col = cudf::allocate_column(
          GDF_FLOAT64, groupby.num_groups(), false);

        cudf::detail::group_medians(sorted_values, groupby.group_offsets(),
                                    group_sizes, result_col, stream);
        break;
      }
      case QUANTILE: {
        auto args = static_cast<quantile_args*>(input_ops_args[i]);

        *result_col = cudf::allocate_column(
          GDF_FLOAT64, args->quantiles.size() * groupby.num_groups(), false);

        cudf::detail::group_quantiles(sorted_values, groupby.group_offsets(),
                                      group_sizes, result_col,
                                      args->quantiles, args->interpolation,
                                      stream);
        break;
      }
      case VARIANCE: {
        auto args = static_cast<std_args*>(input_ops_args[i]);

        *result_col = cudf::allocate_column(
          GDF_FLOAT64, groupby.num_groups());

        cudf::detail::group_var(sorted_values, groupby.group_labels(),
                                group_sizes, result_col,
                                args->ddof, stream);
        break;
      }
      case STD: {
        auto args = static_cast<std_args*>(input_ops_args[i]);

        *result_col = cudf::allocate_column(
          GDF_FLOAT64, groupby.num_groups());

        cudf::detail::group_std(sorted_values, groupby.group_labels(),
                                group_sizes, result_col,
                                args->ddof, stream);
        break;
      }
      default:
        break;
      }
      output_value[i] = result_col;

      gdf_column_free(&sorted_values);
    }
  }
  return output_value;
}

/**---------------------------------------------------------------------------*
 * @brief Prepare input parameters for invoking the `aggregate_all_rows` kernel
 * which compute the simple aggregation(s) of corresponding rows in the output
 * `values` table.
 * @param input_keys The table of keys
 * @param options The options for controlling behavior of the groupby operation.
 * @param groupby The object for computing sort-based groupby
 * @param simple_values_columns The list of simple values columns
 * @param simple_operators The list of simple aggregation operations
 * @param stream[in] CUDA stream on which to execute
 * @return output value table with the aggregation(s) computed 
 *---------------------------------------------------------------------------**/
template <bool keys_have_nulls, bool values_have_nulls>
cudf::table compute_simple_aggregations(const cudf::table &input_keys,
                               const Options &options,
                               detail::helper &groupby,
                               const std::vector<gdf_column *> &simple_values_columns,
                               const std::vector<operators> &simple_operators,
                               cudaStream_t &stream) { 

  const gdf_column& key_sorted_order = groupby.key_sort_order();   

  //group_labels 
  const index_vector& group_labels = groupby.group_labels();
  const cudf::size_type num_groups = groupby.num_groups();
  
  // Output allocation size aligned to 4 bytes. The use of `round_up_safe` 
  // guarantee correct execution with cuda-memcheck  for cases when 
  // num_groups == 1  and with dtype == int_8. 
  cudf::size_type const output_size_estimate = cudf::util::round_up_safe((int64_t)groupby.num_groups(), (int64_t)sizeof(int32_t));

  cudf::table simple_values_table{simple_values_columns};

  cudf::table simple_output_values{
      output_size_estimate, target_dtypes(column_dtypes(simple_values_table), simple_operators),
      column_dtype_infos(simple_values_table), values_have_nulls, false, stream};

  initialize_with_identity(simple_output_values, simple_operators, stream);

  auto d_input_values = device_table::create(simple_values_table);
  auto d_output_values = device_table::create(simple_output_values, stream);
  rmm::device_vector<operators> d_ops(simple_operators);

  auto row_bitmask = cudf::row_bitmask(input_keys, stream);

  cudf::util::cuda::grid_config_1d grid_params{input_keys.num_rows(), 256};

  //Aggregate all rows for simple requests using the key sorted order (indices) and the group labels
  cudf::groupby::sort::aggregate_all_rows<keys_have_nulls, values_have_nulls><<<
    grid_params.num_blocks, grid_params.num_threads_per_block, 0, stream>>>(
      *d_input_values, *d_output_values,
      static_cast<cudf::size_type const*>(key_sorted_order.data),
      group_labels.data().get(), options.ignore_null_keys,
      d_ops.data().get(), row_bitmask.data().get());
  
   std::transform(simple_output_values.begin(), simple_output_values.end(), simple_output_values.begin(),
                 [num_groups](gdf_column *col) {
                   CUDF_EXPECTS(col != nullptr, "Attempt to update Null column.");
                   col->size = num_groups;
                   return col;
                 });
  return simple_output_values;
}

template <bool keys_have_nulls, bool values_have_nulls>
std::pair<cudf::table, std::vector<gdf_column*>> compute_sort_groupby(cudf::table const& input_keys, cudf::table const& input_values,
                          std::vector<operators> const& input_ops,
                          std::vector<operation_args*> const& input_ops_args,
                          Options options,
                          cudaStream_t stream) {
  auto include_nulls = not options.ignore_null_keys;
  auto groupby = detail::helper(input_keys, include_nulls, options.null_sort_behavior, options.input_sorted);

  if (groupby.num_groups() == 0) {
    cudf::table output_values(0, target_dtypes(column_dtypes(input_values), input_ops), column_dtype_infos(input_values));
    return std::make_pair(
        cudf::empty_like(input_keys),
        std::vector<gdf_column*>{output_values.begin(), output_values.end()}
        );
  }
  cudf::size_type num_groups = groupby.num_groups();
  // An "aggregation request" is the combination of a `gdf_column*` to a column
  // of values, and an aggregation operation enum indicating the aggregation
  // requested to be performed on the column
  std::vector<AggRequestType> original_requests(input_values.num_columns());
  std::transform(input_values.begin(), input_values.end(), input_ops.begin(),
                 original_requests.begin(),
                 [](gdf_column const* col, operators op) {
                   return std::make_pair(const_cast<gdf_column*>(col), op);
                 });

  // Some aggregations are "compound", meaning they need be satisfied via the
  // composition of 1 or more "simple" aggregation requests. For example, MEAN
  // is satisfied via the division of the SUM by the COUNT aggregation. We
  // translate these compound requests into simple requests, and compute the
  // groupby operation for these simple requests. Later, we translate the simple
  // requests back to compound request results.
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

  // If there are "simple" aggregation requests, compute the aggregations 
  cudf::table current_output_values{};
  if (simple_values_columns.size() > 0) {
    // Step 1: Aggregate all rows for simple requests 
    cudf::table simple_output_values = compute_simple_aggregations<keys_have_nulls, values_have_nulls>(input_keys,
                              options,
                              groupby,
                              simple_values_columns,
                              simple_operators,
                              stream);
    // Step 2: If any of the original requests were compound, compute them from the
    // results of simple aggregation requests
    current_output_values = compute_original_requests(original_requests, simple_requests, simple_output_values, stream);
  }
  // If there are "ordered" aggregation requests like MEDIAN, QUANTILE, compute these aggregations 
  std::vector<gdf_column*> final_output_values = compute_ordered_aggregations(groupby, original_requests, input_ops_args, current_output_values, stream);

  // Update size and null count of output columns
  std::transform(final_output_values.begin(), final_output_values.end(), final_output_values.begin(),
                 [](gdf_column *col) {
                   CUDF_EXPECTS(col != nullptr, "Attempt to update Null column.");
                   set_null_count(*col);
                   return col;
                 });
  return std::make_pair(groupby.unique_keys(), std::move(final_output_values));
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
 

/**---------------------------------------------------------------------------*
 * @brief Verifies the requested aggregation is valid for the arguments of the 
 * operator.
 *
 * @throw cudf::logic_error if an invalid combination of argument and operator
 * is requested.
 *
 * @param ops The aggregation operators
 * @param ops The aggregation arguments
 *---------------------------------------------------------------------------**/
static void verify_operators_with_arguments(std::vector<operators> const& ops, std::vector<operation_args*> const& args) {
   CUDF_EXPECTS(ops.size() == args.size(),
               "Size mismatch between ops and args");
  for (size_t i = 0; i < ops.size(); i++) {
    if(ops[i] == QUANTILE) { 
      quantile_args* q_args = static_cast<quantile_args*>(args[i]); 
      if (q_args == nullptr or q_args->quantiles.size() == 0) {
        CUDF_FAIL(
                "Missing quantile aggregation arguments.");
      }
    } 
  }
}

std::pair<cudf::table, std::vector<gdf_column*>> groupby(cudf::table const& keys,
                                            cudf::table const& values,
                                            std::vector<operation> const& ops,
                                            Options options,
                                            cudaStream_t stream = 0) {
  CUDF_EXPECTS(keys.num_rows() == values.num_rows(),
               "Size mismatch between number of rows in keys and values.");
  std::vector<operators> optype_list(ops.size());
  std::transform(ops.begin(), ops.end(), optype_list.begin(), [](auto const& op) {
    return op.op_name;
  });
  std::vector<operation_args*> ops_args(ops.size());
  std::transform(ops.begin(), ops.end(), ops_args.begin(), [](auto const& op) {
    return op.args.get();
  });
  verify_operators(values, optype_list);
  verify_operators_with_arguments(optype_list, ops_args);

  // Empty inputs
  if (keys.num_rows() == 0) {
    cudf::table output_values(0, target_dtypes(column_dtypes(values), optype_list), column_dtype_infos(values));
    return std::make_pair(
        cudf::empty_like(keys),
        std::vector<gdf_column*>{output_values.begin(), output_values.end()}
        );
  }

  auto compute_groupby = groupby_null_specialization(keys, values);

  cudf::table output_keys;
  std::vector<gdf_column*> output_values;
  std::tie(output_keys, output_values) =
      compute_groupby(keys, values, optype_list, ops_args, options, stream);
  
  cudf::table table_output_values(output_values);
  
  update_nvcategories(keys, output_keys, values, table_output_values);
  return std::make_pair(std::move(output_keys), std::move(output_values));                                              
}

} // namespace detail
 

std::pair<cudf::table, std::vector<gdf_column*>> groupby(cudf::table const &keys,
                                            cudf::table const &values,
                                            std::vector<operation> const &ops,
                                            Options options) {
  return detail::groupby(keys, values, ops, options);
}


} // END: namespace sort
} // END: namespace groupby
} // END: namespace cudf
