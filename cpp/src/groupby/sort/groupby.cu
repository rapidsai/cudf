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
#include <cudf/copying.hpp>
#include <cudf/cudf.h>
#include <cudf/groupby.hpp>
#include <cudf/legacy/bitmask.hpp>
#include <cudf/legacy/table.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>
#include <utilities/column_utils.hpp>
#include <utilities/cuda_utils.hpp>

#include "../common/aggregation_requests.hpp"
#include "../common/utils.hpp"
#include "groupby.hpp"
#include "groupby_kernels.cuh"

#include <quantiles/groupby.hpp>
#include <quantiles/quantiles.hpp>

namespace cudf {
namespace groupby {
namespace sort {

using index_vector = rmm::device_vector<gdf_size_type>;

namespace {
 
struct quantiles_functor {

  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, void >
  operator()(gdf_column const& values_col,
             rmm::device_vector<gdf_size_type> const& group_indices,
             rmm::device_vector<gdf_size_type> const& group_sizes,
             gdf_column& result_col, rmm::device_vector<double> const& quantile,
             gdf_quantile_method interpolation)
  {
    // prepare args to be used by lambda below
    auto result = reinterpret_cast<double*>(result_col.data);
    auto values = reinterpret_cast<T*>(values_col.data);
    auto grp_id = group_indices.data().get();
    auto grp_size = group_sizes.data().get();
    auto d_quants = quantile.data().get();
    auto num_qnts = quantile.size();

    // For each group, calculate quantile
    thrust::for_each_n(thrust::device,
      thrust::make_counting_iterator(0),
      group_indices.size(),
      [=] __device__ (gdf_size_type i) {
        gdf_size_type segment_size = grp_size[i];

        for (gdf_size_type j = 0; j < num_qnts; j++) {
          gdf_size_type k = i * num_qnts + j;
          result[k] = cudf::detail::select_quantile(values + grp_id[i], segment_size,
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

std::vector<gdf_column*>  compute_complex_request(
    cudf::table current_output_values,
    std::vector<AggRequestType> const& original_requests,
     std::vector<operation_args*> const& input_ops_args,
    cudf::detail::groupby &groupby,
    cudaStream_t stream) {

  std::vector<gdf_column*> final_value_columns(original_requests.size());
  for (gdf_size_type i = 0; i < current_output_values.num_columns(); i++) {
    final_value_columns[i] = current_output_values.get_column(i);
  }

  rmm::device_vector<gdf_size_type> group_indices = groupby.group_indices();
  index_vector group_labels = groupby.group_labels();
  gdf_size_type num_groups = (gdf_size_type)group_indices.size();


  for (size_t i = 0; i < original_requests.size(); ++i) {
    auto const& element = original_requests[i];
    if (element.second == MEDIAN || element.second == QUANTILE) {
      std::vector<double> quantiles;
      gdf_quantile_method interpolation;

      if (element.second == MEDIAN) {
        quantiles.push_back(0.5);
        interpolation = GDF_QUANT_LINEAR;
      } else if (element.second == QUANTILE){
        quantile_args * args = static_cast<quantile_args*>(input_ops_args[i]);
        quantiles = args->quantiles;
        interpolation = args->interpolation;
      }
      gdf_column * value_col = element.first;
      gdf_column sorted_values;
      rmm::device_vector<gdf_size_type> group_sizes;

      std::tie(sorted_values, group_sizes) = groupby.sort_values(*value_col);
      gdf_column* result_col = new gdf_column;
      *result_col = cudf::allocate_column(GDF_FLOAT64, quantiles.size() * groupby.num_groups(), false);
      rmm::device_vector<double> dv_quantiles(quantiles);

      cudf::type_dispatcher(sorted_values.dtype, quantiles_functor{},
                          sorted_values, group_indices, group_sizes, 
                          *result_col,
                          dv_quantiles, interpolation);
      final_value_columns[i] = result_col;
      gdf_column_free(&sorted_values);
    }
  }
  
  // Update size and null count of output columns
  auto update_column = [](gdf_column* col) {
    CUDF_EXPECTS(col != nullptr, "Attempt to update Null column.");
    set_null_count(*col);
    return col;
  };
  for (size_t i = 0; i < final_value_columns.size(); i++) {
    update_column(final_value_columns[i]);
  }
  return final_value_columns;
}

template <bool keys_have_nulls, bool values_have_nulls>
cudf::table process_original_requests(const cudf::table &input_keys,
                               const Options &options,
                               cudf::detail::groupby &groupby,
                               const std::vector<AggRequestType> &original_requests,
                               const std::vector<SimpleAggRequestCounter> &simple_requests,
                               const std::vector<gdf_column *> &simple_values_columns,
                               const std::vector<operators> &simple_operators,
                               cudaStream_t &stream) {
  index_vector group_indices = groupby.group_indices();
  gdf_column key_sorted_order = groupby.key_sorted_order();
  index_vector group_labels = groupby.group_labels();
  gdf_size_type num_groups = (gdf_size_type)group_indices.size();
  
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
          d_ops.data().get(), row_bitmask.data().get(), options.ignore_null_keys);

  // compute_simple_request
  return compute_original_requests(
      original_requests, simple_requests, simple_output_values, stream);
}

template <bool keys_have_nulls, bool values_have_nulls>
auto compute_sort_groupby(cudf::table const& input_keys, cudf::table const& input_values,
                          std::vector<operators> const& input_ops,
                          std::vector<operation_args*> const& input_ops_args,
                          Options options,
                          cudaStream_t stream) {
  auto include_nulls = not options.ignore_null_keys;
  auto groupby = cudf::detail::groupby(input_keys, include_nulls, options.null_sort_behavior, options.input_sorted);
  index_vector group_indices = groupby.group_indices();

  if (group_indices.size() == 0) {
    cudf::table output_values(0, target_dtypes(column_dtypes(input_values), input_ops), column_dtype_infos(input_values));
    return std::make_pair(
        cudf::empty_like(input_keys),
        output_values.get_columns()
        );
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
  cudf::table current_output_values{};
  // process simple columns
  if (simple_values_columns.size() > 0) {
    current_output_values = process_original_requests<keys_have_nulls, values_have_nulls>(input_keys,
                              options,
                              groupby,
                              original_requests,
                              simple_requests,
                              simple_values_columns,
                              simple_operators,
                              stream);
  }
  // compute_complex_request
  std::vector<gdf_column*> final_output_values = compute_complex_request(current_output_values, original_requests, input_ops_args, groupby, stream);
  return std::make_pair(groupby.unique_keys(), final_output_values);
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
 
std::pair<cudf::table, std::vector<gdf_column*>> groupby(cudf::table const& keys,
                                            cudf::table const& values,
                                            std::vector<operation> const& ops,
                                            Options options,
                                            cudaStream_t stream) {
  CUDF_EXPECTS(keys.num_rows() == values.num_rows(),
               "Size mismatch between number of rows in keys and values.");
  std::vector<operators> optype_list;
  for (auto &op : ops) {
    optype_list.push_back( op.op_name );
  }
  verify_operators(values, optype_list);
  // Empty inputs
  if (keys.num_rows() == 0) {
    cudf::table output_values(0, target_dtypes(column_dtypes(values), optype_list), column_dtype_infos(values));
    return std::make_pair(
        cudf::empty_like(keys),
        output_values.get_columns()
        );
  }

  auto compute_groupby = groupby_null_specialization(keys, values);

  std::vector<operation_args*> ops_args;
  for (auto &op : ops) {
    ops_args.emplace_back( op.args.get() );
  }

  cudf::table output_keys;
  std::vector<gdf_column*> output_values;
  std::tie(output_keys, output_values) =
      compute_groupby(keys, values, optype_list, ops_args, options, stream);
  
  cudf::table table_output_values(output_values);
  
  update_nvcategories(keys, output_keys, values, table_output_values);
  return std::make_pair(output_keys, output_values);                                              
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
