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

struct median_result_type {
  template <typename SourceType>
  gdf_dtype operator()() {
    return cudf::gdf_dtype_of<target_type_t<SourceType, MEDIAN>>();  
  }
};

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

cudf::table compute_complex_request(
    cudf::table current_output_values,
    std::vector<AggRequestType> const& original_requests,
    cudf::detail::groupby &groupby,
    cudaStream_t stream) {

  std::vector<gdf_column*> final_value_columns(original_requests.size());
  for (gdf_size_type i = 0; i < current_output_values.num_columns(); i++) {
    final_value_columns[i] = current_output_values.get_column(i);
  }

  rmm::device_vector<gdf_size_type> group_indices = groupby.group_indices();
  index_vector group_labels = groupby.group_labels();
  gdf_size_type num_groups = (gdf_size_type)group_indices.size();

  const std::vector<float> quantiles = {0.5};
  const gdf_quantile_method interpolation = GDF_QUANT_LINEAR;
  rmm::device_vector<double> dv_quantiles(quantiles);

  for (size_t i = 0; i < original_requests.size(); ++i) {
    auto const& element = original_requests[i];
    if (element.second == MEDIAN) {
      gdf_column * value_col = element.first;
      gdf_column sorted_values;
      rmm::device_vector<gdf_size_type> group_sizes;

      std::tie(sorted_values, group_sizes) = groupby.sort_values(*value_col);
      gdf_column* result_col = new gdf_column;
      *result_col = cudf::allocate_column(GDF_FLOAT64, groupby.num_groups(), false);

      cudf::type_dispatcher(sorted_values.dtype, quantiles_functor{},
                          sorted_values, group_indices, group_sizes, 
                          *result_col,
                          dv_quantiles, interpolation);
      final_value_columns[i] = result_col;
      gdf_column_free(&sorted_values);
    }
  }
  return cudf::table{final_value_columns};
}

template <bool keys_have_nulls, bool values_have_nulls>
auto compute_sort_groupby(cudf::table const& input_keys, cudf::table const& input_values,
                          std::vector<operators> const& input_ops, Options options,
                          cudaStream_t stream) {
  auto include_nulls = not options.ignore_null_keys;
  auto groupby = cudf::detail::groupby(input_keys, include_nulls);
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
        d_ops.data().get(), row_bitmask.data().get(), options.ignore_null_keys);

    // compute_simple_request
    current_output_values = compute_original_requests(
        original_requests, simple_requests, simple_output_values, stream);
  }
  // compute_complex_request
  cudf::table final_output_values = compute_complex_request(current_output_values, original_requests, groupby, stream);
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
