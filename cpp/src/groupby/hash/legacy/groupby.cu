
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

#include <cudf/cudf.h>
#include <bitmask/legacy/bit_mask.cuh>
#include <cudf/legacy/copying.hpp>
#include <cudf/legacy/groupby.hpp>
#include <cudf/legacy/bitmask.hpp>
#include <cudf/legacy/table.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <hash/concurrent_unordered_map.cuh>
#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>
#include <utilities/legacy/column_utils.hpp>
#include <utilities/legacy/cuda_utils.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include "groupby_kernels.cuh"
#include "groupby/common/legacy/aggregation_requests.hpp"
#include "groupby/common/legacy/type_info.hpp"
#include "groupby/common/legacy/utils.hpp"
#include <rmm/thrust_rmm_allocator.h>
#include <thrust/fill.h>
#include <type_traits>
#include <vector>
#include <cudf/detail/utilities/integer_utils.hpp>

namespace cudf {
namespace groupby {

namespace hash {
namespace {

template <bool keys_have_nulls, bool values_have_nulls>
auto build_aggregation_map(table const& input_keys, table const& input_values,
                           device_table const& d_input_keys,
                           device_table const& d_input_values,
                           std::vector<operators> const& ops, Options options,
                           cudaStream_t stream) {
  cudf::size_type constexpr unused_key{std::numeric_limits<cudf::size_type>::max()};
  cudf::size_type constexpr unused_value{
      std::numeric_limits<cudf::size_type>::max()};
  CUDF_EXPECTS(input_keys.num_rows() < unused_key,
               "Groupby input size too large.");

  // The exact output size is unknown a priori, therefore, use the input size as
  // an upper bound.
  cudf::size_type const output_size_estimate{input_keys.num_rows()};

  cudf::table sparse_output_values{
      output_size_estimate,
      target_dtypes(column_dtypes(input_values), ops),
      column_dtype_infos(input_values),
      values_have_nulls,
      false,
      stream};

  initialize_with_identity(sparse_output_values, ops, stream);

  auto d_sparse_output_values =
      device_table::create(sparse_output_values, stream);
  rmm::device_vector<operators> d_ops(ops);

  // If we ignore null keys, then nulls are not equivalent
  bool const null_keys_are_equal{not options.ignore_null_keys};
  bool const skip_key_rows_with_nulls{keys_have_nulls and
                                      not null_keys_are_equal};

  row_hasher<keys_have_nulls> hasher{d_input_keys};
  row_equality_comparator<keys_have_nulls> rows_equal{
      d_input_keys, d_input_keys, null_keys_are_equal};

  using map_type =
      concurrent_unordered_map<cudf::size_type, cudf::size_type, decltype(hasher),
                               decltype(rows_equal)>;
  using allocator_type = typename map_type::allocator_type;

  auto map = map_type::create(compute_hash_table_size(input_keys.num_rows()),
                              unused_key, unused_value, hasher, rows_equal,
                              allocator_type(), stream);

  // TODO: Explore optimal block size and work per thread.
  cudf::util::cuda::grid_config_1d grid_params{input_keys.num_rows(), 256};

  if (skip_key_rows_with_nulls) {
    auto row_bitmask{cudf::row_bitmask(input_keys, stream)};
    build_aggregation_map<true, values_have_nulls>
        <<<grid_params.num_blocks, grid_params.num_threads_per_block, 0,
           stream>>>(*map, d_input_keys, d_input_values,
                     *d_sparse_output_values, d_ops.data().get(),
                     row_bitmask.data().get());
  } else {
    build_aggregation_map<false, values_have_nulls>
        <<<grid_params.num_blocks, grid_params.num_threads_per_block, 0,
           stream>>>(*map, d_input_keys, d_input_values,
                     *d_sparse_output_values, d_ops.data().get(), nullptr);
  }
  CHECK_CUDA(stream);

  return std::make_pair(std::move(map), sparse_output_values);
}

template <bool keys_have_nulls, bool values_have_nulls, typename Map>
auto extract_results(table const& input_keys, table const& input_values,
                     device_table const& d_input_keys,
                     table const& sparse_output_values, Map const& map,
                     cudaStream_t stream) {

  cudf::table output_keys{
      cudf::allocate_like(
        input_keys,
        keys_have_nulls ? RETAIN : NEVER,
        stream)};
  cudf::table output_values{
      cudf::allocate_like(
        sparse_output_values,
        values_have_nulls ? RETAIN : NEVER,
        stream)};

  auto d_sparse_output_values =
      device_table::create(sparse_output_values, stream);

  auto d_output_keys = device_table::create(output_keys, stream);
  auto d_output_values = device_table::create(output_values, stream);

  cudf::size_type* d_result_size{nullptr};
  RMM_TRY(RMM_ALLOC(&d_result_size, sizeof(cudf::size_type), stream));
  CUDA_TRY(cudaMemsetAsync(d_result_size, 0, sizeof(cudf::size_type), stream));

  cudf::util::cuda::grid_config_1d grid_params{input_keys.num_rows(), 256};

  extract_groupby_result<keys_have_nulls, values_have_nulls>
      <<<grid_params.num_blocks, grid_params.num_threads_per_block, 0,
         stream>>>(map, d_input_keys, *d_output_keys, *d_sparse_output_values,
                   *d_output_values, d_result_size);

  CHECK_CUDA(stream);

  cudf::size_type result_size{-1};
  CUDA_TRY(cudaMemcpyAsync(&result_size, d_result_size, sizeof(cudf::size_type),
                           cudaMemcpyDeviceToHost, stream));

  // Update size and null count of output columns
  auto update_column = [result_size](gdf_column* col) {
    CUDF_EXPECTS(col != nullptr, "Attempt to update Null column.");
    col->size = result_size;
    set_null_count(*col);
    return col;
  };

  std::transform(output_keys.begin(), output_keys.end(), output_keys.begin(),
                 update_column);
  std::transform(output_values.begin(), output_values.end(),
                 output_values.begin(), update_column);

  return std::make_pair(output_keys, output_values);
}

/**---------------------------------------------------------------------------*
 * @brief Computes the groupby operation for a set of keys, values, and
 * operators using a hash-based implementation.
 *
 * The algorithm has two primary steps:
 * 1.) Build a hash map
 * 2.) Extract the non-empty entries from the hash table
 *
 * 1.) The hash map is built by inserting every row `i` from the `keys` and
 * `values` tables as a single (key,value) pair. When the pair is inserted, if
 * the key was not already present in the map, then the corresponding value is
 * simply copied to the output. If the key was already present in the map,
 * then the inserted `values` row is aggregated with the existing row. This
 * aggregation is done for every element `j` in the row by applying aggregation
 * operation `j` between the new and existing element.
 *
 * This process yields a hash map and table holding the resulting aggregation
 * rows. The aggregation output table is sparse, i.e., not every row is
 * populated. This is because the size of the output is not known a priori, and
 * so the output aggregation table is allocated to be as large as the input (the
 * upper bound of the output size).
 *
 * 2.) The final result is materialized by extracting the non-empty keys from
 * the hash map and the non-empty rows from the sparse output aggregation table.
 * Every non-empty key and value row is appended to the output key and value
 * tables.
 *
 * @tparam keys_have_nulls Indicates keys have one or more null values
 * @tparam values_have_nulls Indicates values have one or more null values
 * @param keys Table whose rows are used as keys of the groupby
 * @param values Table whose rows are aggregated in the groupby
 * @param ops Set of aggregation operations to perform for each element in a row
 * in the values table
 * @param options Options to control behavior of the groupby operation
 * @param stream CUDA stream on which all memory allocations and kernels will be
 * executed
 * @return A pair of the output keys table and output values table
 *---------------------------------------------------------------------------**/
template <bool keys_have_nulls, bool values_have_nulls>
auto compute_hash_groupby(cudf::table const& keys, cudf::table const& values,
                          std::vector<operators> const& ops, Options options,
                          cudaStream_t stream) {
  CUDF_EXPECTS(values.num_columns() == static_cast<cudf::size_type>(ops.size()),
               "Size mismatch between number of value columns and number of "
               "aggregations.");

  // An "aggregation request" is the combination of a `gdf_column*` to a column
  // of values, and an aggregation operation enum indicating the aggregation
  // requested to be performed on the column
  std::vector<AggRequestType> original_requests(values.num_columns());
  std::transform(values.begin(), values.end(), ops.begin(),
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
  std::vector<SimpleAggRequestCounter> simple_agg_columns =
      compound_to_simple(original_requests);

  std::vector<gdf_column*> simple_values_columns;
  std::vector<operators> simple_operators;
  for (auto const& p : simple_agg_columns) {
    const AggRequestType& agg_req_type = p.first;
    simple_values_columns.push_back(
        const_cast<gdf_column*>(agg_req_type.first));
    simple_operators.push_back(agg_req_type.second);
  }

  cudf::table simple_values_table{simple_values_columns};

  auto const d_input_keys = device_table::create(keys);
  auto const d_input_values = device_table::create(simple_values_table);

  // Step 1: Build hash map
  auto result = build_aggregation_map<keys_have_nulls, values_have_nulls>(
      keys, simple_values_table, *d_input_keys, *d_input_values,
      simple_operators, options, stream);

  auto const map{std::move(result.first)};
  cudf::table sparse_output_values{result.second};

  // Step 2: Extract non-empty entries
  cudf::table output_keys;
  cudf::table simple_output_values;
  std::tie(output_keys, simple_output_values) =
      extract_results<keys_have_nulls, values_have_nulls>(
          keys, values, *d_input_keys, sparse_output_values, *map, stream);

  // Delete intermediate results storage
  sparse_output_values.destroy();

  // If any of the original requests were compound, compute them from the
  // results of simple aggregation requests
  cudf::table final_output_values = compute_original_requests(
      original_requests, simple_agg_columns, simple_output_values, stream);

  return std::make_pair(output_keys, final_output_values);
}

/**---------------------------------------------------------------------------*
 * @brief Returns appropriate callable instantiation of `compute_hash_groupby`
 * based on presence of null values in keys and values.
 *
 * @param keys The groupby key columns
 * @param values The groupby value columns
 * @return Instantiated callable of compute_hash_groupby
 *---------------------------------------------------------------------------**/
auto groupby_null_specialization(table const& keys, table const& values) {
  if (cudf::has_nulls(keys)) {
    if (cudf::has_nulls(values)) {
      return compute_hash_groupby<true, true>;
    } else {
      return compute_hash_groupby<true, false>;
    }
  } else {
    if (cudf::has_nulls(values)) {
      return compute_hash_groupby<false, true>;
    } else {
      return compute_hash_groupby<false, false>;
    }
  }
}

}  // namespace
namespace detail {
std::pair<cudf::table, cudf::table> groupby(cudf::table const& keys,
                                            cudf::table const& values,
                                            std::vector<operators> const& ops,
                                            Options options,
                                            cudaStream_t stream = 0) {
  CUDF_EXPECTS(keys.num_rows() == values.num_rows(),
               "Size mismatch between number of rows in keys and values.");

  verify_operators(values, ops);

  // Empty inputs
  if (keys.num_rows() == 0) {
    return std::make_pair(
        cudf::empty_like(keys),
        cudf::table(0, target_dtypes(column_dtypes(values), ops),
                    column_dtype_infos(values)));
  }

  auto compute_groupby = groupby_null_specialization(keys, values);

  cudf::table output_keys;
  cudf::table output_values;
  std::tie(output_keys, output_values) =
      compute_groupby(keys, values, ops, options, stream);

  update_nvcategories(keys, output_keys, values, output_values);

  return std::make_pair(output_keys, output_values);
}
}  // namespace detail

std::pair<cudf::table, cudf::table> groupby(cudf::table const& keys,
                                            cudf::table const& values,
                                            std::vector<operators> const& ops,
                                            Options options) {
  return detail::groupby(keys, values, ops, options);
}
}  // namespace hash
}  // namespace groupby
}  // namespace cudf
