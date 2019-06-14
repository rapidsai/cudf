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
#include <bitmask/bit_mask.cuh>
#include <cudf/binaryop.hpp>
#include <cudf/bitmask.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table.hpp>
#include <hash/concurrent_unordered_map.cuh>
#include <string/nvcategory_util.hpp>
#include <table/device_table.cuh>
#include <table/device_table_row_operators.cuh>
#include <utilities/column_utils.hpp>
#include <utilities/cuda_utils.hpp>
#include <utilities/device_atomics.cuh>
#include <utilities/release_assert.cuh>
#include <utilities/type_dispatcher.hpp>
#include "groupby.hpp"
#include "groupby_kernels.cuh"
#include "type_info.hpp"

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/fill.h>
#include <algorithm>
#include <map>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace cudf {
namespace groupby {
namespace hash {
namespace {
using AggRequestType = std::pair<gdf_column*, operators>;
/**---------------------------------------------------------------------------*
 * @brief Verifies the requested aggregation is valid for the type of the value
 * column.
 *
 * Given a table of values and a set of operators, verifies that `ops[i]` is
 * valid to perform on `column[i]`.
 *
 * @throw cudf::logic_error if an invalid combination of value type and operator
 * is requested.
 *
 * @param values The table of columns
 * @param ops The aggregation operators
 *---------------------------------------------------------------------------**/
void verify_operators(table const& values, std::vector<operators> const& ops) {
  CUDF_EXPECTS(static_cast<gdf_size_type>(ops.size()) == values.num_columns(),
               "Size mismatch between ops and value columns");
  for (gdf_size_type i = 0; i < values.num_columns(); ++i) {
    // TODO Add more checks here, i.e., can't compute sum of non-arithemtic
    // types
    if ((ops[i] == SUM) and
        (values.get_column(i)->dtype == GDF_STRING_CATEGORY)) {
      CUDF_FAIL(
          "Cannot compute SUM aggregation of GDF_STRING_CATEGORY column.");
    }
  }
}

/**---------------------------------------------------------------------------*
 * @brief Deteremines target gdf_dtypes to use for combinations of source
 * gdf_dtypes and aggregation operations.
 *
 * Given vectors of source gdf_dtypes and corresponding aggregation operations
 * to be performed on that type, returns a vector of gdf_dtypes to use to store
 * the result of the aggregation operations.
 *
 * @param source_dtypes The source types
 * @param op The aggregation operations
 * @return Target gdf_dtypes to use for the target aggregation columns
 *---------------------------------------------------------------------------**/
inline std::vector<gdf_dtype> target_dtypes(
    std::vector<gdf_dtype> const& source_dtypes,
    std::vector<operators> const& ops) {
  std::vector<gdf_dtype> output_dtypes(source_dtypes.size());

  std::transform(
      source_dtypes.begin(), source_dtypes.end(), ops.begin(),
      output_dtypes.begin(), [](gdf_dtype source_dtype, operators op) {
        gdf_dtype t =
            cudf::type_dispatcher(source_dtype, target_type_mapper{}, op);
        CUDF_EXPECTS(
            t != GDF_invalid,
            "Invalid combination of input type and aggregation operation.");
        return t;
      });

  return output_dtypes;
}

/**---------------------------------------------------------------------------*
 * @brief Dispatched functor to initialize a column with the identity of an
 *aggregation operation.
 *---------------------------------------------------------------------------**/
struct identity_initializer {
  template <typename T>
  T get_identity(operators op) {
    switch (op) {
      case SUM:
        return corresponding_functor_t<SUM>::identity<T>();
      case MIN:
        return corresponding_functor_t<MIN>::identity<T>();
      case MAX:
        return corresponding_functor_t<MAX>::identity<T>();
      case COUNT:
        return corresponding_functor_t<COUNT>::identity<T>();
      default:
        CUDF_FAIL("Invalid aggregation operation.");
    }
  }

  template <typename T>
  void operator()(gdf_column const& col, operators op,
                  cudaStream_t stream = 0) {
    T* typed_data = static_cast<T*>(col.data);
    thrust::fill(rmm::exec_policy(stream)->on(stream), typed_data,
                 typed_data + col.size, get_identity<T>(op));

    // For COUNT operator, initialize column's bitmask to be all valid
    if ((nullptr != col.valid) and (COUNT == op)) {
      CUDA_TRY(cudaMemsetAsync(
          col.valid, 0xff,
          sizeof(gdf_valid_type) * gdf_valid_allocation_size(col.size),
          stream));
    }
  }
};

/**---------------------------------------------------------------------------*
 * @brief Initializes each column in a table with a corresponding identity value
 * of an aggregation operation.
 *
 * The `i`th column will be initialized with the identity value of the `i`th
 * aggregation operation.
 *
 * @note The validity bitmask (if not `nullptr`) for the column corresponding to
 * a COUNT operator will be initialized to all valid.
 *
 * @param table The table of columns to initialize.
 * @param operators The aggregation operations whose identity values will be
 *used to initialize the columns.
 *---------------------------------------------------------------------------**/
void initialize_with_identity(cudf::table const& table,
                              std::vector<operators> const& ops,
                              cudaStream_t stream = 0) {
  // TODO: Initialize all the columns in a single kernel instead of invoking one
  // kernel per column
  for (gdf_size_type i = 0; i < table.num_columns(); ++i) {
    gdf_column const* col = table.get_column(i);
    cudf::type_dispatcher(col->dtype, identity_initializer{}, *col, ops[i]);
  }
}

/**---------------------------------------------------------------------------*
 * @brief Compacts any GDF_STRING_CATEGORY columns in the output keys or values.
 *
 * After the groupby operation, any GDF_STRING_CATEGORY column in either the
 * keys or values may reference only a subset of the strings in the original
 * input category. This function will create a new associated NVCategory object
 * for the output GDF_STRING_CATEGORY columns whose dictionary contains only the
 * strings referenced in the output result.
 *
 * @param[in] input_keys The set of input key columns
 * @param[in/out] output_keys The set of output key columns
 * @param[in] input_values The set of input value columns
 * @param[in/out] output_values The set of output value columns
 *---------------------------------------------------------------------------**/
void update_nvcategories(table const& input_keys, table& output_keys,
                         table const& input_values, table& output_values) {
  nvcategory_gather_table(input_keys, output_keys);
  nvcategory_gather_table(input_values, output_values);
}

template <bool keys_have_nulls, bool values_have_nulls>
auto build_aggregation_map(table const& input_keys, table const& input_values,
                           device_table const& d_input_keys,
                           device_table const& d_input_values,
                           std::vector<operators> const& ops, Options options,
                           cudaStream_t stream) {
  gdf_size_type constexpr unused_key{std::numeric_limits<gdf_size_type>::max()};
  gdf_size_type constexpr unused_value{
      std::numeric_limits<gdf_size_type>::max()};
  CUDF_EXPECTS(input_keys.num_rows() < unused_key,
               "Groupby input size too large.");

  // The exact output size is unknown a priori, therefore, use the input size as
  // an upper bound
  gdf_size_type const output_size_estimate{input_keys.num_rows()};

  cudf::table sparse_output_values{
      output_size_estimate, target_dtypes(column_dtypes(input_values), ops),
      values_have_nulls, false, stream};

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
      concurrent_unordered_map<gdf_size_type, gdf_size_type, decltype(hasher),
                               decltype(rows_equal)>;

  auto map =
      std::make_unique<map_type>(compute_hash_table_size(input_keys.num_rows()),
                                 unused_key, unused_value, hasher, rows_equal);

  cudf::util::cuda::grid_config_1d grid_params{input_keys.num_rows(), 256};

  if (skip_key_rows_with_nulls) {
    auto row_bitmask{cudf::row_bitmask(input_keys, stream)};
    build_aggregation_map<true, values_have_nulls>
        <<<grid_params.num_blocks, grid_params.num_threads_per_block, 0,
           stream>>>(map.get(), d_input_keys, d_input_values,
                     *d_sparse_output_values, d_ops.data().get(),
                     row_bitmask.data().get());
  } else {
    build_aggregation_map<false, values_have_nulls>
        <<<grid_params.num_blocks, grid_params.num_threads_per_block, 0,
           stream>>>(map.get(), d_input_keys, d_input_values,
                     *d_sparse_output_values, d_ops.data().get(), nullptr);
  }
  CHECK_STREAM(stream);

  return std::make_pair(std::move(map), sparse_output_values);
}

template <bool keys_have_nulls, bool values_have_nulls, typename Map>
auto extract_results(table const& input_keys, table const& input_values,
                     device_table const& d_input_keys,
                     table const& sparse_output_values, Map* map,
                     cudaStream_t stream) {
  cudf::table output_keys{cudf::allocate_like(input_keys, true, stream)};
  cudf::table output_values{
      cudf::allocate_like(sparse_output_values, true, stream)};

  auto d_sparse_output_values =
      device_table::create(sparse_output_values, stream);

  auto d_output_keys = device_table::create(output_keys, stream);
  auto d_output_values = device_table::create(output_values, stream);

  gdf_size_type* d_result_size{nullptr};
  RMM_TRY(RMM_ALLOC(&d_result_size, sizeof(gdf_size_type), stream));
  CUDA_TRY(cudaMemsetAsync(d_result_size, 0, sizeof(gdf_size_type), stream));

  cudf::util::cuda::grid_config_1d grid_params{input_keys.num_rows(), 256};

  extract_groupby_result<keys_have_nulls, values_have_nulls>
      <<<grid_params.num_blocks, grid_params.num_threads_per_block, 0,
         stream>>>(map, d_input_keys, *d_output_keys, *d_sparse_output_values,
                   *d_output_values, d_result_size);

  CHECK_STREAM(stream);

  gdf_size_type result_size{-1};
  CUDA_TRY(cudaMemcpyAsync(&result_size, d_result_size, sizeof(gdf_size_type),
                           cudaMemcpyDeviceToHost, stream));

  // Update size and null count of output columns
  auto update_column = [result_size](gdf_column* col) {
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
 * @brief Converts a set of "compound" aggregation requests into a set of
 *"simple" aggregation requests that can be used to satisfy the compound
 *request.
 *
 * An "aggregation request" is the combination of an input `gdf_column`
 * an aggregation operation to perform on that column.
 *
 * A "compound" aggregation request is a requested aggregation operation that
 * can only be satisfied by first computing 1 or more "simple" aggregation
 * requests, and then transforming the result of the simple aggregation request
 * into the requested compound aggregation.
 *
 * For example, `AVG` is a "compound" aggregation. The request to compute an AVG
 * on a column can be satisfied via the `COUNT` and `SUM` "simple" aggregation
 * operations.
 *
 * @param compound_requests The set of compound aggregation requests
 * @return The set of corresponding simple aggregation requests that can be used
 * to satisfy the compound requests
 *---------------------------------------------------------------------------**/
std::vector<AggRequestType> compound_to_simple(
    std::vector<AggRequestType> const& compound_requests) {
  // Contructs a mapping of every value column to the minimal set of simple
  // ops to be performed on that column
  std::unordered_map<gdf_column*, std::set<operators>> columns_to_ops;
  std::for_each(
      compound_requests.begin(), compound_requests.end(),
      [&columns_to_ops](std::pair<gdf_column const*, operators> pair) {
        gdf_column* col = const_cast<gdf_column*>(pair.first);
        auto op = pair.second;
        // AVG requires computing a COUNT and SUM aggregation and then doing
        // elementwise division
        if (op == AVG) {
          columns_to_ops[col].insert(COUNT);
          columns_to_ops[col].insert(SUM);
        } else {
          columns_to_ops[col].insert(op);
        }
      });

  // Create minimal set of columns and simple operators
  std::vector<std::pair<gdf_column*, operators>> simple_requests;
  for (auto& p : columns_to_ops) {
    auto col = p.first;
    std::set<operators>& ops = p.second;
    while (not ops.empty()) {
      simple_requests.emplace_back(col, *ops.begin());
      ops.erase(ops.begin());
    }
  }
  return simple_requests;
}

/**---------------------------------------------------------------------------*
 * @brief Computes the `AVG` aggregation of a column by doing element-wise
 * division of the corresponding `SUM` and `COUNT` aggregation columns.
 *
 * @param sum The result of a `SUM` aggregation request
 * @param count The result of a `COUNT` aggregation request
 * @param stream Stream on which to perform operation
 * @return gdf_column* New column containing the result of elementwise division
 * of the sum and count columns
 *---------------------------------------------------------------------------**/
gdf_column* compute_average(gdf_column sum, gdf_column count,
                            cudaStream_t stream) {
  CUDF_EXPECTS(sum.size == count.size,
               "Size mismatch between sum and count columns.");
  gdf_column* avg = new gdf_column{};
  avg->dtype = GDF_FLOAT64;
  RMM_TRY(RMM_ALLOC(&avg->data, sizeof(double) * sum.size, stream));
  if (cudf::has_nulls(sum) or cudf::has_nulls(count)) {
    RMM_TRY(RMM_ALLOC(
        &avg->valid,
        sizeof(gdf_size_type) * gdf_valid_allocation_size(sum.size), stream));
  }
  cudf::binary_operation(avg, &sum, &count, GDF_DIV);
  return avg;
}

/**---------------------------------------------------------------------------*
 * @brief Computes the results of a set of aggregation requests from a set of
 * computed simple requests.
 *
 * Given a set of pre-computed results for simple aggregation requests, computes
 * the results of a set of (potentially compound) requests. If the simple
 * aggregation request neccessary to compute the original request is not
 *present, an exception is thrown.
 *
 * @param original_requests[in] The original set of potentially compound
 * aggregation requests
 * @param simple_requests[in] Set of simple requests generated from the original
 * requests
 * @param simple_outputs[in] Set of output aggregation columns corresponding to
 *the simple requests
 * @param stream[in] CUDA stream on which to execute
 * @return table Set of columns satisfying each of the original requests
 *---------------------------------------------------------------------------**/
table compute_original_requests(
    std::vector<AggRequestType> const& original_requests,
    std::vector<AggRequestType> const& simple_requests, table simple_outputs,
    cudaStream_t stream) {
  // Maps the requested simple aggregation to the resulting output column
  std::map<AggRequestType, gdf_column*> simple_requests_to_outputs;

  for (std::size_t i = 0; i < simple_requests.size(); ++i) {
    simple_requests_to_outputs[simple_requests[i]] =
        simple_outputs.get_column(i);
  }

  std::vector<gdf_column*> final_value_columns;

  // Iterate requests. For any compound request, compute the compound result
  // from the corresponding simple requests
  for (auto const& req : original_requests) {
    if (req.second == AVG) {
      auto found = simple_requests_to_outputs.find({req.first, SUM});
      CUDF_EXPECTS(found != simple_requests_to_outputs.end(),
                   "SUM request missing.");
      gdf_column* sum = found->second;

      found = simple_requests_to_outputs.find({req.first, COUNT});
      CUDF_EXPECTS(found != simple_requests_to_outputs.end(),
                   "COUNT request missing.");
      gdf_column* count = found->second;

      final_value_columns.push_back(compute_average(*sum, *count, stream));
    } else {
      // For non-compound requests, append the result to the final output
      // and remove it from the map
      auto found = simple_requests_to_outputs.find(req);
      CUDF_EXPECTS(found != simple_requests_to_outputs.end(),
                   "Aggregation missing!");
      final_value_columns.push_back(found->second);
      simple_requests_to_outputs.erase(req);
    }
  }

  // Any remaining columns in the `simple_outputs` are intermediary columns used
  // to satisfy a compound request that should be deleted.
  for (auto& p : simple_requests_to_outputs) {
    gdf_column_free(p.second);
    delete p.second;
  }

  return cudf::table{final_value_columns};
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
  CUDF_EXPECTS(values.num_columns() == static_cast<gdf_size_type>(ops.size()),
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
  // composition of 1 or more "simple" aggregation requests. For example, AVG is
  // satisfied via the division of the SUM by the COUNT aggregation. We
  // translate these compound requests into simple requests, and compute the
  // groupby operation for these simple requests. Later, we translate the simple
  // requests back to compound request results.
  std::vector<AggRequestType> simple_requests =
      compound_to_simple(original_requests);

  std::vector<gdf_column*> simple_values_columns;
  std::vector<operators> simple_operators;
  for (auto const& p : simple_requests) {
    simple_values_columns.push_back(const_cast<gdf_column*>(p.first));
    simple_operators.push_back(p.second);
  }

  cudf::table simple_values_table{simple_values_columns};

  auto const d_input_keys = device_table::create(keys);
  auto const d_input_values = device_table::create(simple_values_table);

  auto result = build_aggregation_map<keys_have_nulls, values_have_nulls>(
      keys, values, *d_input_keys, *d_input_values, simple_operators, options,
      stream);

  auto const map{std::move(result.first)};
  cudf::table const sparse_output_values{result.second};

  cudf::table output_keys;
  cudf::table simple_output_values;
  std::tie(output_keys, simple_output_values) =
      extract_results<keys_have_nulls, values_have_nulls>(
          keys, values, *d_input_keys, sparse_output_values, map.get(), stream);

  // If any of the original requests were compound, compute them from the
  // results of simple aggregation requests
  cudf::table final_output_values = compute_original_requests(
      original_requests, simple_requests, simple_output_values, stream);

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
                                            cudaStream_t stream) {
  // TODO Handle Empty inputs
  CUDF_EXPECTS(keys.num_rows() == values.num_rows(),
               "Size mismatch between number of rows in keys and values.");

  verify_operators(values, ops);

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
