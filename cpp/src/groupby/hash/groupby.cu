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

#include <cudf.h>
#include <bitmask.hpp>
#include <bitmask/bit_mask.cuh>
#include <groupby.hpp>
#include <hash/concurrent_unordered_map.cuh>
#include <string/nvcategory_util.hpp>
#include <table.hpp>
#include <table/device_table.cuh>
#include <table/device_table_row_operators.cuh>
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
#include <type_traits>
#include <vector>

namespace cudf {
namespace groupby {
namespace hash {
namespace detail {

namespace {

using namespace groupby;


/**---------------------------------------------------------------------------*
 * @brief Deteremines target gdf_dtypes to use for combinations of source
 * gdf_dtypes and aggregation operations.
 *
 * Given vectors of source gdf_dtypes and corresponding aggregation operations
 * to be performed on that type, returns a vector the gdf_dtypes to use to store
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
        gdf_dtype t = cudf::type_dispatcher(source_dtype, dtype_mapper{}, op);
        CUDF_EXPECTS(
            t != GDF_invalid,
            "Invalid combination of input type and aggregation operation.");
        return t;
      });

  return output_dtypes;
}

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
 * @note The validity bitmask for the column corresponding to a COUNT operator
 * will be initialized to all valid.
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

template <bool keys_have_nulls, bool values_have_nulls>
auto compute_hash_groupby(cudf::table const& keys, cudf::table const& values,
                          std::vector<operators> const& ops, Options options,
                          cudaStream_t stream) {
  gdf_size_type constexpr unused_key{std::numeric_limits<gdf_size_type>::max()};
  gdf_size_type constexpr unused_value{
      std::numeric_limits<gdf_size_type>::max()};
  CUDF_EXPECTS(keys.num_rows() < unused_key, "Groupby input size too large.");

  // The exact output size is unknown a priori, therefore, use the input size as
  // an upper bound
  gdf_size_type const output_size_estimate{keys.num_rows()};

  // TODO Do we always need to allocate the output bitmask?
  cudf::table sparse_output_values{output_size_estimate,
                                   target_dtypes(column_dtypes(values), ops),
                                   values_have_nulls, stream};

  initialize_with_identity(sparse_output_values, ops, stream);

  auto const d_input_keys = device_table::create(keys);
  auto const d_input_values = device_table::create(values);
  auto d_sparse_output_values = device_table::create(sparse_output_values);
  rmm::device_vector<operators> d_ops(ops);

  // If we ignore null keys, then nulls are not equivalent
  bool const null_keys_are_equal{not options.ignore_null_keys};
  bool const skip_rows_with_nulls{keys_have_nulls and not null_keys_are_equal};

  row_hasher<keys_have_nulls> hasher{*d_input_keys};
  row_equality_comparator<keys_have_nulls> rows_equal{
      *d_input_keys, *d_input_keys, null_keys_are_equal};

  using map_type =
      concurrent_unordered_map<gdf_size_type, gdf_size_type, decltype(hasher),
                               decltype(rows_equal)>;

  auto map =
      std::make_unique<map_type>(compute_hash_table_size(keys.num_rows()),
                                 unused_key, unused_value, hasher, rows_equal);

  cudf::util::cuda::grid_config_1d grid_params{keys.num_rows(), 256};

  if (skip_rows_with_nulls) {
    auto row_bitmask{cudf::row_bitmask(keys, stream)};
    build_aggregation_table<true, values_have_nulls>
        <<<grid_params.num_blocks, grid_params.num_threads_per_block, 0,
           stream>>>(map.get(), *d_input_keys, *d_input_values,
                     *d_sparse_output_values, d_ops.data().get(),
                     row_bitmask.data().get());
  } else {
    build_aggregation_table<false, values_have_nulls>
        <<<grid_params.num_blocks, grid_params.num_threads_per_block, 0,
           stream>>>(map.get(), *d_input_keys, *d_input_values,
                     *d_sparse_output_values, d_ops.data().get(), nullptr);
  }

  // TODO Extract results

  CHECK_STREAM(stream);

  // TODO Set output key/value columns null counts
  cudf::table output_keys;
  cudf::table output_values;

  return std::make_tuple(output_keys, output_values);
}
}  // namespace

std::tuple<cudf::table, cudf::table> groupby(cudf::table const& keys,
                                             cudf::table const& values,
                                             std::vector<operators> const& ops,
                                             Options options,
                                             cudaStream_t stream) {
  cudf::table output_keys;
  cudf::table output_values;

  if (cudf::has_nulls(keys)) {
    if (cudf::has_nulls(values)) {
      std::tie(output_keys, output_values) =
          compute_hash_groupby<true, true>(keys, values, ops, options, stream);
    } else {
      std::tie(output_keys, output_values) =
          compute_hash_groupby<true, false>(keys, values, ops, options, stream);
    }
  } else {
    if (cudf::has_nulls(values)) {
      std::tie(output_keys, output_values) =
          compute_hash_groupby<false, true>(keys, values, ops, options, stream);
    } else {
      std::tie(output_keys, output_values) = compute_hash_groupby<false, false>(
          keys, values, ops, options, stream);
    }
  }

  return std::make_tuple(output_keys, output_values);
}

}  // namespace detail

std::tuple<cudf::table, cudf::table> groupby(cudf::table const& keys,
                                             cudf::table const& values,
                                             std::vector<operators> const& ops,
                                             Options options) {
  CUDF_EXPECTS(static_cast<gdf_size_type>(ops.size()) == values.num_columns(),
               "Size mismatch between ops and value columns");

  for (gdf_size_type i = 0; i < values.num_columns(); ++i) {
    if ((ops[i] == SUM) and
        (values.get_column(i)->dtype == GDF_STRING_CATEGORY)) {
      CUDF_FAIL(
          "Cannot compute SUM aggregation of GDF_STRING_CATEGORY column.");
    }
  }

  cudf::table output_keys;
  cudf::table output_values;
  std::tie(output_keys, output_values) =
      detail::groupby(keys, values, ops, options);

  // Compact NVCategory columns to contain only the strings referenced by the
  // indices in the output key/value columns
  auto gather_nvcategories = [](gdf_column const* input_column,
                                gdf_column* output_column) {
    CUDF_EXPECTS(input_column->dtype == output_column->dtype,
                 "Column type mismatch");
    if (input_column->dtype == GDF_STRING_CATEGORY) {
      auto status = nvcategory_gather(
          output_column,
          static_cast<NVCategory*>(input_column->dtype_info.category));
      CUDF_EXPECTS(status == GDF_SUCCESS, "Failed to gather NVCategory.");
    }
    return output_column;
  };

  std::transform(keys.begin(), keys.end(), output_keys.begin(),
                 output_keys.begin(), gather_nvcategories);

  std::transform(values.begin(), values.end(), output_values.begin(),
                 output_values.begin(), gather_nvcategories);

  return std::make_tuple(output_keys, output_values);
}
}  // namespace hash
}  // namespace groupby
}  // namespace cudf
