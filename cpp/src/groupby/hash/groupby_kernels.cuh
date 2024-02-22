/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#pragma once

#include "multi_pass_kernels.cuh"

#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/utilities/bit.hpp>

#include <thrust/pair.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace hash {
/**
 * @brief Computes single-pass aggregations and store results into a sparse `output_values` table,
 * and populate `set` with indices of unique keys
 *
 * The hash set is built by inserting every row index `i` from the `keys` and `values` tables. If
 * the index was not present in the set, insert they index and then copy it to the output. If the
 * key was already present in the set, then the inserted index is aggregated with the existing row.
 * This aggregation is done for every element `j` in the row by applying aggregation operation `j`
 * between the new and existing element.
 *
 * Instead of storing the entire rows from `input_keys` and `input_values` in
 * the hashset, we instead store the row indices. For example, when inserting
 * row at index `i` from `input_keys` into the hash set, the value `i` is what
 * gets stored for the hash set's "key". It is assumed the `set` was constructed
 * with a custom comparator that uses these row indices to check for equality
 * between key rows. For example, comparing two keys `k0` and `k1` will compare
 * the two rows `input_keys[k0] ?= input_keys[k1]`
 *
 * The exact size of the result is not known a priori, but can be upper bounded
 * by the number of rows in `input_keys` & `input_values`. Therefore, it is
 * assumed `output_values` has sufficient storage for an equivalent number of
 * rows. In this way, after all rows are aggregated, `output_values` will likely
 * be "sparse", meaning that not all rows contain the result of an aggregation.
 *
 * @tparam SetType The type of the hash set device ref
 */
template <typename SetType>
struct compute_single_pass_aggs_fn {
  SetType set;
  table_device_view input_values;
  mutable_table_device_view output_values;
  aggregation::Kind const* __restrict__ aggs;
  bitmask_type const* __restrict__ row_bitmask;
  bool skip_rows_with_nulls;

  /**
   * @brief Construct a new compute_single_pass_aggs_fn functor object
   *
   * @param set_ref Hash set object to insert key,value pairs into.
   * @param input_values The table whose rows will be aggregated in the values
   * of the hash set
   * @param output_values Table that stores the results of aggregating rows of
   * `input_values`.
   * @param aggs The set of aggregation operations to perform across the
   * columns of the `input_values` rows
   * @param row_bitmask Bitmask where bit `i` indicates the presence of a null
   * value in row `i` of input keys. Only used if `skip_rows_with_nulls` is `true`
   * @param skip_rows_with_nulls Indicates if rows in `input_keys` containing
   * null values should be skipped. It `true`, it is assumed `row_bitmask` is a
   * bitmask where bit `i` indicates the presence of a null value in row `i`.
   */
  compute_single_pass_aggs_fn(SetType set,
                              table_device_view input_values,
                              mutable_table_device_view output_values,
                              aggregation::Kind const* aggs,
                              bitmask_type const* row_bitmask,
                              bool skip_rows_with_nulls)
    : set(set),
      input_values(input_values),
      output_values(output_values),
      aggs(aggs),
      row_bitmask(row_bitmask),
      skip_rows_with_nulls(skip_rows_with_nulls)
  {
  }

  __device__ void operator()(size_type i)
  {
    if (not skip_rows_with_nulls or cudf::bit_is_set(row_bitmask, i)) {
      auto const result = set.insert_and_find(i);

      cudf::detail::aggregate_row<true, true>(output_values, *result.first, input_values, i, aggs);
    }
  }
};

}  // namespace hash
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
