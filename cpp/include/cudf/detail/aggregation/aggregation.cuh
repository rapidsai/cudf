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

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/table/table_device_view.cuh>

namespace cudf {
namespace experimental {
namespace detail {

template <typename Source, aggregation::Kind k, typename Enable = void>
struct update_target_element {};

template <typename Source, aggregation::Kind k>
struct update_target_element<
    Source, k, std::enable_if_t<is_valid_aggregation<Source, k>()>> {};

template <typename Source>
struct update_target_element<
    Source, aggregation::COUNT,
    std::enable_if_t<is_valid_aggregation<Source, aggregation::COUNT>()>> {};

template <typename Source>
struct element_wise_aggregator_impl {
  template <aggregation::Kind k>
  __device__ inline void operator()(mutable_column_device_view target,
                                    size_type target_index,
                                    column_device_view source,
                                    size_type source_index) {}
};

struct elementwise_aggregator {
  template <typename Source>
  __device__ inline void operator()(mutable_column_device_view target,
                                    size_type target_index,
                                    column_device_view source,
                                    size_type source_index,
                                    aggregation::Kind k) {
    return aggregation_dispatcher(k, element_wise_aggregator_impl<Source>{},
                                  target, target_index, source, source_index);
  }
};

/**
 * @brief Updates a row in `target` by performing elementwise aggregation
 * operations with a row in `source`.
 *
 * For the row in `target` specified by `target_index`, each element at `i` is
 * updated by:
 * ```c++
 * target_row[i] = aggs[i](target_row[i], source_row[i])
 * ```
 * If `source_row[i]` is null, it is skipped.
 *
 * @param target Table containing the row to update
 * @param target_index Index of the row to update in `target`
 * @param source Table containing the row used to update the row in `target`.
 * The invariant `source.num_columns() >= target.num_columns()` must hold.
 * @param source_index Index of the row to use in `source`
 * @param aggs Array of aggregations to perform between elements of the `target`
 * and `source` rows. Must contain at least `target.num_columns()` valid
 * `aggregation::Kind` values.
 */
__device__ inline void aggregate_row(mutable_table_device_view target,
                                     size_type target_index,
                                     table_device_view source,
                                     size_type source_index,
                                     aggregation::Kind* aggs) {}

}  // namespace detail
}  // namespace experimental
}  // namespace cudf