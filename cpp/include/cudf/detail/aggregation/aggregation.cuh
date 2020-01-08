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
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/detail/utilities/release_assert.cuh>
#include <cudf/table/table_device_view.cuh>

namespace cudf {
namespace experimental {
namespace detail {

template <typename Source, aggregation::Kind k, bool target_has_nulls,
          bool source_has_nulls, typename Enable = void>
struct update_target_element {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index, column_device_view source,
                             size_type source_index) const noexcept {
    release_assert(false and
                   "Invalid source type and aggregation combination.");
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<Source, aggregation::MIN, target_has_nulls,
                             source_has_nulls,
                             std::enable_if_t<is_fixed_width<Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index, column_device_view source,
                             size_type source_index) const noexcept {
    if (source_has_nulls and source.is_null(source_index)) {
      return;
    }

    using Target = target_type_t<Source, aggregation::MIN>;
    atomicMin(&target.element<Target>(target_index),
              static_cast<Target>(source.element<Source>(source_index)));

    if (target_has_nulls and target.is_null(target_index)) {
      target.set_valid(target_index);
    }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<Source, aggregation::MAX, target_has_nulls,
                             source_has_nulls,
                             std::enable_if_t<is_fixed_width<Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index, column_device_view source,
                             size_type source_index) const noexcept {
    if (source_has_nulls and source.is_null(source_index)) {
      return;
    }

    using Target = target_type_t<Source, aggregation::MAX>;
    atomicMax(&target.element<Target>(target_index),
              static_cast<Target>(source.element<Source>(source_index)));

    if (target_has_nulls and target.is_null(target_index)) {
      target.set_valid(target_index);
    }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<Source, aggregation::SUM, target_has_nulls,
                             source_has_nulls,
                             std::enable_if_t<is_fixed_width<Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index, column_device_view source,
                             size_type source_index) const noexcept {
    if (source_has_nulls and source.is_null(source_index)) {
      return;
    }

    using Target = target_type_t<Source, aggregation::SUM>;
    atomicAdd(&target.element<Target>(target_index),
              static_cast<Target>(source.element<Source>(source_index)));

    if (target_has_nulls and target.is_null(target_index)) {
      target.set_valid(target_index);
    }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<
    Source, aggregation::COUNT, target_has_nulls, source_has_nulls,
    std::enable_if_t<is_valid_aggregation<Source, aggregation::COUNT>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index, column_device_view source,
                             size_type source_index) const noexcept {

    if (source_has_nulls and source.is_null(source_index)) {
      return;
    }

    using Target = target_type_t<Source, aggregation::COUNT>;
    atomicAdd(&target.element<Target>(target_index), Target{1});

    // It is assumed the output for COUNT is initialized to be all valid
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<
    Source, aggregation::ARGMAX, target_has_nulls, source_has_nulls,
    std::enable_if_t<is_valid_aggregation<Source, aggregation::ARGMAX>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index, column_device_view source,
                             size_type source_index) const noexcept {
    if (source_has_nulls and source.is_null(source_index)) {
      return;
    }

    using Target = target_type_t<Source, aggregation::ARGMAX>;
    auto old = atomicCAS(&target.element<Target>(target_index), ARGMAX_SENTINEL,
                         source_index);
    if (old == ARGMAX_SENTINEL) {
      return;
    }

    while (source.element<Source>(source_index) > source.element<Source>(old)) {
      old = atomicCAS(&target.element<Target>(target_index), old, source_index);
    }

    if (target_has_nulls and target.is_null(target_index)) {
      target.set_valid(target_index);
    }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element<
    Source, aggregation::ARGMIN, target_has_nulls, source_has_nulls,
    std::enable_if_t<is_valid_aggregation<Source, aggregation::ARGMIN>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index, column_device_view source,
                             size_type source_index) const noexcept {
    if (source_has_nulls and source.is_null(source_index)) {
      return;
    }

    using Target = target_type_t<Source, aggregation::ARGMIN>;
    auto old = atomicCAS(&target.element<Target>(target_index), ARGMIN_SENTINEL,
                         source_index);
    if (old == ARGMIN_SENTINEL) {
      return;
    }

    while (source.element<Source>(source_index) < source.element<Source>(old)) {
      old = atomicCAS(&target.element<Target>(target_index), old, source_index);
    }

    if (target_has_nulls and target.is_null(target_index)) {
      target.set_valid(target_index);
    }
  }
};

/**
 * @brief Function object to update a single element in a target column by
 * performing an aggregation operation with a single element from a source
 * column.
 *
 * @tparam target_has_nulls Indicates presence of null elements in `target`
 * @tparam source_has_nulls Indicates presence of null elements in `source`.
 */
template <bool target_has_nulls = true, bool source_has_nulls = true>
struct elementwise_aggregator {
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index, column_device_view source,
                             size_type source_index) const noexcept {
    update_target_element<Source, k, target_has_nulls, source_has_nulls>{}(
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
 * TODO: Document null handling, expectations for initialization/validity of output columns
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
// template <bool target_has_nulls = true, bool source_has_nulls = true>
__device__ inline void aggregate_row(mutable_table_device_view target,
                                     size_type target_index,
                                     table_device_view source,
                                     size_type source_index,
                                     aggregation::Kind* aggs) {
  for (auto i = 0; i < target.num_columns(); ++i) {
    dispatch_type_and_aggregation(
        target.column(i).type(), aggs[i], elementwise_aggregator<true, true>{},
        // elementwise_aggregator<target_has_nulls, source_has_nulls>{},
        target.column(i), target_index, source.column(i), source_index);
  }
}
}  // namespace detail
}  // namespace experimental
}  // namespace cudf