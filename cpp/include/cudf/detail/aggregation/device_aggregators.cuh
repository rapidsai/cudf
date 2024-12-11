/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.cuh>

#include <cuda/std/type_traits>

namespace cudf::detail {
/// Checks if an aggregation kind needs to operate on the underlying storage type
template <aggregation::Kind k>
__device__ constexpr bool uses_underlying_type()
{
  return k == aggregation::MIN or k == aggregation::MAX or k == aggregation::SUM;
}

/// Gets the underlying target type for the given source type and aggregation kind
template <typename Source, aggregation::Kind k>
using underlying_target_t =
  cuda::std::conditional_t<uses_underlying_type<k>(),
                           cudf::device_storage_type_t<cudf::detail::target_type_t<Source, k>>,
                           cudf::detail::target_type_t<Source, k>>;

/// Gets the underlying source type for the given source type and aggregation kind
template <typename Source, aggregation::Kind k>
using underlying_source_t =
  cuda::std::conditional_t<uses_underlying_type<k>(), cudf::device_storage_type_t<Source>, Source>;

template <typename Source, aggregation::Kind k, typename Enable = void>
struct update_target_element {
  __device__ void operator()(mutable_column_device_view,
                             size_type,
                             column_device_view,
                             size_type) const noexcept
  {
    CUDF_UNREACHABLE("Invalid source type and aggregation combination.");
  }
};

template <typename Source>
struct update_target_element<
  Source,
  aggregation::MIN,
  cuda::std::enable_if_t<is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                         !is_fixed_point<Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::MIN>;
    cudf::detail::atomic_min(&target.element<Target>(target_index),
                             static_cast<Target>(source.element<Source>(source_index)));

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source>
struct update_target_element<
  Source,
  aggregation::MIN,
  cuda::std::enable_if_t<is_fixed_point<Source>() &&
                         cudf::has_atomic_support<device_storage_type_t<Source>>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target       = target_type_t<Source, aggregation::MIN>;
    using DeviceTarget = device_storage_type_t<Target>;
    using DeviceSource = device_storage_type_t<Source>;

    cudf::detail::atomic_min(&target.element<DeviceTarget>(target_index),
                             static_cast<DeviceTarget>(source.element<DeviceSource>(source_index)));

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source>
struct update_target_element<
  Source,
  aggregation::MAX,
  cuda::std::enable_if_t<is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                         !is_fixed_point<Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::MAX>;
    cudf::detail::atomic_max(&target.element<Target>(target_index),
                             static_cast<Target>(source.element<Source>(source_index)));

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source>
struct update_target_element<
  Source,
  aggregation::MAX,
  cuda::std::enable_if_t<is_fixed_point<Source>() &&
                         cudf::has_atomic_support<device_storage_type_t<Source>>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target       = target_type_t<Source, aggregation::MAX>;
    using DeviceTarget = device_storage_type_t<Target>;
    using DeviceSource = device_storage_type_t<Source>;

    cudf::detail::atomic_max(&target.element<DeviceTarget>(target_index),
                             static_cast<DeviceTarget>(source.element<DeviceSource>(source_index)));

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source>
struct update_target_element<
  Source,
  aggregation::SUM,
  cuda::std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                         !cudf::is_fixed_point<Source>() && !cudf::is_timestamp<Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::SUM>;
    cudf::detail::atomic_add(&target.element<Target>(target_index),
                             static_cast<Target>(source.element<Source>(source_index)));

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source>
struct update_target_element<
  Source,
  aggregation::SUM,
  cuda::std::enable_if_t<is_fixed_point<Source>() &&
                         cudf::has_atomic_support<device_storage_type_t<Source>>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target       = target_type_t<Source, aggregation::SUM>;
    using DeviceTarget = device_storage_type_t<Target>;
    using DeviceSource = device_storage_type_t<Source>;

    cudf::detail::atomic_add(&target.element<DeviceTarget>(target_index),
                             static_cast<DeviceTarget>(source.element<DeviceSource>(source_index)));

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

/**
 * @brief Function object to update a single element in a target column using
 * the dictionary key addressed by the specific index.
 *
 * SFINAE is used to prevent recursion for dictionary type. Dictionary keys cannot be a
 * dictionary.
 *
 */
struct update_target_from_dictionary {
  template <typename Source,
            aggregation::Kind k,
            cuda::std::enable_if_t<!is_dictionary<Source>()>* = nullptr>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    update_target_element<Source, k>{}(target, target_index, source, source_index);
  }
  template <typename Source,
            aggregation::Kind k,
            cuda::std::enable_if_t<is_dictionary<Source>()>* = nullptr>
  __device__ void operator()(mutable_column_device_view,
                             size_type,
                             column_device_view,
                             size_type) const noexcept
  {
  }
};

/**
 * @brief Specialization function for dictionary type and aggregations.
 *
 * The `source` column is a dictionary type. This functor de-references the
 * dictionary's keys child column and maps the input source index through
 * the dictionary's indices child column to pass to the `update_target_element`
 * in the above `update_target_from_dictionary` using the type-dispatcher to
 * resolve the keys column type.
 *
 * `update_target_element( target, target_index, source.keys(), source.indices()[source_index] )`
 */
template <aggregation::Kind k>
struct update_target_element<
  dictionary32,
  k,
  cuda::std::enable_if_t<not(k == aggregation::ARGMIN or k == aggregation::ARGMAX or
                             k == aggregation::COUNT_VALID or k == aggregation::COUNT_ALL)>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    dispatch_type_and_aggregation(
      source.child(cudf::dictionary_column_view::keys_column_index).type(),
      k,
      update_target_from_dictionary{},
      target,
      target_index,
      source.child(cudf::dictionary_column_view::keys_column_index),
      static_cast<cudf::size_type>(source.element<dictionary32>(source_index)));
  }
};

template <typename Source>
struct update_target_element<Source,
                             aggregation::SUM_OF_SQUARES,
                             cuda::std::enable_if_t<is_product_supported<Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::SUM_OF_SQUARES>;
    auto value   = static_cast<Target>(source.element<Source>(source_index));
    cudf::detail::atomic_add(&target.element<Target>(target_index), value * value);
    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source>
struct update_target_element<Source,
                             aggregation::PRODUCT,
                             cuda::std::enable_if_t<is_product_supported<Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::PRODUCT>;
    cudf::detail::atomic_mul(&target.element<Target>(target_index),
                             static_cast<Target>(source.element<Source>(source_index)));
    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source>
struct update_target_element<
  Source,
  aggregation::COUNT_VALID,
  cuda::std::enable_if_t<is_valid_aggregation<Source, aggregation::COUNT_VALID>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::COUNT_VALID>;
    cudf::detail::atomic_add(&target.element<Target>(target_index), Target{1});

    // It is assumed the output for COUNT_VALID is initialized to be all valid
  }
};

template <typename Source>
struct update_target_element<
  Source,
  aggregation::COUNT_ALL,
  cuda::std::enable_if_t<is_valid_aggregation<Source, aggregation::COUNT_ALL>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::COUNT_ALL>;
    cudf::detail::atomic_add(&target.element<Target>(target_index), Target{1});

    // It is assumed the output for COUNT_ALL is initialized to be all valid
  }
};

template <typename Source>
struct update_target_element<
  Source,
  aggregation::ARGMAX,
  cuda::std::enable_if_t<is_valid_aggregation<Source, aggregation::ARGMAX>() and
                         cudf::is_relationally_comparable<Source, Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::ARGMAX>;
    auto old     = cudf::detail::atomic_cas(
      &target.element<Target>(target_index), ARGMAX_SENTINEL, source_index);
    if (old != ARGMAX_SENTINEL) {
      while (source.element<Source>(source_index) > source.element<Source>(old)) {
        old = cudf::detail::atomic_cas(&target.element<Target>(target_index), old, source_index);
      }
    }

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source>
struct update_target_element<
  Source,
  aggregation::ARGMIN,
  cuda::std::enable_if_t<is_valid_aggregation<Source, aggregation::ARGMIN>() and
                         cudf::is_relationally_comparable<Source, Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::ARGMIN>;
    auto old     = cudf::detail::atomic_cas(
      &target.element<Target>(target_index), ARGMIN_SENTINEL, source_index);
    if (old != ARGMIN_SENTINEL) {
      while (source.element<Source>(source_index) < source.element<Source>(old)) {
        old = cudf::detail::atomic_cas(&target.element<Target>(target_index), old, source_index);
      }
    }

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

/**
 * @brief Function object to update a single element in a target column by
 * performing an aggregation operation with a single element from a source
 * column.
 */
struct elementwise_aggregator {
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if constexpr (k != cudf::aggregation::COUNT_ALL) {
      if (source.is_null(source_index)) { return; }
    }
    update_target_element<Source, k>{}(target, target_index, source, source_index);
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
 *
 * This function only supports aggregations that can be done in a "single pass",
 * i.e., given an initial value `R`, the aggregation `op` can be computed on a series
 * of elements `e[i] for i in [0,n)` by computing `R = op(e[i],R)` for any order
 * of the values of `i`.
 *
 * The initial value and validity of `R` depends on the aggregation:
 * SUM: 0 and NULL
 * MIN: Max value of type and NULL
 * MAX: Min value of type and NULL
 * COUNT_VALID: 0 and VALID
 * COUNT_ALL:   0 and VALID
 * ARGMAX: `ARGMAX_SENTINEL` and NULL
 * ARGMIN: `ARGMIN_SENTINEL` and NULL
 *
 * It is required that the elements of `target` be initialized with the corresponding
 * initial values and validity specified above.
 *
 * Handling of null elements in both `source` and `target` depends on the aggregation:
 * SUM, MIN, MAX, ARGMIN, ARGMAX:
 *  - `source`: Skipped
 *  - `target`: Updated from null to valid upon first successful aggregation
 * COUNT_VALID, COUNT_ALL:
 *  - `source`: Skipped
 *  - `target`: Cannot be null
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
                                     aggregation::Kind const* aggs)
{
  for (auto i = 0; i < target.num_columns(); ++i) {
    dispatch_type_and_aggregation(source.column(i).type(),
                                  aggs[i],
                                  elementwise_aggregator{},
                                  target.column(i),
                                  target_index,
                                  source.column(i),
                                  source_index);
  }
}
}  // namespace cudf::detail
