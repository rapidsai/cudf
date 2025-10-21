/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
#include <cudf/types.hpp>
#include <cudf/utilities/traits.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace cudf::detail {

template <typename Source, aggregation::Kind k>
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
  requires(is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
           !is_fixed_point<Source>())
struct update_target_element<Source, aggregation::MIN> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::MIN>;
    cudf::detail::atomic_min(&target.element<Target>(target_index),
                             static_cast<Target>(source.element<Source>(source_index)));
  }
};

template <typename Source>
  requires(is_fixed_point<Source>() && cudf::has_atomic_support<device_storage_type_t<Source>>())
struct update_target_element<Source, aggregation::MIN> {
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
  }
};

template <typename Source>
  requires(is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
           !is_fixed_point<Source>())
struct update_target_element<Source, aggregation::MAX> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::MAX>;
    cudf::detail::atomic_max(&target.element<Target>(target_index),
                             static_cast<Target>(source.element<Source>(source_index)));
  }
};

template <typename Source>
  requires(is_fixed_point<Source>() && cudf::has_atomic_support<device_storage_type_t<Source>>())
struct update_target_element<Source, aggregation::MAX> {
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
  }
};

template <typename Source>
  requires(cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
           !cudf::is_fixed_point<Source>() && !cudf::is_timestamp<Source>())
struct update_target_element<Source, aggregation::SUM> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::SUM>;
    cudf::detail::atomic_add(&target.element<Target>(target_index),
                             static_cast<Target>(source.element<Source>(source_index)));
  }
};

template <typename Source>
  requires(is_fixed_point<Source>() && cudf::has_atomic_support<device_storage_type_t<Source>>())
struct update_target_element<Source, aggregation::SUM> {
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
  }
};

template <typename Source>
  requires((cudf::is_integral_not_bool<Source>() && cudf::is_signed<Source>()) ||
           cudf::is_fixed_point<Source>() &&
             cudf::has_atomic_support<device_storage_type_t<Source>>())
struct update_target_element<Source, aggregation::SUM_WITH_OVERFLOW> {
  using DeviceType               = device_storage_type_t<Source>;
  static constexpr auto type_max = cuda::std::numeric_limits<DeviceType>::max();
  static constexpr auto type_min = cuda::std::numeric_limits<DeviceType>::min();

  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    auto sum_column      = target.child(0);
    auto overflow_column = target.child(1);

    auto const source_value = source.element<DeviceType>(source_index);
    auto const old_sum =
      cudf::detail::atomic_add(&sum_column.element<DeviceType>(target_index), source_value);

    // Early exit if overflow is already set
    auto bool_ref = cuda::atomic_ref<bool, cuda::thread_scope_device>{
      *(overflow_column.data<bool>() + target_index)};
    if (bool_ref.load(cuda::memory_order_relaxed)) { return; }

    // Check for overflow before performing the addition to avoid UB
    // For positive overflow: old_sum > 0, source_value > 0, and old_sum > max - source_value
    // For negative overflow: old_sum < 0, source_value < 0, and old_sum < min - source_value
    // TODO: to be replaced by CCCL equivalents once https://github.com/NVIDIA/cccl/pull/3755 is
    // ready
    auto const overflow = ((old_sum > 0 && source_value > 0 && old_sum > type_max - source_value) ||
                           (old_sum < 0 && source_value < 0 && old_sum < type_min - source_value));

    if (overflow) { cudf::detail::atomic_max(&overflow_column.element<bool>(target_index), true); }
  }
};

/**
 * @brief Specialization of update_target_element for SUM_WITH_OVERFLOW with decimal128
 *
 * Uses two-64-bit atomic operations to provide atomic addition for __int128_t.
 * This enables SUM_WITH_OVERFLOW for decimal128 on all GPU architectures.
 */
template <>
struct update_target_element<numeric::decimal128, aggregation::SUM_WITH_OVERFLOW> {
  using DeviceType               = device_storage_type_t<numeric::decimal128>;  // __int128_t
  static constexpr auto type_max = cuda::std::numeric_limits<DeviceType>::max();
  static constexpr auto type_min = cuda::std::numeric_limits<DeviceType>::min();

  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    auto sum_column      = target.child(0);
    auto overflow_column = target.child(1);

    auto const source_value = source.element<DeviceType>(source_index);

    // Use the new 128-bit atomic addition implementation
    auto const old_sum =
      cudf::detail::atomic_add(&sum_column.element<DeviceType>(target_index), source_value);

    // Early exit if overflow is already set
    auto bool_ref = cuda::atomic_ref<bool, cuda::thread_scope_device>{
      *(overflow_column.data<bool>() + target_index)};
    if (bool_ref.load(cuda::memory_order_relaxed)) { return; }

    // Check for overflow using the improved logic
    auto const overflow = ((old_sum > 0 && source_value > 0 && old_sum > type_max - source_value) ||
                           (old_sum < 0 && source_value < 0 && old_sum < type_min - source_value));

    if (overflow) { cudf::detail::atomic_max(&overflow_column.element<bool>(target_index), true); }
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
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
    requires(!is_dictionary<Source>())
  {
    update_target_element<Source, k>{}(target, target_index, source, source_index);
  }
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(mutable_column_device_view,
                             size_type,
                             column_device_view,
                             size_type) const noexcept
    requires(is_dictionary<Source>())
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
  requires(not(k == aggregation::ARGMIN or k == aggregation::ARGMAX or
               k == aggregation::COUNT_VALID or k == aggregation::COUNT_ALL))
struct update_target_element<dictionary32, k> {
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
  requires(is_product_supported<Source>())
struct update_target_element<Source, aggregation::SUM_OF_SQUARES> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::SUM_OF_SQUARES>;
    auto value   = static_cast<Target>(source.element<Source>(source_index));
    cudf::detail::atomic_add(&target.element<Target>(target_index), value * value);
  }
};

template <typename Source>
  requires(is_product_supported<Source>())
struct update_target_element<Source, aggregation::PRODUCT> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::PRODUCT>;
    cudf::detail::atomic_mul(&target.element<Target>(target_index),
                             static_cast<Target>(source.element<Source>(source_index)));
  }
};

template <typename Source>
  requires(is_valid_aggregation<Source, aggregation::COUNT_VALID>())
struct update_target_element<Source, aggregation::COUNT_VALID> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::COUNT_VALID>;
    cudf::detail::atomic_add(&target.element<Target>(target_index), Target{1});
  }
};

template <typename Source>
  requires(is_valid_aggregation<Source, aggregation::COUNT_ALL>())
struct update_target_element<Source, aggregation::COUNT_ALL> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    using Target = target_type_t<Source, aggregation::COUNT_ALL>;
    cudf::detail::atomic_add(&target.element<Target>(target_index), Target{1});
  }
};

template <typename Source>
  requires(is_valid_aggregation<Source, aggregation::ARGMAX>() &&
           cudf::is_relationally_comparable<Source, Source>())
struct update_target_element<Source, aggregation::ARGMAX> {
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
  }
};

template <typename Source>
  requires(is_valid_aggregation<Source, aggregation::ARGMIN>() &&
           cudf::is_relationally_comparable<Source, Source>())
struct update_target_element<Source, aggregation::ARGMIN> {
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
  }
};

/**
 * @brief Function object to update a single element in a target column by
 * performing an aggregation operation with a single element from a source
 * column.
 *
 * This functor only supports aggregations that can be done in a "single pass",
 * i.e., given an initial value `r = R`, the aggregation `op` can be computed on a series
 * of elements `e[i] for i in [0,n)` just by updating `r = op(r, e[i])` using any order
 * of the values `e[i]`.
 *
 * The initial value and validity of `R` depends on the specific aggregation. For example:
 *  - SUM: 0 and NULL
 *  - MIN: Max value of type and NULL
 *  - MAX: Min value of type and NULL
 *  - COUNT_VALID: 0 and VALID
 *  - COUNT_ALL:   0 and VALID
 *  - ARGMAX: `ARGMAX_SENTINEL` and NULL
 *  - ARGMIN: `ARGMIN_SENTINEL` and NULL
 *
 * It is required that the elements of `target` be initialized with the corresponding
 * initial values and validity specified above.
 *
 * Handling of null elements in both `source` and `target` depends on the aggregation. For example:
 *  - SUM, MIN, MAX, ARGMIN, ARGMAX:  `source` is skipped, `target` is updated from null to valid
 *    upon first successful aggregation
 *  - COUNT_VALID, COUNT_ALL:  `source` is skipped, `target` cannot be null
 */
struct element_aggregator {
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if constexpr (k != cudf::aggregation::COUNT_ALL) {
      if (source.is_null(source_index)) { return; }
    }

    // The output for COUNT_VALID and COUNT_ALL is initialized to be all valid
    if constexpr (!(k == cudf::aggregation::COUNT_VALID or k == cudf::aggregation::COUNT_ALL)) {
      if (target.is_null(target_index)) { target.set_valid(target_index); }
    }

    update_target_element<Source, k>{}(target, target_index, source, source_index);
  }
};

}  // namespace cudf::detail
