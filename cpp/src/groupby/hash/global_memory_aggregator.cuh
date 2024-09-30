/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/traits.cuh>

namespace cudf::groupby::detail::hash {

template <typename Source, cudf::aggregation::Kind k, typename Enable = void>
struct update_target_element_gmem {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const
  {
    CUDF_UNREACHABLE("Invalid source type and aggregation combination.");
  }
};

template <typename Source>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::MIN,
  std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                   !cudf::is_fixed_point<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_null[source_index]) { return; }

    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::MIN>;

    Target* source_casted = reinterpret_cast<Target*>(source);
    cudf::detail::atomic_min(&target.element<Target>(target_index),
                             static_cast<Target>(source_casted[source_index]));

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::MIN,
  std::enable_if_t<cudf::is_fixed_point<Source>() &&
                   cudf::has_atomic_support<cudf::device_storage_type_t<Source>>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_null[source_index]) { return; }
    using Target              = cudf::detail::target_type_t<Source, cudf::aggregation::MIN>;
    using DeviceType          = cudf::device_storage_type_t<Target>;
    DeviceType* source_casted = reinterpret_cast<DeviceType*>(source);
    cudf::detail::atomic_min(&target.element<DeviceType>(target_index),
                             static_cast<DeviceType>(source_casted[source_index]));

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::MAX,
  std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                   !cudf::is_fixed_point<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_null[source_index]) { return; }
    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::MAX>;
    Target* source_casted = reinterpret_cast<Target*>(source);
    cudf::detail::atomic_max(&target.element<Target>(target_index),
                             static_cast<Target>(source_casted[source_index]));

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::MAX,
  std::enable_if_t<cudf::is_fixed_point<Source>() &&
                   cudf::has_atomic_support<cudf::device_storage_type_t<Source>>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_null[source_index]) { return; }
    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::MAX>;

    using DeviceType          = cudf::device_storage_type_t<Target>;
    DeviceType* source_casted = reinterpret_cast<DeviceType*>(source);
    cudf::detail::atomic_max(&target.element<DeviceType>(target_index),
                             static_cast<DeviceType>(source_casted[source_index]));

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::SUM,
  std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                   !cudf::is_fixed_point<Source>() && !cudf::is_timestamp<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_null[source_index]) { return; }
    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::SUM>;

    Target* source_casted = reinterpret_cast<Target*>(source);
    cudf::detail::atomic_add(&target.element<Target>(target_index),
                             static_cast<Target>(source_casted[source_index]));

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::SUM,
  std::enable_if_t<cudf::has_atomic_support<cudf::device_storage_type_t<Source>>() &&
                   cudf::is_fixed_point<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_null[source_index]) { return; }
    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::SUM>;

    using DeviceType          = cudf::device_storage_type_t<Target>;
    DeviceType* source_casted = reinterpret_cast<DeviceType*>(source);
    cudf::detail::atomic_add(&target.element<DeviceType>(target_index),
                             static_cast<DeviceType>(source_casted[source_index]));
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
struct update_target_from_dictionary_gmem {
  template <typename Source,
            aggregation::Kind k,
            std::enable_if_t<!is_dictionary<Source>()>* = nullptr>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    update_target_element_gmem<Source, k>{}(
      target, target_index, source_column, source, source_index, source_null);
  }
  template <typename Source,
            aggregation::Kind k,
            std::enable_if_t<is_dictionary<Source>()>* = nullptr>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
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
struct update_target_element_gmem<
  dictionary32,
  k,
  std::enable_if_t<not(k == aggregation::ARGMIN or k == aggregation::ARGMAX or
                       k == aggregation::COUNT_VALID or k == aggregation::COUNT_ALL)>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_null[source_index]) { return; }

    dispatch_type_and_aggregation(
      source_column.child(cudf::dictionary_column_view::keys_column_index).type(),
      k,
      update_target_from_dictionary_gmem{},
      target,
      target_index,
      source_column,
      source,
      source_index,
      source_null);
  }
};

// The shared memory will already have it squared
template <typename Source>
struct update_target_element_gmem<Source,
                                  cudf::aggregation::SUM_OF_SQUARES,
                                  std::enable_if_t<cudf::detail::is_product_supported<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_null[source_index]) { return; }
    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::SUM_OF_SQUARES>;

    Target* source_casted = reinterpret_cast<Target*>(source);
    Target value          = static_cast<Target>(source_casted[source_index]);

    cudf::detail::atomic_add(&target.element<Target>(target_index), value);

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source>
struct update_target_element_gmem<Source,
                                  cudf::aggregation::PRODUCT,
                                  std::enable_if_t<cudf::detail::is_product_supported<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_null[source_index]) { return; }
    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::PRODUCT>;

    Target* source_casted = reinterpret_cast<Target*>(source);
    cudf::detail::atomic_mul(&target.element<Target>(target_index),
                             static_cast<Target>(source_casted[source_index]));

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

// Assuming that the target column of COUNT_VALID, COUNT_ALL would be using fixed_width column and
// non-fixed point column
template <typename Source>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::COUNT_VALID,
  std::enable_if_t<cudf::detail::is_valid_aggregation<Source, cudf::aggregation::COUNT_VALID>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::COUNT_VALID>;

    Target* source_casted = reinterpret_cast<Target*>(source);
    cudf::detail::atomic_add(&target.element<Target>(target_index),
                             static_cast<Target>(source_casted[source_index]));

    // It is assumed the output for COUNT_VALID is initialized to be all valid
  }
};

// TODO: VALID and ALL have same code
template <typename Source>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::COUNT_ALL,
  std::enable_if_t<cudf::detail::is_valid_aggregation<Source, cudf::aggregation::COUNT_ALL>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::COUNT_ALL>;

    Target* source_casted = reinterpret_cast<Target*>(source);
    cudf::detail::atomic_add(&target.element<Target>(target_index),
                             static_cast<Target>(source_casted[source_index]));

    // It is assumed the output for COUNT_VALID is initialized to be all valid
  }
};

template <typename Source>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::ARGMAX,
  std::enable_if_t<cudf::detail::is_valid_aggregation<Source, cudf::aggregation::ARGMAX>() and
                   cudf::is_relationally_comparable<Source, Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_null[source_index]) { return; }
    using Target             = cudf::detail::target_type_t<Source, cudf::aggregation::ARGMAX>;
    Target* source_casted    = reinterpret_cast<Target*>(source);
    auto source_argmax_index = source_casted[source_index];
    auto old                 = cudf::detail::atomic_cas(
      &target.element<Target>(target_index), cudf::detail::ARGMAX_SENTINEL, source_argmax_index);
    if (old != cudf::detail::ARGMAX_SENTINEL) {
      while (source_column.element<Source>(source_argmax_index) >
             source_column.element<Source>(old)) {
        old =
          cudf::detail::atomic_cas(&target.element<Target>(target_index), old, source_argmax_index);
      }
    }

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};
template <typename Source>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::ARGMIN,
  std::enable_if_t<cudf::detail::is_valid_aggregation<Source, cudf::aggregation::ARGMIN>() and
                   cudf::is_relationally_comparable<Source, Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_null[source_index]) { return; }
    using Target             = cudf::detail::target_type_t<Source, cudf::aggregation::ARGMIN>;
    Target* source_casted    = reinterpret_cast<Target*>(source);
    auto source_argmin_index = source_casted[source_index];
    auto old                 = cudf::detail::atomic_cas(
      &target.element<Target>(target_index), cudf::detail::ARGMIN_SENTINEL, source_argmin_index);
    if (old != cudf::detail::ARGMIN_SENTINEL) {
      while (source_column.element<Source>(source_argmin_index) <
             source_column.element<Source>(old)) {
        old =
          cudf::detail::atomic_cas(&target.element<Target>(target_index), old, source_argmin_index);
      }
    }

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

struct gmem_element_aggregator {
  template <typename Source, cudf::aggregation::Kind k>
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    update_target_element_gmem<Source, k>{}(
      target, target_index, source_column, source, source_index, source_null);
  }
};

}  // namespace cudf::groupby::detail::hash
