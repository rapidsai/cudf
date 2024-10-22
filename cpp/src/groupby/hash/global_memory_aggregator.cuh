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

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/device_aggregators.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/utilities/traits.cuh>

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

namespace cudf::groupby::detail::hash {
template <typename Source, cudf::aggregation::Kind k, typename Enable = void>
struct update_target_element_gmem {
  __device__ void operator()(cudf::mutable_column_device_view,
                             cudf::size_type,
                             cudf::column_device_view,
                             cuda::std::byte*,
                             cudf::size_type) const noexcept
  {
    CUDF_UNREACHABLE("Invalid source type and aggregation combination.");
  }
};

template <typename Source>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::MIN,
  cuda::std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             cuda::std::byte* source,
                             cudf::size_type source_index) const noexcept
  {
    using DeviceType          = cudf::detail::underlying_target_t<Source, aggregation::MIN>;
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
  cuda::std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             cuda::std::byte* source,
                             cudf::size_type source_index) const noexcept
  {
    using DeviceType          = cudf::detail::underlying_target_t<Source, aggregation::MAX>;
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
  cuda::std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                         !cudf::is_timestamp<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             cuda::std::byte* source,
                             cudf::size_type source_index) const noexcept
  {
    using DeviceType          = cudf::detail::underlying_target_t<Source, aggregation::SUM>;
    DeviceType* source_casted = reinterpret_cast<DeviceType*>(source);
    cudf::detail::atomic_add(&target.element<DeviceType>(target_index),
                             static_cast<DeviceType>(source_casted[source_index]));

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

// The shared memory will already have it squared
template <typename Source>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::SUM_OF_SQUARES,
  cuda::std::enable_if_t<cudf::detail::is_product_supported<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             cuda::std::byte* source,
                             cudf::size_type source_index) const noexcept
  {
    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::SUM_OF_SQUARES>;

    Target* source_casted = reinterpret_cast<Target*>(source);
    Target value          = static_cast<Target>(source_casted[source_index]);

    cudf::detail::atomic_add(&target.element<Target>(target_index), value);

    if (target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::PRODUCT,
  cuda::std::enable_if_t<cudf::detail::is_product_supported<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             cuda::std::byte* source,
                             cudf::size_type source_index) const noexcept
  {
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
  cuda::std::enable_if_t<
    cudf::detail::is_valid_aggregation<Source, cudf::aggregation::COUNT_VALID>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             cuda::std::byte* source,
                             cudf::size_type source_index) const noexcept
  {
    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::COUNT_VALID>;

    Target* source_casted = reinterpret_cast<Target*>(source);
    cudf::detail::atomic_add(&target.element<Target>(target_index),
                             static_cast<Target>(source_casted[source_index]));

    // It is assumed the output for COUNT_VALID is initialized to be all valid
  }
};

template <typename Source>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::COUNT_ALL,
  cuda::std::enable_if_t<
    cudf::detail::is_valid_aggregation<Source, cudf::aggregation::COUNT_ALL>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             cuda::std::byte* source,
                             cudf::size_type source_index) const noexcept
  {
    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::COUNT_ALL>;

    Target* source_casted = reinterpret_cast<Target*>(source);
    cudf::detail::atomic_add(&target.element<Target>(target_index),
                             static_cast<Target>(source_casted[source_index]));

    // It is assumed the output for COUNT_ALL is initialized to be all valid
  }
};

template <typename Source>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::ARGMAX,
  cuda::std::enable_if_t<cudf::detail::is_valid_aggregation<Source, cudf::aggregation::ARGMAX>() and
                         cudf::is_relationally_comparable<Source, Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             cuda::std::byte* source,
                             cudf::size_type source_index) const noexcept
  {
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
  cuda::std::enable_if_t<cudf::detail::is_valid_aggregation<Source, cudf::aggregation::ARGMIN>() and
                         cudf::is_relationally_comparable<Source, Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             cuda::std::byte* source,
                             cudf::size_type source_index) const noexcept
  {
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

/**
 * @brief A functor that updates a single element in the target column stored in global memory by
 * applying an aggregation operation to a corresponding element from a source column in shared
 * memory.
 *
 * This functor can NOT be used for dictionary columns.
 *
 * This is a redundant copy replicating the behavior of `elementwise_aggregator` from
 * `cudf/detail/aggregation/device_aggregators.cuh`. The key difference is that this functor accepts
 * a pointer to raw bytes as the source, as `column_device_view` cannot yet be constructed from
 * shared memory.
 */
struct gmem_element_aggregator {
  template <typename Source, cudf::aggregation::Kind k>
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             cuda::std::byte* source,
                             bool* source_mask,
                             cudf::size_type source_index) const noexcept
  {
    // Early exit for all aggregation kinds since shared memory aggregation of
    // `COUNT_ALL` is always valid
    if (!source_mask[source_index]) { return; }

    update_target_element_gmem<Source, k>{}(
      target, target_index, source_column, source, source_index);
  }
};
}  // namespace cudf::groupby::detail::hash
