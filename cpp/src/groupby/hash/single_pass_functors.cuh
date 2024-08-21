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
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/utilities/traits.cuh>

namespace cudf::groupby::detail::hash {

template <typename Source,
          cudf::aggregation::Kind k,
          bool target_has_nulls,
          bool source_has_nulls,
          typename Enable = void>
struct update_target_element_gmem {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    CUDF_UNREACHABLE("Invalid source type and aggregation combination.");
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::MIN,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                   !cudf::is_fixed_point<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_has_nulls and source_null[source_index]) { return; }

    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::MIN>;

    Target* source_casted = reinterpret_cast<Target*>(source);
    cudf::detail::atomic_min(&target.element<Target>(target_index),
                             static_cast<Target>(source_casted[source_index]));

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::MIN,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::is_fixed_point<Source>() &&
                   cudf::has_atomic_support<cudf::device_storage_type_t<Source>>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_has_nulls and source_null[source_index]) { return; }
    using Target              = cudf::detail::target_type_t<Source, cudf::aggregation::MIN>;
    using DeviceType          = cudf::device_storage_type_t<Target>;
    DeviceType* source_casted = reinterpret_cast<DeviceType*>(source);
    cudf::detail::atomic_min(&target.element<DeviceType>(target_index),
                             static_cast<DeviceType>(source_casted[source_index]));

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::MAX,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                   !cudf::is_fixed_point<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_has_nulls and source_null[source_index]) { return; }
    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::MAX>;
    Target* source_casted = reinterpret_cast<Target*>(source);
    cudf::detail::atomic_max(&target.element<Target>(target_index),
                             static_cast<Target>(source_casted[source_index]));

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::MAX,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::is_fixed_point<Source>() &&
                   cudf::has_atomic_support<cudf::device_storage_type_t<Source>>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_has_nulls and source_null[source_index]) { return; }
    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::MAX>;

    using DeviceType          = cudf::device_storage_type_t<Target>;
    DeviceType* source_casted = reinterpret_cast<DeviceType*>(source);
    cudf::detail::atomic_max(&target.element<DeviceType>(target_index),
                             static_cast<DeviceType>(source_casted[source_index]));

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::SUM,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                   !cudf::is_fixed_point<Source>() && !cudf::is_timestamp<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_has_nulls and source_null[source_index]) { return; }
    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::SUM>;

    Target* source_casted = reinterpret_cast<Target*>(source);
    cudf::detail::atomic_add(&target.element<Target>(target_index),
                             static_cast<Target>(source_casted[source_index]));

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::SUM,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::has_atomic_support<cudf::device_storage_type_t<Source>>() &&
                   cudf::is_fixed_point<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_has_nulls and source_null[source_index]) { return; }
    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::SUM>;

    using DeviceType          = cudf::device_storage_type_t<Target>;
    DeviceType* source_casted = reinterpret_cast<DeviceType*>(source);
    cudf::detail::atomic_add(&target.element<DeviceType>(target_index),
                             static_cast<DeviceType>(source_casted[source_index]));
    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

// The shared memory will already have it squared
template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_gmem<Source,
                                  cudf::aggregation::SUM_OF_SQUARES,
                                  target_has_nulls,
                                  source_has_nulls,
                                  std::enable_if_t<cudf::detail::is_product_supported<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_has_nulls and source_null[source_index]) { return; }
    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::SUM_OF_SQUARES>;

    Target* source_casted = reinterpret_cast<Target*>(source);
    Target value          = static_cast<Target>(source_casted[source_index]);

    cudf::detail::atomic_add(&target.element<Target>(target_index), value);

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_gmem<Source,
                                  cudf::aggregation::PRODUCT,
                                  target_has_nulls,
                                  source_has_nulls,
                                  std::enable_if_t<cudf::detail::is_product_supported<Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_has_nulls and source_null[source_index]) { return; }
    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::PRODUCT>;

    Target* source_casted = reinterpret_cast<Target*>(source);
    cudf::detail::atomic_mul(&target.element<Target>(target_index),
                             static_cast<Target>(source_casted[source_index]));

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

// Assuming that the target column of COUNT_VALID, COUNT_ALL would be using fixed_width column and
// non-fixed point column
template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::COUNT_VALID,
  target_has_nulls,
  source_has_nulls,
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
template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::COUNT_ALL,
  target_has_nulls,
  source_has_nulls,
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

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::ARGMAX,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::detail::is_valid_aggregation<Source, cudf::aggregation::ARGMAX>() and
                   cudf::is_relationally_comparable<Source, Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_has_nulls and source_null[source_index]) { return; }
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

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};
template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_gmem<
  Source,
  cudf::aggregation::ARGMIN,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::detail::is_valid_aggregation<Source, cudf::aggregation::ARGMIN>() and
                   cudf::is_relationally_comparable<Source, Source>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    if (source_has_nulls and source_null[source_index]) { return; }
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

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <bool target_has_nulls = true, bool source_has_nulls = true>
struct gmem_element_aggregator {
  template <typename Source, cudf::aggregation::Kind k>
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index,
                             cudf::column_device_view source_column,
                             std::byte* source,
                             cudf::size_type source_index,
                             bool* source_null) const noexcept
  {
    update_target_element_gmem<Source, k, target_has_nulls, source_has_nulls>{}(
      target, target_index, source_column, source, source_index, source_null);
  }
};

template <typename Source,
          cudf::aggregation::Kind k,
          bool target_has_nulls,
          bool source_has_nulls,
          typename Enable = void>
struct update_target_element_shmem {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    CUDF_UNREACHABLE("Invalid source type and aggregation combination.");
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::MIN,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                   !cudf::is_fixed_point<Source>()>> {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::MIN>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    cudf::detail::atomic_min(&target_casted[target_index],
                             static_cast<Target>(source.element<Source>(source_index)));

    if (target_has_nulls and target_null[target_index]) { target_null[target_index] = false; }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::MIN,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::is_fixed_point<Source>() &&
                   cudf::has_atomic_support<cudf::device_storage_type_t<Source>>()>> {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target       = cudf::detail::target_type_t<Source, cudf::aggregation::MIN>;
    using DeviceTarget = cudf::device_storage_type_t<Target>;
    using DeviceSource = cudf::device_storage_type_t<Source>;

    DeviceTarget* target_casted = reinterpret_cast<DeviceTarget*>(target);
    cudf::detail::atomic_min(&target_casted[target_index],
                             static_cast<DeviceTarget>(source.element<DeviceSource>(source_index)));
    if (target_has_nulls and target_null[target_index]) { target_null[target_index] = false; }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::MAX,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                   !cudf::is_fixed_point<Source>()>> {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::MAX>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    cudf::detail::atomic_max(&target_casted[target_index],
                             static_cast<Target>(source.element<Source>(source_index)));
    if (target_has_nulls and target_null[target_index]) { target_null[target_index] = false; }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::MAX,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::is_fixed_point<Source>() &&
                   cudf::has_atomic_support<cudf::device_storage_type_t<Source>>()>> {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::MAX>;

    using DeviceTarget = cudf::device_storage_type_t<Target>;
    using DeviceSource = cudf::device_storage_type_t<Source>;

    DeviceTarget* target_casted = reinterpret_cast<DeviceTarget*>(target);
    cudf::detail::atomic_max(&target_casted[target_index],
                             static_cast<DeviceTarget>(source.element<DeviceSource>(source_index)));

    if (target_has_nulls and target_null[target_index]) { target_null[target_index] = false; }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::SUM,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                   !cudf::is_fixed_point<Source>() && !cudf::is_timestamp<Source>()>> {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::SUM>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    cudf::detail::atomic_add(&target_casted[target_index],
                             static_cast<Target>(source.element<Source>(source_index)));

    if (target_has_nulls and target_null[target_index]) { target_null[target_index] = false; }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::SUM,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::has_atomic_support<cudf::device_storage_type_t<Source>>() &&
                   cudf::is_fixed_point<Source>()>> {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target = cudf::detail::target_type_t<Source, cudf::aggregation::SUM>;

    using DeviceTarget = cudf::device_storage_type_t<Target>;
    using DeviceSource = cudf::device_storage_type_t<Source>;

    DeviceTarget* target_casted = reinterpret_cast<DeviceTarget*>(target);
    cudf::detail::atomic_add(&target_casted[target_index],
                             static_cast<DeviceTarget>(source.element<DeviceSource>(source_index)));

    if (target_has_nulls and target_null[target_index]) { target_null[target_index] = false; }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_shmem<Source,
                                   cudf::aggregation::SUM_OF_SQUARES,
                                   target_has_nulls,
                                   source_has_nulls,
                                   std::enable_if_t<cudf::detail::is_product_supported<Source>()>> {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::SUM_OF_SQUARES>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    auto value            = static_cast<Target>(source.element<Source>(source_index));
    cudf::detail::atomic_add(&target_casted[target_index], value * value);

    if (target_has_nulls and target_null[target_index]) { target_null[target_index] = false; }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_shmem<Source,
                                   cudf::aggregation::PRODUCT,
                                   target_has_nulls,
                                   source_has_nulls,
                                   std::enable_if_t<cudf::detail::is_product_supported<Source>()>> {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::PRODUCT>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    cudf::detail::atomic_mul(&target_casted[target_index],
                             static_cast<Target>(source.element<Source>(source_index)));

    if (target_has_nulls and target_null[target_index]) { target_null[target_index] = false; }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::COUNT_VALID,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::detail::is_valid_aggregation<Source, cudf::aggregation::COUNT_VALID>()>> {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::COUNT_VALID>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    cudf::detail::atomic_add(&target_casted[target_index], Target{1});
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::COUNT_ALL,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::detail::is_valid_aggregation<Source, cudf::aggregation::COUNT_ALL>()>> {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::COUNT_ALL>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    cudf::detail::atomic_add(&target_casted[target_index], Target{1});

    // Assumes target is already set to be valid
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::ARGMAX,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::detail::is_valid_aggregation<Source, cudf::aggregation::ARGMAX>() and
                   cudf::is_relationally_comparable<Source, Source>()>> {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::ARGMAX>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    auto old              = cudf::detail::atomic_cas(
      &target_casted[target_index], cudf::detail::ARGMAX_SENTINEL, source_index);
    if (old != cudf::detail::ARGMAX_SENTINEL) {
      while (source.element<Source>(source_index) > source.element<Source>(old)) {
        old = cudf::detail::atomic_cas(&target_casted[target_index], old, source_index);
      }
    }

    if (target_has_nulls and target_null[target_index]) { target_null[target_index] = false; }
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::ARGMIN,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<cudf::detail::is_valid_aggregation<Source, cudf::aggregation::ARGMIN>() and
                   cudf::is_relationally_comparable<Source, Source>()>> {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::ARGMIN>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    auto old              = cudf::detail::atomic_cas(
      &target_casted[target_index], cudf::detail::ARGMIN_SENTINEL, source_index);
    if (old != cudf::detail::ARGMIN_SENTINEL) {
      while (source.element<Source>(source_index) < source.element<Source>(old)) {
        old = cudf::detail::atomic_cas(&target_casted[target_index], old, source_index);
      }
    }

    if (target_has_nulls and target_null[target_index]) { target_null[target_index] = false; }
  }
};

template <bool target_has_nulls = true, bool source_has_nulls = true>
struct shmem_element_aggregator {
  template <typename Source, cudf::aggregation::Kind k>
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    update_target_element_shmem<Source, k, target_has_nulls, source_has_nulls>{}(
      target, target_index, target_null, source, source_index);
  }
};

template <typename T, cudf::aggregation::Kind k>
__device__ constexpr bool is_supported()
{
  return cudf::is_fixed_width<T>() and
         ((k == cudf::aggregation::SUM) or (k == cudf::aggregation::MIN) or
          (k == cudf::aggregation::MAX) or (k == cudf::aggregation::COUNT_VALID) or
          (k == cudf::aggregation::COUNT_ALL) or (k == cudf::aggregation::ARGMAX) or
          (k == cudf::aggregation::ARGMIN) or (k == cudf::aggregation::SUM_OF_SQUARES) or
          (k == cudf::aggregation::STD) or (k == cudf::aggregation::VARIANCE) or
          (k == cudf::aggregation::PRODUCT) and cudf::detail::is_product_supported<T>());
}

template <typename T, cudf::aggregation::Kind k>
__device__ std::enable_if_t<not std::is_same_v<cudf::detail::corresponding_operator_t<k>, void>, T>
identity_from_operator()
{
  using DeviceType = cudf::device_storage_type_t<T>;
  return cudf::detail::corresponding_operator_t<k>::template identity<DeviceType>();
}

template <typename T, cudf::aggregation::Kind k, typename Enable = void>
__device__ std::enable_if_t<std::is_same_v<cudf::detail::corresponding_operator_t<k>, void>, T>
identity_from_operator()
{
  CUDF_UNREACHABLE("Unable to get identity/sentinel from device operator");
}

template <typename T, cudf::aggregation::Kind k>
__device__ T get_identity()
{
  if ((k == cudf::aggregation::ARGMAX) || (k == cudf::aggregation::ARGMIN)) {
    if constexpr (cudf::is_timestamp<T>())
      return k == cudf::aggregation::ARGMAX
               ? T{typename T::duration(cudf::detail::ARGMAX_SENTINEL)}
               : T{typename T::duration(cudf::detail::ARGMIN_SENTINEL)};
    else {
      using DeviceType = cudf::device_storage_type_t<T>;
      return k == cudf::aggregation::ARGMAX
               ? static_cast<DeviceType>(cudf::detail::ARGMAX_SENTINEL)
               : static_cast<DeviceType>(cudf::detail::ARGMIN_SENTINEL);
    }
  }
  return identity_from_operator<T, k>();
}

template <typename Target, cudf::aggregation::Kind k, typename Enable = void>
struct initialize_target_element {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null) const noexcept
  {
    CUDF_UNREACHABLE("Invalid source type and aggregation combination.");
  }
};

// TODO: are the conditions correctly checked?
template <typename Target, cudf::aggregation::Kind k>
struct initialize_target_element<Target, k, std::enable_if_t<is_supported<Target, k>()>> {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null) const noexcept
  {
    using DeviceType            = cudf::device_storage_type_t<Target>;
    DeviceType* target_casted   = reinterpret_cast<DeviceType*>(target);
    target_casted[target_index] = get_identity<DeviceType, k>();

    if (k == cudf::aggregation::COUNT_ALL || k == cudf::aggregation::COUNT_VALID) {
      target_null[target_index] = false;
    } else {
      target_null[target_index] = true;
    }
  }
};

struct initialize_shmem {
  template <typename Target, cudf::aggregation::Kind k>
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null) const noexcept
  {
    // TODO: typecasting work for every datatype

    initialize_target_element<Target, k>{}(target, target_index, target_null);
  }
};

template <typename Target, cudf::aggregation::Kind k, typename Enable = void>
struct initialize_target_element_gmem {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index) const noexcept
  {
    CUDF_UNREACHABLE("Invalid source type and aggregation combination.");
  }
};

template <typename Target, cudf::aggregation::Kind k>
struct initialize_target_element_gmem<
  Target,
  k,
  std::enable_if_t<is_supported<Target, k>() && cudf::is_fixed_width<Target>() &&
                   !cudf::is_fixed_point<Target>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index) const noexcept
  {
    using DeviceType                     = cudf::device_storage_type_t<Target>;
    target.element<Target>(target_index) = get_identity<DeviceType, k>();
  }
};

template <typename Target, cudf::aggregation::Kind k>
struct initialize_target_element_gmem<
  Target,
  k,
  std::enable_if_t<is_supported<Target, k>() && cudf::is_fixed_point<Target>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index) const noexcept
  {
    using DeviceType                         = cudf::device_storage_type_t<Target>;
    target.element<DeviceType>(target_index) = get_identity<DeviceType, k>();
  }
};

struct initialize_gmem {
  template <typename Target, cudf::aggregation::Kind k>
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index) const noexcept
  {
    initialize_target_element_gmem<Target, k>{}(target, target_index);
  }
};

struct initialize_sparse_table {
  cudf::size_type const* row_indices;
  cudf::mutable_table_device_view sparse_table;
  cudf::aggregation::Kind const* __restrict__ aggs;
  initialize_sparse_table(cudf::size_type const* row_indices,
                          cudf::mutable_table_device_view sparse_table,
                          cudf::aggregation::Kind const* aggs)
    : row_indices(row_indices), sparse_table(sparse_table), aggs(aggs)
  {
  }
  __device__ void operator()(cudf::size_type i)
  {
    auto key_idx = row_indices[i];
    for (auto col_idx = 0; col_idx < sparse_table.num_columns(); col_idx++) {
      cudf::detail::dispatch_type_and_aggregation(sparse_table.column(col_idx).type(),
                                                  aggs[col_idx],
                                                  initialize_gmem{},
                                                  sparse_table.column(col_idx),
                                                  key_idx);
    }
  }
};

template <typename SetType>
struct compute_direct_aggregates {
  SetType set;
  cudf::table_device_view input_values;
  cudf::mutable_table_device_view output_values;
  cudf::aggregation::Kind const* __restrict__ aggs;
  cudf::size_type* block_cardinality;
  int stride;
  compute_direct_aggregates(SetType set,
                            cudf::table_device_view input_values,
                            cudf::mutable_table_device_view output_values,
                            cudf::aggregation::Kind const* aggs,
                            cudf::size_type* block_cardinality,
                            int stride)
    : set(set),
      input_values(input_values),
      output_values(output_values),
      aggs(aggs),
      block_cardinality(block_cardinality),
      stride(stride)
  {
  }
  __device__ void operator()(cudf::size_type i)
  {
    int block_id = (i % stride) / GROUPBY_BLOCK_SIZE;
    if (block_cardinality[block_id] >= GROUPBY_CARDINALITY_THRESHOLD) {
      auto const result = set.insert_and_find(i);
      cudf::detail::aggregate_row<true, true>(output_values, *result.first, input_values, i, aggs);
    }
  }
};

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

}  // namespace cudf::groupby::detail::hash
