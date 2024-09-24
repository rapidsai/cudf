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

#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/utilities/traits.cuh>

namespace cudf::groupby::detail::hash {

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

template <bool target_has_nulls = true>
struct update_target_from_dictionary_shmem {
  template <typename Source,
            aggregation::Kind k,
            std::enable_if_t<!is_dictionary<Source>()>* = nullptr>
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    update_target_element_shmem<Source, k, target_has_nulls, false>{}(
      target, target_index, target_null, source, source_index);
  }
  template <typename Source,
            aggregation::Kind k,
            std::enable_if_t<is_dictionary<Source>()>* = nullptr>
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
  }
};

template <aggregation::Kind k, bool target_has_nulls, bool source_has_nulls>
struct update_target_element_shmem<
  dictionary32,
  k,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<not(k == aggregation::ARGMIN or k == aggregation::ARGMAX or
                       k == aggregation::COUNT_VALID or k == aggregation::COUNT_ALL)>> {
  __device__ void operator()(std::byte* target,
                             cudf::size_type target_index,
                             bool* target_null,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    if (source_has_nulls and source.is_null(source_index)) { return; }

    dispatch_type_and_aggregation(
      source.child(cudf::dictionary_column_view::keys_column_index).type(),
      k,
      update_target_from_dictionary_shmem<target_has_nulls>{},
      target,
      target_index,
      target_null,
      source.child(cudf::dictionary_column_view::keys_column_index),
      static_cast<cudf::size_type>(source.element<dictionary32>(source_index)));
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

}  // namespace cudf::groupby::detail::hash
