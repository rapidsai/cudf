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
struct update_target_element_shmem {
  __device__ void operator()(
    cuda::std::byte*, bool*, cudf::size_type, cudf::column_device_view, cudf::size_type) const
  {
    CUDF_UNREACHABLE("Invalid source type and aggregation combination.");
  }
};

template <typename Source>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::MIN,
  cuda::std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>()>> {
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using DeviceTarget = cudf::detail::underlying_target_t<Source, aggregation::MIN>;
    using DeviceSource = cudf::detail::underlying_source_t<Source, aggregation::MIN>;

    DeviceTarget* target_casted = reinterpret_cast<DeviceTarget*>(target);
    cudf::detail::atomic_min(&target_casted[target_index],
                             static_cast<DeviceTarget>(source.element<DeviceSource>(source_index)));

    if (!target_mask[target_index]) { target_mask[target_index] = true; }
  }
};

template <typename Source>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::MAX,
  cuda::std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>()>> {
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using DeviceTarget = cudf::detail::underlying_target_t<Source, aggregation::MAX>;
    using DeviceSource = cudf::detail::underlying_source_t<Source, aggregation::MAX>;

    DeviceTarget* target_casted = reinterpret_cast<DeviceTarget*>(target);
    cudf::detail::atomic_max(&target_casted[target_index],
                             static_cast<DeviceTarget>(source.element<DeviceSource>(source_index)));

    if (!target_mask[target_index]) { target_mask[target_index] = true; }
  }
};

template <typename Source>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::SUM,
  cuda::std::enable_if_t<cudf::is_fixed_width<Source>() && cudf::has_atomic_support<Source>() &&
                         !cudf::is_timestamp<Source>()>> {
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using DeviceTarget = cudf::detail::underlying_target_t<Source, aggregation::SUM>;
    using DeviceSource = cudf::detail::underlying_source_t<Source, aggregation::SUM>;

    DeviceTarget* target_casted = reinterpret_cast<DeviceTarget*>(target);
    cudf::detail::atomic_add(&target_casted[target_index],
                             static_cast<DeviceTarget>(source.element<DeviceSource>(source_index)));

    if (!target_mask[target_index]) { target_mask[target_index] = true; }
  }
};

template <typename Source>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::SUM_OF_SQUARES,
  cuda::std::enable_if_t<cudf::detail::is_product_supported<Source>()>> {
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::SUM_OF_SQUARES>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    auto value            = static_cast<Target>(source.element<Source>(source_index));
    cudf::detail::atomic_add(&target_casted[target_index], value * value);

    if (!target_mask[target_index]) { target_mask[target_index] = true; }
  }
};

template <typename Source>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::PRODUCT,
  cuda::std::enable_if_t<cudf::detail::is_product_supported<Source>()>> {
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::PRODUCT>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    cudf::detail::atomic_mul(&target_casted[target_index],
                             static_cast<Target>(source.element<Source>(source_index)));

    if (!target_mask[target_index]) { target_mask[target_index] = true; }
  }
};

template <typename Source>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::COUNT_VALID,
  cuda::std::enable_if_t<
    cudf::detail::is_valid_aggregation<Source, cudf::aggregation::COUNT_VALID>()>> {
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    // The nullability was checked prior to this call in the `shmem_element_aggregator` functor
    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::COUNT_VALID>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    cudf::detail::atomic_add(&target_casted[target_index], Target{1});
  }
};

template <typename Source>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::COUNT_ALL,
  cuda::std::enable_if_t<
    cudf::detail::is_valid_aggregation<Source, cudf::aggregation::COUNT_ALL>()>> {
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::COUNT_ALL>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    cudf::detail::atomic_add(&target_casted[target_index], Target{1});

    // Assumes target is already set to be valid
  }
};

template <typename Source>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::ARGMAX,
  cuda::std::enable_if_t<cudf::detail::is_valid_aggregation<Source, cudf::aggregation::ARGMAX>() and
                         cudf::is_relationally_comparable<Source, Source>()>> {
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::ARGMAX>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    auto old              = cudf::detail::atomic_cas(
      &target_casted[target_index], cudf::detail::ARGMAX_SENTINEL, source_index);
    if (old != cudf::detail::ARGMAX_SENTINEL) {
      while (source.element<Source>(source_index) > source.element<Source>(old)) {
        old = cudf::detail::atomic_cas(&target_casted[target_index], old, source_index);
      }
    }

    if (!target_mask[target_index]) { target_mask[target_index] = true; }
  }
};

template <typename Source>
struct update_target_element_shmem<
  Source,
  cudf::aggregation::ARGMIN,
  cuda::std::enable_if_t<cudf::detail::is_valid_aggregation<Source, cudf::aggregation::ARGMIN>() and
                         cudf::is_relationally_comparable<Source, Source>()>> {
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::ARGMIN>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    auto old              = cudf::detail::atomic_cas(
      &target_casted[target_index], cudf::detail::ARGMIN_SENTINEL, source_index);
    if (old != cudf::detail::ARGMIN_SENTINEL) {
      while (source.element<Source>(source_index) < source.element<Source>(old)) {
        old = cudf::detail::atomic_cas(&target_casted[target_index], old, source_index);
      }
    }

    if (!target_mask[target_index]) { target_mask[target_index] = true; }
  }
};

/**
 * @brief A functor that updates a single element in the target column stored in shared memory by
 * applying an aggregation operation to a corresponding element from a source column in global
 * memory.
 *
 * This functor can NOT be used for dictionary columns.
 *
 * This is a redundant copy replicating the behavior of `elementwise_aggregator` from
 * `cudf/detail/aggregation/device_aggregators.cuh`. The key difference is that this functor accepts
 * a pointer to raw bytes as the source, as `column_device_view` cannot yet be constructed from
 * shared memory.
 */
struct shmem_element_aggregator {
  template <typename Source, cudf::aggregation::Kind k>
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    // Check nullability for all aggregation kinds but `COUNT_ALL`
    if constexpr (k != cudf::aggregation::COUNT_ALL) {
      if (source.is_null(source_index)) { return; }
    }
    update_target_element_shmem<Source, k>{}(
      target, target_mask, target_index, source, source_index);
  }
};
}  // namespace cudf::groupby::detail::hash
