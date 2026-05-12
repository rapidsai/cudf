/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/column/column_device_view.cuh>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <jit/element_storage.cuh>
#include <jit/sync.cuh>

namespace CUDF_EXPORT cudf {

template <bool has_nulls, typename element_storage_t>
__device__ void load_element(column_device_view_core const* column,
                             size_type element_index,
                             element_storage_t* storage)
{
  auto op = [&] __device__<typename T>() {
    if constexpr (!has_nulls) {
      if constexpr (storage_compatible<element_storage_t, T>) {
        auto v                               = column->template element<T>(element_index);
        *reinterpret_cast<T*>(storage->data) = v;
      }
    } else {
      if constexpr (storage_compatible<element_storage_t, cuda::std::optional<T>>) {
        auto v = column->template nullable_element<T>(element_index);
        *reinterpret_cast<cuda::std::optional<T>*>(storage->data) = v;
      }
    }
  };

  switch (column->type().id()) {
    case type_id::INT8:
    case type_id::UINT8:
    case type_id::BOOL8: op.template operator()<uint8_t>(); break;
    case type_id::INT16:
    case type_id::UINT16: op.template operator()<uint16_t>(); break;
    case type_id::INT32:
    case type_id::UINT32:
    case type_id::FLOAT32:
    case type_id::TIMESTAMP_DAYS:
    case type_id::DURATION_DAYS: op.template operator()<uint32_t>(); break;
    case type_id::INT64:
    case type_id::UINT64:
    case type_id::FLOAT64:
    case type_id::TIMESTAMP_SECONDS:
    case type_id::TIMESTAMP_MILLISECONDS:
    case type_id::TIMESTAMP_MICROSECONDS:
    case type_id::TIMESTAMP_NANOSECONDS:
    case type_id::DURATION_SECONDS:
    case type_id::DURATION_MILLISECONDS:
    case type_id::DURATION_MICROSECONDS:
    case type_id::DURATION_NANOSECONDS: op.template operator()<uint64_t>(); break;
    case type_id::DECIMAL32: op.template operator()<numeric::decimal32>(); break;
    case type_id::DECIMAL64: op.template operator()<numeric::decimal64>(); break;
    case type_id::DECIMAL128: op.template operator()<numeric::decimal128>(); break;
    case type_id::STRING: op.template operator()<string_view>(); break;
    default: CUDF_UNREACHABLE();
  }
}

template <bool has_nulls, typename element_storage_t>
__device__ void store_element(mutable_column_device_view_core const* column,
                              element_storage_t const* storage,
                              size_type element_index,
                              unsigned int active_mask)
{
  auto op = [&] __device__<typename T>() {
    if constexpr (!has_nulls) {
      if constexpr (storage_compatible<element_storage_t, T>) {
        auto v = *reinterpret_cast<T const*>(storage->data);
        column->template assign<T>(element_index, v);
      }
    } else {
      if constexpr (storage_compatible<element_storage_t, cuda::std::optional<T>>) {
        auto v = *reinterpret_cast<cuda::std::optional<T> const*>(storage->data);
        column->template assign<T>(element_index, *v);

        auto null_word = __ballot_sync(active_mask, v.has_value());
        if (column->nullable()) {
          if (warp_elect(active_mask)) {
            column->null_mask()[element_index / detail::warp_size] = null_word;
          }
        }
      }
    }
  };

  switch (column->type().id()) {
    case type_id::INT8:
    case type_id::UINT8:
    case type_id::BOOL8: op.template operator()<uint8_t>(); break;
    case type_id::INT16:
    case type_id::UINT16: op.template operator()<uint16_t>(); break;
    case type_id::INT32:
    case type_id::UINT32:
    case type_id::FLOAT32:
    case type_id::TIMESTAMP_DAYS:
    case type_id::DURATION_DAYS: op.template operator()<uint32_t>(); break;
    case type_id::INT64:
    case type_id::UINT64:
    case type_id::FLOAT64:
    case type_id::TIMESTAMP_SECONDS:
    case type_id::TIMESTAMP_MILLISECONDS:
    case type_id::TIMESTAMP_MICROSECONDS:
    case type_id::TIMESTAMP_NANOSECONDS:
    case type_id::DURATION_SECONDS:
    case type_id::DURATION_MILLISECONDS:
    case type_id::DURATION_MICROSECONDS:
    case type_id::DURATION_NANOSECONDS: op.template operator()<uint64_t>(); break;
    case type_id::DECIMAL32: op.template operator()<numeric::decimal32>(); break;
    case type_id::DECIMAL64: op.template operator()<numeric::decimal64>(); break;
    case type_id::DECIMAL128: op.template operator()<numeric::decimal128>(); break;
    default: CUDF_UNREACHABLE();
  }
}

}  // namespace CUDF_EXPORT cudf
