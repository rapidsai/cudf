/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/resource_ref.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstring>

namespace CUDF_EXPORT cudf {
namespace detail {

template <typename T>
class device_scalar : public rmm::device_scalar<T> {
 public:
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  ~device_scalar() = default;

  device_scalar(device_scalar&& other) noexcept      = default;
  device_scalar& operator=(device_scalar&&) noexcept = default;

  device_scalar(device_scalar const&)            = delete;
  device_scalar& operator=(device_scalar const&) = delete;

  device_scalar() = delete;

  explicit device_scalar(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
    : rmm::device_scalar<T>(stream, mr),
      bounce_buffer{make_pinned_vector<std::byte>(sizeof(T), stream)}
  {
  }

  explicit device_scalar(
    T const& initial_value,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
    : rmm::device_scalar<T>(stream, mr),
      bounce_buffer{make_pinned_vector<std::byte>(sizeof(T), stream)}
  {
    std::memcpy(bounce_buffer.data(), &initial_value, sizeof(T));
    copy_to_device(stream);
  }

  device_scalar(device_scalar const& other,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
    : rmm::device_scalar<T>(other, stream, mr),
      bounce_buffer{make_pinned_vector<std::byte>(sizeof(T), stream)}
  {
  }

  [[nodiscard]] T value(rmm::cuda_stream_view stream) const
  {
    cuda_memcpy<std::byte>(
      bounce_buffer,
      device_span<std::byte const>{reinterpret_cast<std::byte const*>(this->data()), sizeof(T)},
      stream);
    std::array<std::byte, sizeof(T)> value_bytes;
    std::memcpy(value_bytes.data(), bounce_buffer.data(), sizeof(T));
    return std::bit_cast<T>(value_bytes);
  }

  void set_value_async(T const& value, rmm::cuda_stream_view stream)
  {
    std::memcpy(bounce_buffer.data(), &value, sizeof(T));
    copy_to_device(stream);
  }

  void set_value_async(T&& value, rmm::cuda_stream_view stream)
  {
    std::memcpy(bounce_buffer.data(), &value, sizeof(T));
    copy_to_device(stream);
  }

  void set_value_to_zero_async(rmm::cuda_stream_view stream) { set_value_async(T{}, stream); }

 private:
  void copy_to_device(rmm::cuda_stream_view stream)
  {
    cuda_memcpy_async<std::byte>(
      device_span<std::byte>{reinterpret_cast<std::byte*>(this->data()), sizeof(T)},
      bounce_buffer,
      stream);
  }

  // Byte storage supports every trivially copyable T, including types that are not default
  // constructible or assignable.
  mutable cudf::detail::host_vector<std::byte> bounce_buffer;
};

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
