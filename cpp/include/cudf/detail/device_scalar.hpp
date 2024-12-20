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

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/resource_ref.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {

template <typename T>
class device_scalar : public rmm::device_scalar<T> {
 public:
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  ~device_scalar() = default;

// Implementation is the same as what compiler should generate
// Could not use default move constructor as 11.8 compiler fails to generate it
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  device_scalar(device_scalar&& other) noexcept
    : rmm::device_scalar<T>{std::move(other)}, bounce_buffer{std::move(other.bounce_buffer)}
  {
  }
  device_scalar& operator=(device_scalar&&) noexcept = default;

  device_scalar(device_scalar const&)            = delete;
  device_scalar& operator=(device_scalar const&) = delete;

  device_scalar() = delete;

  explicit device_scalar(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
    : rmm::device_scalar<T>(stream, mr), bounce_buffer{make_host_vector<T>(1, stream)}
  {
  }

  explicit device_scalar(
    T const& initial_value,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
    : rmm::device_scalar<T>(stream, mr), bounce_buffer{make_host_vector<T>(1, stream)}
  {
    bounce_buffer[0] = initial_value;
    cuda_memcpy_async<T>(device_span<T>{this->data(), 1}, bounce_buffer, stream);
  }

  device_scalar(device_scalar const& other,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
    : rmm::device_scalar<T>(other, stream, mr), bounce_buffer{make_host_vector<T>(1, stream)}
  {
  }

  [[nodiscard]] T value(rmm::cuda_stream_view stream) const
  {
    cuda_memcpy<T>(bounce_buffer, device_span<T const>{this->data(), 1}, stream);
    return bounce_buffer[0];
  }

  void set_value_async(T const& value, rmm::cuda_stream_view stream)
  {
    bounce_buffer[0] = value;
    cuda_memcpy_async<T>(device_span<T>{this->data(), 1}, bounce_buffer, stream);
  }

  void set_value_async(T&& value, rmm::cuda_stream_view stream)
  {
    bounce_buffer[0] = std::move(value);
    cuda_memcpy_async<T>(device_span<T>{this->data(), 1}, bounce_buffer, stream);
  }

  void set_value_to_zero_async(rmm::cuda_stream_view stream) { set_value_async(T{}, stream); }

 private:
  mutable cudf::detail::host_vector<T> bounce_buffer;
};

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
