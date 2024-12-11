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

#include "hostdevice_span.hpp"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/host/host_memory_resource.hpp>

namespace cudf::detail {

/**
 * @brief A helper class that wraps fixed-length device memory for the GPU, and
 * a mirror host pinned memory for the CPU.
 *
 * This abstraction allocates a specified fixed chunk of device memory that can
 * initialized upfront, or gradually initialized as required.
 * The host-side memory can be used to manipulate data on the CPU before and
 * after operating on the same data on the GPU.
 */
template <typename T>
class hostdevice_vector {
 public:
  using value_type = T;

  hostdevice_vector() : hostdevice_vector(0, cudf::get_default_stream()) {}

  explicit hostdevice_vector(size_t size, rmm::cuda_stream_view stream)
    : hostdevice_vector(size, size, stream)
  {
  }

  explicit hostdevice_vector(size_t initial_size, size_t max_size, rmm::cuda_stream_view stream)
    : h_data{make_pinned_vector_async<T>(0, stream)}, d_data(max_size, stream)
  {
    CUDF_EXPECTS(initial_size <= max_size, "initial_size cannot be larger than max_size");

    h_data.reserve(max_size);
    h_data.resize(initial_size);
  }

  void push_back(T const& data)
  {
    CUDF_EXPECTS(size() < capacity(),
                 "Cannot insert data into hostdevice_vector because capacity has been exceeded.");
    h_data.push_back(data);
  }

  [[nodiscard]] size_t capacity() const noexcept { return d_data.size(); }
  [[nodiscard]] size_t size() const noexcept { return h_data.size(); }
  [[nodiscard]] size_t size_bytes() const noexcept { return sizeof(T) * size(); }
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }

  [[nodiscard]] T& operator[](size_t i) { return h_data[i]; }
  [[nodiscard]] T const& operator[](size_t i) const { return h_data[i]; }

  [[nodiscard]] T* host_ptr(size_t offset = 0) { return h_data.data() + offset; }
  [[nodiscard]] T const* host_ptr(size_t offset = 0) const { return h_data.data() + offset; }

  [[nodiscard]] T* begin() { return host_ptr(); }
  [[nodiscard]] T const* begin() const { return host_ptr(); }

  [[nodiscard]] T* end() { return host_ptr(size()); }
  [[nodiscard]] T const* end() const { return host_ptr(size()); }

  [[nodiscard]] T& front() { return h_data.front(); }
  [[nodiscard]] T const& front() const { return front(); }

  [[nodiscard]] T& back() { return h_data.back(); }
  [[nodiscard]] T const& back() const { return back(); }

  [[nodiscard]] T* device_ptr(size_t offset = 0) { return d_data.data() + offset; }
  [[nodiscard]] T const* device_ptr(size_t offset = 0) const { return d_data.data() + offset; }

  [[nodiscard]] T* d_begin() { return device_ptr(); }
  [[nodiscard]] T const* d_begin() const { return device_ptr(); }

  [[nodiscard]] T* d_end() { return device_ptr(size()); }
  [[nodiscard]] T const* d_end() const { return device_ptr(size()); }

  /**
   * @brief Returns the specified element from device memory
   *
   * @note This function incurs a device to host memcpy and should be used sparingly.
   * @note This function synchronizes `stream`.
   *
   * @throws rmm::out_of_range exception if `element_index >= size()`
   *
   * @param element_index Index of the desired element
   * @param stream The stream on which to perform the copy
   * @return The value of the specified element
   */
  [[nodiscard]] T element(std::size_t element_index, rmm::cuda_stream_view stream) const
  {
    return d_data.element(element_index, stream);
  }

  operator cudf::host_span<T>() { return host_span<T>{h_data}.subspan(0, size()); }
  operator cudf::host_span<T const>() const
  {
    return host_span<T const>{h_data}.subspan(0, size());
  }

  operator cudf::device_span<T>() { return {device_ptr(), size()}; }
  operator cudf::device_span<T const>() const { return {device_ptr(), size()}; }

  void host_to_device_async(rmm::cuda_stream_view stream)
  {
    cuda_memcpy_async<T>(d_data, h_data, stream);
  }

  void host_to_device_sync(rmm::cuda_stream_view stream) { cuda_memcpy<T>(d_data, h_data, stream); }

  void device_to_host_async(rmm::cuda_stream_view stream)
  {
    cuda_memcpy_async<T>(h_data, d_data, stream);
  }

  void device_to_host_sync(rmm::cuda_stream_view stream) { cuda_memcpy<T>(h_data, d_data, stream); }

  /**
   * @brief Converts a hostdevice_vector into a hostdevice_span.
   *
   * @return A typed hostdevice_span of the hostdevice_vector's data
   */
  [[nodiscard]] operator hostdevice_span<T>() { return {host_span<T>{h_data}, device_ptr()}; }

  [[nodiscard]] operator hostdevice_span<T const>() const
  {
    return {host_span<T const>{h_data}, device_ptr()};
  }

 private:
  cudf::detail::host_vector<T> h_data;
  rmm::device_uvector<T> d_data;
};

/**
 * @brief Wrapper around hostdevice_vector to enable two-dimensional indexing.
 *
 * Does not incur additional allocations.
 */
template <typename T>
class hostdevice_2dvector {
 public:
  hostdevice_2dvector() : hostdevice_2dvector(0, 0, cudf::get_default_stream()) {}

  hostdevice_2dvector(size_t rows, size_t columns, rmm::cuda_stream_view stream)
    : _data{rows * columns, stream}, _size{rows, columns}
  {
  }

  operator device_2dspan<T>() { return {device_span<T>{_data}, _size.second}; }
  operator device_2dspan<T const>() const { return {device_span<T const>{_data}, _size.second}; }

  device_2dspan<T> device_view() { return static_cast<device_2dspan<T>>(*this); }
  [[nodiscard]] device_2dspan<T const> device_view() const
  {
    return static_cast<device_2dspan<T const>>(*this);
  }

  operator host_2dspan<T>() { return {host_span<T>{_data}, _size.second}; }
  operator host_2dspan<T const>() const { return {host_span<T const>{_data}, _size.second}; }

  host_2dspan<T> host_view() { return static_cast<host_2dspan<T>>(*this); }
  [[nodiscard]] host_2dspan<T const> host_view() const
  {
    return static_cast<host_2dspan<T const>>(*this);
  }

  host_span<T> operator[](size_t row)
  {
    return host_span<T>{_data}.subspan(row * _size.second, _size.second);
  }

  host_span<T const> operator[](size_t row) const
  {
    return host_span<T const>{_data}.subspan(row * _size.second, _size.second);
  }

  [[nodiscard]] auto size() const noexcept { return _size; }
  [[nodiscard]] auto count() const noexcept { return _size.first * _size.second; }
  [[nodiscard]] auto is_empty() const noexcept { return count() == 0; }

  T* base_host_ptr(size_t offset = 0) { return _data.host_ptr(offset); }
  T* base_device_ptr(size_t offset = 0) { return _data.device_ptr(offset); }

  [[nodiscard]] T const* base_host_ptr(size_t offset = 0) const { return _data.host_ptr(offset); }

  [[nodiscard]] T const* base_device_ptr(size_t offset = 0) const
  {
    return _data.device_ptr(offset);
  }

  [[nodiscard]] size_t size_bytes() const noexcept { return _data.size_bytes(); }

  void host_to_device_async(rmm::cuda_stream_view stream) { _data.host_to_device_async(stream); }
  void host_to_device_sync(rmm::cuda_stream_view stream) { _data.host_to_device_sync(stream); }

  void device_to_host_async(rmm::cuda_stream_view stream) { _data.device_to_host_async(stream); }
  void device_to_host_sync(rmm::cuda_stream_view stream) { _data.device_to_host_sync(stream); }

 private:
  hostdevice_vector<T> _data;
  typename host_2dspan<T>::size_type _size;
};

}  // namespace cudf::detail
