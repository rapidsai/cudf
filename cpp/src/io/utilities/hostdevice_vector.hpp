/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

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

  hostdevice_vector() {}

  hostdevice_vector(hostdevice_vector&& v) { move(std::move(v)); }
  hostdevice_vector& operator=(hostdevice_vector&& v)
  {
    move(std::move(v));
    return *this;
  }

  explicit hostdevice_vector(size_t max_size, rmm::cuda_stream_view stream)
    : hostdevice_vector(max_size, max_size, stream)
  {
  }

  explicit hostdevice_vector(size_t initial_size, size_t max_size, rmm::cuda_stream_view stream)
    : max_elements(max_size), num_elements(initial_size)
  {
    if (max_elements != 0) {
      CUDF_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&h_data), sizeof(T) * max_elements));
      d_data.resize(sizeof(T) * max_elements, stream);
    }
  }

  ~hostdevice_vector()
  {
    if (max_elements != 0) {
      [[maybe_unused]] auto const free_result = cudaFreeHost(h_data);
      assert(free_result == cudaSuccess);
    }
  }

  bool insert(const T& data)
  {
    if (num_elements < max_elements) {
      h_data[num_elements] = data;
      num_elements++;
      return true;
    }
    return false;
  }

  [[nodiscard]] size_t max_size() const noexcept { return max_elements; }
  [[nodiscard]] size_t size() const noexcept { return num_elements; }
  [[nodiscard]] size_t memory_size() const noexcept { return sizeof(T) * num_elements; }

  T& operator[](size_t i) { return h_data[i]; }
  T const& operator[](size_t i) const { return h_data[i]; }

  T* host_ptr(size_t offset = 0) { return h_data + offset; }
  T const* host_ptr(size_t offset = 0) const { return h_data + offset; }

  T* begin() { return h_data; }
  T const* begin() const { return h_data; }

  T* end() { return h_data + num_elements; }
  T const* end() const { return h_data + num_elements; }

  auto d_begin() { return static_cast<T*>(d_data.data()); }
  auto d_begin() const { return static_cast<T const*>(d_data.data()); }

  auto d_end() { return static_cast<T*>(d_data.data()) + num_elements; }
  auto d_end() const { return static_cast<T const*>(d_data.data()) + num_elements; }

  auto device_ptr(size_t offset = 0) { return static_cast<T*>(d_data.data()) + offset; }
  auto device_ptr(size_t offset = 0) const { return static_cast<T const*>(d_data.data()) + offset; }

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
    CUDF_EXPECTS(element_index < size(), "Attempt to access out of bounds element.");
    T value;
    CUDF_CUDA_TRY(cudaMemcpyAsync(&value,
                                  reinterpret_cast<T const*>(d_data.data()) + element_index,
                                  sizeof(value),
                                  cudaMemcpyDefault,
                                  stream.value()));
    stream.synchronize();
    return value;
  }

  operator cudf::device_span<T>() { return {device_ptr(), max_elements}; }
  operator cudf::device_span<T const>() const { return {device_ptr(), max_elements}; }

  operator cudf::host_span<T>() { return {h_data, max_elements}; }
  operator cudf::host_span<T const>() const { return {h_data, max_elements}; }

  void host_to_device(rmm::cuda_stream_view stream, bool synchronize = false)
  {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      d_data.data(), h_data, memory_size(), cudaMemcpyHostToDevice, stream.value()));
    if (synchronize) { stream.synchronize(); }
  }

  void device_to_host(rmm::cuda_stream_view stream, bool synchronize = false)
  {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      h_data, d_data.data(), memory_size(), cudaMemcpyDeviceToHost, stream.value()));
    if (synchronize) { stream.synchronize(); }
  }

 private:
  void move(hostdevice_vector&& v)
  {
    stream       = v.stream;
    max_elements = v.max_elements;
    num_elements = v.num_elements;
    h_data       = v.h_data;
    d_data       = std::move(v.d_data);

    v.max_elements = 0;
    v.num_elements = 0;
    v.h_data       = nullptr;
  }

  rmm::cuda_stream_view stream{cudf::default_stream_value};
  size_t max_elements{};
  size_t num_elements{};
  T* h_data{};
  rmm::device_buffer d_data{};
};

namespace cudf {
namespace detail {

/**
 * @brief Wrapper around hostdevice_vector to enable two-dimensional indexing.
 *
 * Does not incur additional allocations.
 */
template <typename T>
class hostdevice_2dvector {
 public:
  hostdevice_2dvector(size_t rows, size_t columns, rmm::cuda_stream_view stream)
    : _size{rows, columns}, _data{rows * columns, stream}
  {
  }

  operator device_2dspan<T>() { return {_data.device_ptr(), _size}; }
  operator device_2dspan<T const>() const { return {_data.device_ptr(), _size}; }

  device_2dspan<T> device_view() { return static_cast<device_2dspan<T>>(*this); }
  device_2dspan<T> device_view() const { return static_cast<device_2dspan<T const>>(*this); }

  operator host_2dspan<T>() { return {_data.host_ptr(), _size}; }
  operator host_2dspan<T const>() const { return {_data.host_ptr(), _size}; }

  host_2dspan<T> host_view() { return static_cast<host_2dspan<T>>(*this); }
  host_2dspan<T> host_view() const { return static_cast<host_2dspan<T const>>(*this); }

  host_span<T> operator[](size_t row)
  {
    return {_data.host_ptr() + host_2dspan<T>::flatten_index(row, 0, _size), _size.second};
  }

  host_span<T const> operator[](size_t row) const
  {
    return {_data.host_ptr() + host_2dspan<T>::flatten_index(row, 0, _size), _size.second};
  }

  auto size() const noexcept { return _size; }
  auto count() const noexcept { return _size.first * _size.second; }
  auto is_empty() const noexcept { return count() == 0; }

  T* base_host_ptr(size_t offset = 0) { return _data.host_ptr(offset); }
  T* base_device_ptr(size_t offset = 0) { return _data.device_ptr(offset); }

  T const* base_host_ptr(size_t offset = 0) const { return _data.host_ptr(offset); }

  T const* base_device_ptr(size_t offset = 0) const { return _data.device_ptr(offset); }

  size_t memory_size() const noexcept { return _data.memory_size(); }

  void host_to_device(rmm::cuda_stream_view stream, bool synchronize = false)
  {
    _data.host_to_device(stream, synchronize);
  }

  void device_to_host(rmm::cuda_stream_view stream, bool synchronize = false)
  {
    _data.device_to_host(stream, synchronize);
  }

 private:
  hostdevice_vector<T> _data;
  typename host_2dspan<T>::size_type _size;
};

}  // namespace detail
}  // namespace cudf
