/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf::detail {

template <typename T>
class hostdevice_span {
 public:
  using value_type = T;

  hostdevice_span()                       = default;
  ~hostdevice_span()                      = default;
  hostdevice_span(hostdevice_span const&) = default;  ///< Copy constructor
  hostdevice_span(hostdevice_span&&)      = default;  ///< Move constructor

  hostdevice_span(T* cpu_data, T* gpu_data, size_t size)
    : _size(size), _device_data(gpu_data), _host_data(cpu_data)
  {
  }

  /// Constructor from container
  /// @param in The container to construct the span from
  template <typename C,
            // Only supported containers of types convertible to T
            std::enable_if_t<std::is_convertible_v<
              std::remove_pointer_t<decltype(std::declval<C&>().host_ptr())> (*)[],
              T (*)[]>>* = nullptr>
  constexpr hostdevice_span(C& in) : hostdevice_span(in.host_ptr(), in.device_ptr(), in.size())
  {
  }

  /// Constructor from const container
  /// @param in The container to construct the span from
  template <typename C,
            // Only supported containers of types convertible to T
            std::enable_if_t<std::is_convertible_v<
              std::remove_pointer_t<decltype(std::declval<C&>().host_ptr())> (*)[],
              T (*)[]>>* = nullptr>
  constexpr hostdevice_span(C const& in)
    : hostdevice_span(in.host_ptr(), in.device_ptr(), in.size())
  {
  }

  /**
   * @brief Copy assignment operator.
   *
   * @return Reference to this hostdevice_span.
   */
  constexpr hostdevice_span& operator=(hostdevice_span const&) noexcept = default;

  /**
   * @brief Converts a hostdevice view into a device span.
   *
   * @tparam T The device span type.
   * @return A typed device span of the hostdevice view's data.
   */
  [[nodiscard]] operator cudf::device_span<T>() { return {_device_data, size()}; }

  /**
   * @brief Converts a hostdevice view into a device span of const data.
   *
   * @tparam T The device span type.
   * @return A const typed device span of the hostdevice view's data.
   */
  [[nodiscard]] operator cudf::device_span<T const>() const { return {_device_data, size()}; }

  /**
   * @brief Returns the underlying device data.
   *
   * @tparam T The type to cast to
   * @return T const* Typed pointer to underlying data
   */
  [[nodiscard]] T* device_ptr(size_t offset = 0) const noexcept { return _device_data + offset; }

  /**
   * @brief Return first element in device data.
   *
   * @tparam T The desired type
   * @return T const* Pointer to the first element
   */
  [[nodiscard]] T* device_begin() const noexcept { return device_ptr(); }

  /**
   * @brief Return one past the last element in device_data.
   *
   * @tparam T The desired type
   * @return T const* Pointer to one past the last element
   */
  [[nodiscard]] T* device_end() const noexcept { return device_begin() + size(); }

  /**
   * @brief Converts a hostdevice_span into a host span.
   *
   * @tparam T The host span type.
   * @return A typed host span of the hostdevice_span's data.
   */
  [[nodiscard]] operator cudf::host_span<T>() const noexcept
  {
    return cudf::host_span<T>(_host_data, size());
  }

  /**
   * @brief Returns the underlying host data.
   *
   * @tparam T The type to cast to
   * @return T* Typed pointer to underlying data
   */
  [[nodiscard]] T* host_ptr(size_t offset = 0) const noexcept { return _host_data + offset; }

  /**
   * @brief Return first element in host data.
   *
   * @tparam T The desired type
   * @return T const* Pointer to the first element
   */
  [[nodiscard]] T* host_begin() const noexcept { return host_ptr(); }

  /**
   * @brief Return one past the last elementin host data.
   *
   * @tparam T The desired type
   * @return T const* Pointer to one past the last element
   */
  [[nodiscard]] T* host_end() const noexcept { return host_begin() + size(); }

  /**
   * @brief Returns the number of elements in the view
   *
   * @return The number of elements in the view
   */
  [[nodiscard]] std::size_t size() const noexcept { return _size; }

  /**
   * @brief Returns true if `size()` returns zero, or false otherwise
   *
   * @return True if `size()` returns zero, or false otherwise
   */
  [[nodiscard]] bool is_empty() const noexcept { return size() == 0; }

  [[nodiscard]] size_t size_bytes() const noexcept { return sizeof(T) * size(); }

  [[nodiscard]] T& operator[](size_t i) { return _host_data[i]; }
  [[nodiscard]] T const& operator[](size_t i) const { return _host_data[i]; }

  /**
   * @brief Obtains a hostdevice_span that is a view over the `count` elements of this
   * hostdevice_span starting at offset
   *
   * @param offset The offset of the first element in the subspan
   * @param count The number of elements in the subspan
   * @return A subspan of the sequence, of requested count and offset
   */
  constexpr hostdevice_span<T> subspan(size_t offset, size_t count) const noexcept
  {
    return hostdevice_span<T>(_host_data + offset, _device_data + offset, count);
  }

  void host_to_device_async(rmm::cuda_stream_view stream)
  {
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(device_ptr(), host_ptr(), size_bytes(), cudaMemcpyDefault, stream.value()));
  }

  void host_to_device_sync(rmm::cuda_stream_view stream)
  {
    host_to_device_async(stream);
    stream.synchronize();
  }

  void device_to_host_async(rmm::cuda_stream_view stream)
  {
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(host_ptr(), device_ptr(), size_bytes(), cudaMemcpyDefault, stream.value()));
  }

  void device_to_host_sync(rmm::cuda_stream_view stream)
  {
    device_to_host_async(stream);
    stream.synchronize();
  }

 private:
  size_t _size{};     ///< Number of elements
  T* _device_data{};  ///< Pointer to device memory containing elements
  T* _host_data{};    ///< Pointer to host memory containing elements
};

}  // namespace cudf::detail
