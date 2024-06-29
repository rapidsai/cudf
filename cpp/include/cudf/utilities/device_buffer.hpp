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

#include <cudf/utilities/prefetch.hpp>

#include <rmm/device_buffer.hpp>

namespace cudf::experimental::buffer::detail {

/**
 * A wrapper around rmm::device_buffer that provides an additional data()
 * method that prefetches the data to the GPU if enabled.
 */
class device_buffer : public rmm::device_buffer {
 public:
  using rmm::device_buffer::device_buffer;
  /**
   * @brief Const pointer to the device memory allocation
   *
   * @return A pointer to the underlying device memory.
   */
  [[nodiscard]] void const* data() const noexcept
  {
    auto const* data = rmm::device_buffer::data();
    cudf::experimental::prefetch::detail::prefetch(
      "device_buffer::data_const", data, rmm::device_buffer::size());
    return data;
  }

  /**
   * @brief Pointer to the device memory allocation
   *
   * @return A pointer to the underlying device memory.
   */
  void* data() noexcept
  {
    auto data = rmm::device_buffer::data();
    cudf::experimental::prefetch::detail::prefetch(
      "device_buffer::data", data, rmm::device_buffer::size());
    return data;
  }

  /**
   * @brief Move assignment operator
   * @param other The device_buffer to move from
   * @return A reference to this device_buffer
   */
  device_buffer& operator=(rmm::device_buffer&& other) noexcept
  {
    rmm::device_buffer::operator=(std::move(other));
    return *this;
  }
};

}  // namespace cudf::experimental::buffer::detail
