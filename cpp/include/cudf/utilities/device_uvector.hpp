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

#include <rmm/device_uvector.hpp>

namespace cudf::experimental::detail {

/**
 * A wrapper around rmm::device_uvector that provides an additional data()
 * method that prefetches the data to the GPU if enabled.
 */
template <typename T>
class device_uvector : public rmm::device_uvector<T> {
 public:
  using rmm::device_uvector<T>::device_uvector;
  /**
   * @brief Returns a pointer to the underlying device memory.
   *
   * @return A pointer to the underlying device memory.
   */
  [[nodiscard]] typename rmm::device_uvector<T>::pointer data() noexcept
  {
    auto data = rmm::device_uvector<T>::data();
    // May not need this
    // if constexpr (std::is_integral_v<T>) {
    cudf::experimental::prefetch::detail::prefetch(
      "device_uvector::data", data, rmm::device_uvector<T>::size() * sizeof(T));
    return data;
  }

  /**
   * @brief Returns an iterator to the beginning of the allocation.
   *
   * This override is required because rmm::device_uvector<T>::data is not a
   * virtual method so the above override will not propagate. If we move
   * forward with this approach, we may need to override more methods.
   *
   * @return An iterator to the beginning of the allocation.
   */
  [[nodiscard]] typename rmm::device_uvector<T>::iterator begin() noexcept { return data(); }
};

}  // namespace cudf::experimental::detail
