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

#include "io/utilities/hostdevice_vector.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <sstream>

namespace cudf::io::parquet {

/**
 * @brief Specialized device scalar for use in reporting errors that occur in
 * kernel calls.
 *
 * The `kernel_error` object is created with a `rmm::cuda_stream_view` which is used throughout
 * the object's lifetime.
 */
class kernel_error {
 public:
  using value_type = uint32_t;
  using pointer    = value_type*;

 private:
  mutable cudf::detail::hostdevice_vector<value_type> _error_code;

 public:
  /**
   * @brief Construct a new `kernel_error` with an initial value of 0.
   *
   * Note: the initial value is set asynchronously.
   *
   * @throws `rmm::bad_alloc` if allocating the device memory for `initial_value` fails.
   * @throws `rmm::cuda_error` if copying `initial_value` to device memory fails.
   *
   * @param CUDA stream to use
   */
  kernel_error(rmm::cuda_stream_view stream) : _error_code(1, stream)
  {
    _error_code[0] = 0;
    _error_code.host_to_device_async(stream);
  }

  /**
   * @brief Return a pointer to the device memory for the error
   */
  [[nodiscard]] auto data() { return _error_code.device_ptr(); }

  /**
   * @brief Return the current value of the error
   *
   * @param stream The CUDA stream to synchronize with
   */
  [[nodiscard]] auto value_sync(rmm::cuda_stream_view stream) const
  {
    _error_code.device_to_host_sync(stream);
    return _error_code[0];
  }

  /**
   * @brief Return a hexadecimal string representation of an error code
   *
   * Returned string will have "0x" prepended.
   *
   * @param value The error code to convert to a string
   */
  [[nodiscard]] static std::string to_string(value_type value)
  {
    std::stringstream sstream;
    sstream << std::hex << value;
    return "0x" + sstream.str();
  }
};

}  // namespace cudf::io::parquet
