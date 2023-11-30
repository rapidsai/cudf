/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>

#include <cstdint>
#include <sstream>

namespace cudf::io::parquet {

/**
 * @brief Wrapper around a `rmm::device_scalar` for use in reporting errors that occur in
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
  rmm::device_scalar<value_type> _error_code;

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
  kernel_error(rmm::cuda_stream_view stream) : _error_code{0, stream} {}

  /**
   * @brief Return a pointer to the device memory for the error
   */
  [[nodiscard]] auto data() { return _error_code.data(); }

  /**
   * @brief Return the current value of the error
   *
   * This uses the stream used to create this instance. This does a synchronize on the stream
   * this object was instantiated with.
   */
  [[nodiscard]] auto value() const { return _error_code.value(_error_code.stream()); }

  /**
   * @brief Return a hexadecimal string representation of the current error code
   *
   * Returned string will have "0x" prepended.
   */
  [[nodiscard]] std::string str() const
  {
    std::stringstream sstream;
    sstream << std::hex << value();
    return "0x" + sstream.str();
  }
};

}  // namespace cudf::io::parquet
