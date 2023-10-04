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

#include <cstdint>

namespace cudf::io::parquet {

/**
 * @brief Return a pointer to the global error scalar.
 *
 * This pointer is valid only on the device.
 *
 * @return Device pointer to the global error scalar.
 */
int32_t* get_error();

/**
 * @brief Return the value stored in the global error scalar.
 *
 * This will use the stream set in `set_error_stream`.
 *
 * @return The value of the global error scalar.
 */
int32_t get_error_code();

/**
 * @brief Returns the current error value as a hex string.
 * @return The current error value as a hext string.
 */
std::string get_error_string();

/**
 * @brief Reset the global error scalar to 0 and set the stream used for error reporting.
 *
 * This should be called before passing the error code to a kernel since this is a shared
 * resource.
 *
 * @param stream CUDA stream to use for error reporting.
 */
void reset_error_code(rmm::cuda_stream_view stream);

}  // namespace cudf::io::parquet
