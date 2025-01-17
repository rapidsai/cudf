/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <cudf/io/datasource.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <string>

namespace cudf {
namespace io {
namespace detail {

[[noreturn]] void throw_on_file_open_failure(std::string const& filepath, bool is_create);

// Call before any cuFile API calls to ensure the CUDA context is initialized.
void force_init_cuda_context();

/**
 * @brief Class that provides RAII for file handling.
 */
class file_wrapper {
  int fd       = -1;
  size_t _size = 0;

 public:
  explicit file_wrapper(std::string const& filepath, int flags, mode_t mode = 0);
  ~file_wrapper();
  [[nodiscard]] auto size() const { return _size; }
  [[nodiscard]] auto desc() const { return fd; }
};

/**
 * @brief Byte range to be read/written in a single operation.
 */
CUDF_EXPORT struct file_io_slice {
  size_t offset;
  size_t size;
};

}  // namespace detail
}  // namespace io
}  // namespace cudf
