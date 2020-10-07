/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <string>

#include <cufile.h>

#include <cudf/io/datasource.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace io {

/**
 * @brief Class that provides RAII for file handling.
 */
class file_wrapper {
  int const fd       = -1;
  long mutable _size = -1;

 public:
  explicit file_wrapper(std::string const &filepath, int flags);
  explicit file_wrapper(std::string const &filepath, int flags, mode_t mode);
  ~file_wrapper();
  long size() const;
  auto desc() const { return fd; }
};

/**
 * @brief Class that provides RAII for cuFile file registration.
 */
struct cufile_registered_file {
  CUfileHandle_t handle = nullptr;
  explicit cufile_registered_file(int fd);
  ~cufile_registered_file();
};

/**
 * @brief Base class for cuFile input/output.
 *
 * Contains the file handles and common API for cuFile input and output classes.
 */
class cufile_io_base {
 public:
  cufile_io_base(std::string const &filepath, int flags)
    : file(filepath, flags), cf_file{file.desc()}
  {
  }
  cufile_io_base(std::string const &filepath, int flags, mode_t mode)
    : file(filepath, flags, mode), cf_file{file.desc()}
  {
  }

  virtual ~cufile_io_base() = default;

  /**
   * @brief Returns an estimate of whether the cuFile operation is the optimal option.
   *
   * @param size Read/write operation size, in bytes.
   * @return Whether a cuFile operation with the given size is expected to be faster than a host
   * read + H2D copy
   */
  static bool is_cufile_io_preferred(size_t size) { return size > op_size_threshold; }

 protected:
  /**
   * @brief The read/write size above which cuFile is faster then host read + copy
   *
   * This may not be the optimal threshold for all systems. `is_cufile_io_preferred` can use a
   * different logic based on the system config.
   */
  static constexpr size_t op_size_threshold = 128 << 10;
  file_wrapper const file;
  cufile_registered_file const cf_file;
};

/**
 * @brief Adapter for the `cuFileRead` API.
 *
 * Exposes APIs to read directly from a file into device memory.
 */
class cufile_input final : public cufile_io_base {
 public:
  cufile_input(std::string const &filepath);

  /**
   * @brief Reads into a new device buffer.
   */
  std::unique_ptr<datasource::buffer> read(size_t offset, size_t size);

  /**
   * @brief Reads into existing device memory.
   *
   * Returns the number of bytes read.
   */
  size_t read(size_t offset, size_t size, uint8_t *dst);
};

/**
 * @brief Adapter for the `cuFileWrite` API.
 *
 * Exposes an API to write directly into a file from device memory.
 */
class cufile_output final : public cufile_io_base {
 public:
  cufile_output(std::string const &filepath);

  /**
   * @brief Writes the data from a device buffer into a file.
   */
  void write(void const *data, size_t offset, size_t size);
};

};  // namespace io
};  // namespace cudf