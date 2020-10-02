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

struct cf_file_wrapper {
  CUfileHandle_t handle = nullptr;
  explicit cf_file_wrapper(int fd);
  ~cf_file_wrapper();
};

class gds_io_base {
 public:
  gds_io_base(std::string const &filepath, int flags) : file(filepath, flags), cf_file{file.desc()}
  {
  }
  gds_io_base(std::string const &filepath, int flags, mode_t mode)
    : file(filepath, flags, mode), cf_file{file.desc()}
  {
  }

  static bool is_gds_io_preferred(size_t size) { return size > op_size_threshold; }

 protected:
  /**
   * @brief The read/write size above which GDS is faster then host read + copy
   *
   * This may not be the optimal threshold for all systems. `is_gds_io_preferred` can use a
   * different logic based on the system config.
   */
  static constexpr size_t op_size_threshold = 128 << 10;
  file_wrapper const file;
  cf_file_wrapper const cf_file;
};

class gds_input : public gds_io_base {
 public:
  gds_input(std::string const &filepath);

  std::unique_ptr<datasource::buffer> read(size_t offset, size_t size);

  size_t read(size_t offset, size_t size, uint8_t *dst);
};

class gds_output : public gds_io_base {
 public:
  gds_output(std::string const &filepath);

  void write(void const *data, size_t offset, size_t size);
};

};  // namespace io
};  // namespace cudf