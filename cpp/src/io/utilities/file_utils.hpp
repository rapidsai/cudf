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

class gdsinfile {
 public:
  gdsinfile(std::string const &filepath);

  std::unique_ptr<datasource::buffer> read(size_t offset, size_t size);

  size_t read(size_t offset, size_t size, uint8_t *dst);

  ~gdsinfile();

 private:
  file_wrapper const file;
  CUfileHandle_t cufile_handle = nullptr;
};

class gdsoutfile {
 public:
  gdsoutfile(std::string const &filepath);

  void write(void const *data, size_t offset, size_t size);

  ~gdsoutfile();

 private:
  file_wrapper const file;
  CUfileHandle_t cufile_handle = nullptr;
};

};  // namespace io
};  // namespace cudf