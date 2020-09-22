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

#include <cufile.h>

#include <cudf/io/datasource.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace io {

class file_wrapper {
  int const fd = -1;

 public:
  explicit file_wrapper(const char *filepath, int oflags);
  ~file_wrapper();
  size_t size() const;
  auto get_desc() const { return fd; }
};

struct cufile_driver {
  cufile_driver()
  {
    if (cuFileDriverOpen().err != CU_FILE_SUCCESS) CUDF_FAIL("Cannot init cufile driver");
  }
  ~cufile_driver() { cuFileDriverClose(); }
};

class gdsfile {
 public:
  gdsfile(const char *filepath);

  std::unique_ptr<datasource::buffer> read(size_t offset, size_t size);

  size_t read(size_t offset, size_t size, uint8_t *dst);

  ~gdsfile();

 private:
  file_wrapper const file;
  CUfileHandle_t cufile_handle = nullptr;
};

};  // namespace io
};  // namespace cudf