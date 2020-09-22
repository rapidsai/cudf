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
#include <io/utilities/file_utils.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <rmm/device_buffer.hpp>

namespace cudf {
namespace io {

file_wrapper::file_wrapper(std::string const &filepath, int flags)
  : fd(open(filepath.c_str(), flags))
{
  CUDF_EXPECTS(fd != -1, "Cannot open file");
}

file_wrapper::file_wrapper(std::string const &filepath, int flags, mode_t mode)
  : fd(open(filepath.c_str(), flags, mode))
{
  CUDF_EXPECTS(fd != -1, "Cannot open file");
}

struct cufile_driver {
  cufile_driver()
  {
    if (cuFileDriverOpen().err != CU_FILE_SUCCESS) CUDF_FAIL("Cannot init cufile driver");
  }
  ~cufile_driver() { cuFileDriverClose(); }
};

void init_cufile_driver() { static cufile_driver driver; }

file_wrapper::~file_wrapper() { close(fd); }

long file_wrapper::size() const
{
  if (_size < 0) {
    struct stat st;
    CUDF_EXPECTS(fstat(fd, &st) != -1, "Cannot query file size");
    _size = static_cast<size_t>(st.st_size);
  }
  return _size;
}

gdsinfile::gdsinfile(std::string const &filepath) : file(filepath, O_RDONLY | O_DIRECT)
{
  init_cufile_driver();

  CUfileDescr_t cufile_desc{};
  cufile_desc.handle.fd = file.desc();
  cufile_desc.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  CUDF_EXPECTS(cuFileHandleRegister(&cufile_handle, &cufile_desc).err == CU_FILE_SUCCESS,
               "Cannot register file handle with cuFile");
}

std::unique_ptr<datasource::buffer> gdsinfile::read(size_t offset, size_t size)
{
  rmm::device_buffer out_data(size);
  CUDF_EXPECTS(cuFileRead(cufile_handle, out_data.data(), size, offset, 0) != -1,
               "cuFile error reading from a file");

  return datasource::buffer::create(std::move(out_data));
}

size_t gdsinfile::read(size_t offset, size_t size, uint8_t *dst)
{
  CUDF_EXPECTS(cuFileRead(cufile_handle, dst, size, offset, 0) != -1,
               "cuFile error reading from a file");
  // have to read the requested size for now
  return size;
}

gdsinfile::~gdsinfile() { cuFileHandleDeregister(cufile_handle); }

gdsoutfile::gdsoutfile(std::string const &filepath)
  : file(filepath, O_CREAT | O_RDWR | O_DIRECT, 0664)
{
  init_cufile_driver();

  CUfileDescr_t cufile_desc{};
  cufile_desc.handle.fd = file.desc();
  cufile_desc.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  CUDF_EXPECTS(cuFileHandleRegister(&cufile_handle, &cufile_desc).err == CU_FILE_SUCCESS,
               "Cannot register file handle with cuFile");
}

void gdsoutfile::write(void const *data, size_t offset, size_t size)
{
  CUDF_EXPECTS(cuFileWrite(cufile_handle, data, size, offset, 0) != -1,
               "cuFile error writing to a file");
}

gdsoutfile::~gdsoutfile() { cuFileHandleDeregister(cufile_handle); }

};  // namespace io
};  // namespace cudf