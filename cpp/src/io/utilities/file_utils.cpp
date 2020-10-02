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

cf_file_wrapper::cf_file_wrapper(int fd)
{
  init_cufile_driver();

  CUfileDescr_t cufile_desc{};
  cufile_desc.handle.fd = fd;
  cufile_desc.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  CUDF_EXPECTS(cuFileHandleRegister(&handle, &cufile_desc).err == CU_FILE_SUCCESS,
               "Cannot register file handle with cuFile");
}

cf_file_wrapper::~cf_file_wrapper() { cuFileHandleDeregister(handle); }

gds_input::gds_input(std::string const &filepath) : gds_io_base(filepath, O_RDONLY | O_DIRECT) {}

std::unique_ptr<datasource::buffer> gds_input::read(size_t offset, size_t size)
{
  rmm::device_buffer out_data(size);
  CUDF_EXPECTS(cuFileRead(cf_file.handle, out_data.data(), size, offset, 0) != -1,
               "cuFile error reading from a file");

  return datasource::buffer::create(std::move(out_data));
}

size_t gds_input::read(size_t offset, size_t size, uint8_t *dst)
{
  CUDF_EXPECTS(cuFileRead(cf_file.handle, dst, size, offset, 0) != -1,
               "cuFile error reading from a file");
  // have to read the requested size for now
  return size;
}

gds_output::gds_output(std::string const &filepath)
  : gds_io_base(filepath, O_CREAT | O_RDWR | O_DIRECT, 0664)
{
}

void gds_output::write(void const *data, size_t offset, size_t size)
{
  CUDF_EXPECTS(cuFileWrite(cf_file.handle, data, size, offset, 0) != -1,
               "cuFile error writing to a file");
}

};  // namespace io
};  // namespace cudf