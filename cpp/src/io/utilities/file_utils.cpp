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

file_wrapper::file_wrapper(const char *filepath, int oflags) : fd(open(filepath, oflags))
{
  CUDF_EXPECTS(fd != -1, "Cannot open file");
}

file_wrapper::~file_wrapper() { close(fd); }

size_t file_wrapper::size() const
{
  struct stat st;
  CUDF_EXPECTS(fstat(fd, &st) != -1, "Cannot query file size");
  return static_cast<size_t>(st.st_size);
}

gdsfile::gdsfile(const char *filepath) : file(filepath, O_RDONLY | O_DIRECT)
{
  static cufile_driver driver;
  CUDF_EXPECTS(file.get_desc() != -1, "Cannot open file");

  CUfileDescr_t cufile_desc{};
  cufile_desc.handle.fd = file.get_desc();
  cufile_desc.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  CUDF_EXPECTS(cuFileHandleRegister(&cufile_handle, &cufile_desc).err == CU_FILE_SUCCESS,
               "Cannot map cufile");

  struct stat st;
  CUDF_EXPECTS(fstat(file.get_desc(), &st) != -1, "Cannot query file size");
}

std::unique_ptr<datasource::buffer> gdsfile::read(size_t offset, size_t size)
{
  rmm::device_buffer out_data(size);
  cuFileRead(cufile_handle, out_data.data(), size, offset, 0);

  return datasource::buffer::create(std::move(out_data));
}

size_t gdsfile::read(size_t offset, size_t size, uint8_t *dst)
{
  cuFileRead(cufile_handle, dst, size, offset, 0);
  // have to read the requested size for now
  return size;
}

gdsfile::~gdsfile() { cuFileHandleDeregister(cufile_handle); }
};  // namespace io
};  // namespace cudf