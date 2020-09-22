/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cufile.h>

#include <rmm/device_buffer.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace io {

struct file_wrapper {
  int const fd = -1;
  explicit file_wrapper(const char *filepath, int oflags = O_RDONLY) : fd(open(filepath, oflags)) {}
  ~file_wrapper() { close(fd); }
};

struct cufile_driver {
  cufile_driver()
  {
    if (cuFileDriverOpen().err != CU_FILE_SUCCESS) throw "Cannot init cufile driver";
  }
  ~cufile_driver() { cuFileDriverClose(); }
};

class gdsfile {
 public:
  gdsfile(const char *filepath) : handle(filepath, O_RDONLY | O_DIRECT)
  {
    static cufile_driver driver;
    CUDF_EXPECTS(handle.fd != -1, "Cannot open file");

    CUfileDescr_t cf_desc{};
    cf_desc.handle.fd = handle.fd;
    cf_desc.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUDF_EXPECTS(cuFileHandleRegister(&cf_handle, &cf_desc).err == CU_FILE_SUCCESS,
                 "Cannot map cufile");

    struct stat st;
    CUDF_EXPECTS(fstat(handle.fd, &st) != -1, "Cannot query file size");
  }

  std::unique_ptr<datasource::buffer> read(size_t offset, size_t size)
  {
    rmm::device_buffer out_data(size);
    cuFileRead(cf_handle, out_data.data(), size, offset, 0);

    return datasource::buffer::create(std::move(out_data));
  }

  size_t read(size_t offset, size_t size, uint8_t *dst)
  {
    cuFileRead(cf_handle, dst, size, offset, 0);
    // have to read the requested size for now
    return size;
  }

  ~gdsfile() { cuFileHandleDeregister(cf_handle); }

 private:
  file_wrapper handle;
  CUfileHandle_t cf_handle = nullptr;
};

/**
 * @brief Implementation class for reading from a file or memory source using
 * memory mapped access.
 *
 * Unlike Arrow's memory mapped IO class, this implementation allows memory
 * mapping a subset of the file where the starting offset may not be zero.
 */
class memory_mapped_source : public datasource {
  class memory_mapped_buffer : public buffer {
    size_t _size   = 0;
    uint8_t *_data = nullptr;

   public:
    memory_mapped_buffer(uint8_t *data, size_t size) : _data(data), _size(size) {}
    size_t size() const override { return _size; }
    const uint8_t *data() const override { return _data; }
  };

 public:
  explicit memory_mapped_source(const char *filepath, size_t offset, size_t size)
    : _gds_file(filepath)
  {
    auto const file = file_wrapper(filepath);
    CUDF_EXPECTS(file.fd != -1, "Cannot open file");

    struct stat st;
    CUDF_EXPECTS(fstat(file.fd, &st) != -1, "Cannot query file size");
    file_size_ = static_cast<size_t>(st.st_size);

    if (file_size_ != 0) { map(file.fd, offset, size); }
  }

  virtual ~memory_mapped_source()
  {
    if (map_addr_ != nullptr) { munmap(map_addr_, map_size_); }
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    CUDF_EXPECTS(offset >= map_offset_, "Requested offset is outside mapping");

    // Clamp length to available data in the mapped region
    auto const read_size = std::min(size, map_size_ - (offset - map_offset_));

    return std::make_unique<memory_mapped_buffer>(
      static_cast<uint8_t *>(map_addr_) + (offset - map_offset_), read_size);
  }

  size_t host_read(size_t offset, size_t size, uint8_t *dst) override
  {
    CUDF_EXPECTS(offset >= map_offset_, "Requested offset is outside mapping");

    // Clamp length to available data in the mapped region
    auto const read_size = std::min(size, map_size_ - (offset - map_offset_));

    auto const src = static_cast<uint8_t *>(map_addr_) + (offset - map_offset_);
    std::memcpy(dst, src, read_size);
    return read_size;
  }

  bool supports_device_read() const override { return true; }

  std::unique_ptr<datasource::buffer> device_read(size_t offset, size_t size) override
  {
    auto const read_size = std::min(size, map_size_ - (offset - map_offset_));
    return _gds_file.read(offset, size);
  }

  size_t device_read(size_t offset, size_t size, uint8_t *dst) override
  {
    auto const read_size = std::min(size, map_size_ - (offset - map_offset_));
    return _gds_file.read(offset, size, dst);
  }

  size_t size() const override { return file_size_; }

 private:
  void map(int fd, size_t offset, size_t size)
  {
    CUDF_EXPECTS(offset < file_size_, "Offset is past end of file");

    // Offset for `mmap()` must be page aligned
    auto const map_offset = offset & ~(sysconf(_SC_PAGESIZE) - 1);

    // Clamp length to available data in the file
    if (size == 0) {
      size = file_size_ - offset;
    } else {
      if ((offset + size) > file_size_) { size = file_size_ - offset; }
    }

    // Size for `mmap()` needs to include the page padding
    const auto map_size = size + (offset - map_offset);

    // Check if accessing a region within already mapped area
    map_addr_ = mmap(NULL, map_size, PROT_READ, MAP_PRIVATE, fd, map_offset);
    CUDF_EXPECTS(map_addr_ != MAP_FAILED, "Cannot create memory mapping");
    map_offset_ = map_offset;
    map_size_   = map_size;
  }

 private:
  size_t file_size_  = 0;
  void *map_addr_    = nullptr;
  size_t map_size_   = 0;
  size_t map_offset_ = 0;
  gdsfile _gds_file;
};

/**
 * @brief Wrapper class for user implemented data sources
 *
 * Holds the user-implemented object with a non-owning pointer; The user object is not deleted
 * when the wrapper object is destroyed.
 * All API calls are forwarded to the user datasource object.
 */
class user_datasource_wrapper : public datasource {
 public:
  explicit user_datasource_wrapper(datasource *const source) : source(source) {}

  size_t host_read(size_t offset, size_t size, uint8_t *dst) override
  {
    return source->host_read(offset, size, dst);
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    return source->host_read(offset, size);
  }

  bool supports_device_read() const override { return source->supports_device_read(); }

  size_t device_read(size_t offset, size_t size, uint8_t *dst) override
  {
    return source->device_read(offset, size, dst);
  }

  std::unique_ptr<buffer> device_read(size_t offset, size_t size) override
  {
    return source->device_read(offset, size);
  }

  size_t size() const override { return source->size(); }

 private:
  datasource *const source;  ///< A non-owning pointer to the user-implemented datasource
};

std::unique_ptr<datasource> datasource::create(const std::string &filepath,
                                               size_t offset,
                                               size_t size)
{
  // Use our own memory mapping implementation for direct file reads
  return std::make_unique<memory_mapped_source>(filepath.c_str(), offset, size);
}

std::unique_ptr<datasource> datasource::create(host_buffer const &buffer)
{
  // Use Arrow IO buffer class for zero-copy reads of host memory
  return std::make_unique<arrow_io_source>(std::make_shared<arrow::io::BufferReader>(
    reinterpret_cast<const uint8_t *>(buffer.data), buffer.size));
}

std::unique_ptr<datasource> datasource::create(datasource *source)
{
  // instantiate a wrapper that forwards the calls to the user implementation
  return std::make_unique<user_datasource_wrapper>(source);
}

}  // namespace io
}  // namespace cudf
