/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/io/datasource.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cudf/utilities/error.hpp>
#include <io/utilities/file_utils.hpp>

namespace cudf {
namespace io {

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
    auto const file = file_wrapper(filepath, O_RDONLY);
    file_size_      = file.size();
    if (file_size_ != 0) { map(file.get_desc(), offset, size); }
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
    map_offset_ = offset & ~(sysconf(_SC_PAGESIZE) - 1);

    // Clamp length to available data in the file
    if (size == 0) {
      size = file_size_ - offset;
    } else {
      if ((offset + size) > file_size_) { size = file_size_ - offset; }
    }

    // Size for `mmap()` needs to include the page padding
    map_size_ = size + (offset - map_offset_);

    // Check if accessing a region within already mapped area
    map_addr_ = mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd, map_offset_);
    CUDF_EXPECTS(map_addr_ != MAP_FAILED, "Cannot create memory mapping");
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
