/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <io/utilities/file_io_utilities.hpp>

namespace cudf {
namespace io {
namespace {

/**
 * @brief Base class for file input. Only implements direct device reads.
 */
class file_source : public datasource {
 public:
  explicit file_source(const char *filepath)
    : _file(filepath, O_RDONLY), _cufile_in(detail::make_cufile_input(filepath))
  {
  }

  virtual ~file_source() = default;

  bool supports_device_read() const override { return _cufile_in != nullptr; }

  bool is_device_read_preferred(size_t size) const
  {
    return _cufile_in != nullptr && _cufile_in->is_cufile_io_preferred(size);
  }

  std::unique_ptr<datasource::buffer> device_read(size_t offset,
                                                  size_t size,
                                                  rmm::cuda_stream_view stream) override
  {
    CUDF_EXPECTS(supports_device_read(), "Device reads are not supported for this file.");

    auto const read_size = std::min(size, _file.size() - offset);
    return _cufile_in->read(offset, read_size, stream);
  }

  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t *dst,
                     rmm::cuda_stream_view stream) override
  {
    CUDF_EXPECTS(supports_device_read(), "Device reads are not supported for this file.");

    auto const read_size = std::min(size, _file.size() - offset);
    return _cufile_in->read(offset, read_size, dst, stream);
  }

  size_t size() const override { return _file.size(); }

 protected:
  detail::file_wrapper _file;

 private:
  std::unique_ptr<detail::cufile_input_impl> _cufile_in;
};

/**
 * @brief Implementation class for reading from a file using memory mapped access.
 *
 * Unlike Arrow's memory mapped IO class, this implementation allows memory mapping a subset of the
 * file where the starting offset may not be zero.
 */
class memory_mapped_source : public file_source {
 public:
  explicit memory_mapped_source(const char *filepath, size_t offset, size_t size)
    : file_source(filepath)
  {
    if (_file.size() != 0) map(_file.desc(), offset, size);
  }

  virtual ~memory_mapped_source()
  {
    if (_map_addr != nullptr) { munmap(_map_addr, _map_size); }
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    CUDF_EXPECTS(offset >= _map_offset, "Requested offset is outside mapping");

    // Clamp length to available data in the mapped region
    auto const read_size = std::min(size, _map_size - (offset - _map_offset));

    return std::make_unique<non_owning_buffer>(
      static_cast<uint8_t *>(_map_addr) + (offset - _map_offset), read_size);
  }

  size_t host_read(size_t offset, size_t size, uint8_t *dst) override
  {
    CUDF_EXPECTS(offset >= _map_offset, "Requested offset is outside mapping");

    // Clamp length to available data in the mapped region
    auto const read_size = std::min(size, _map_size - (offset - _map_offset));

    auto const src = static_cast<uint8_t *>(_map_addr) + (offset - _map_offset);
    std::memcpy(dst, src, read_size);
    return read_size;
  }

 private:
  void map(int fd, size_t offset, size_t size)
  {
    CUDF_EXPECTS(offset < _file.size(), "Offset is past end of file");

    // Offset for `mmap()` must be page aligned
    _map_offset = offset & ~(sysconf(_SC_PAGESIZE) - 1);

    if (size == 0 || (offset + size) > _file.size()) { size = _file.size() - offset; }

    // Size for `mmap()` needs to include the page padding
    _map_size = size + (offset - _map_offset);

    // Check if accessing a region within already mapped area
    _map_addr = mmap(nullptr, _map_size, PROT_READ, MAP_PRIVATE, fd, _map_offset);
    CUDF_EXPECTS(_map_addr != MAP_FAILED, "Cannot create memory mapping");
  }

 private:
  size_t _map_size   = 0;
  size_t _map_offset = 0;
  void *_map_addr    = nullptr;
};

/**
 * @brief Implementation class for reading from a file using `read` calls
 *
 * Potentially faster than `memory_mapped_source` when only a small portion of the file is read
 * through the host.
 */
class direct_read_source : public file_source {
 public:
  explicit direct_read_source(const char *filepath) : file_source(filepath) {}

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    lseek(_file.desc(), offset, SEEK_SET);

    // Clamp length to available data
    ssize_t const read_size = std::min(size, _file.size() - offset);

    std::vector<uint8_t> v(read_size);
    CUDF_EXPECTS(read(_file.desc(), v.data(), read_size) == read_size, "read failed");
    return buffer::create(std::move(v));
  }

  size_t host_read(size_t offset, size_t size, uint8_t *dst) override
  {
    lseek(_file.desc(), offset, SEEK_SET);

    // Clamp length to available data
    auto const read_size = std::min(size, _file.size() - offset);

    CUDF_EXPECTS(read(_file.desc(), dst, read_size) == static_cast<ssize_t>(read_size),
                 "read failed");
    return read_size;
  }
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

  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t *dst,
                     rmm::cuda_stream_view stream) override
  {
    return source->device_read(offset, size, dst, stream);
  }

  std::unique_ptr<buffer> device_read(size_t offset,
                                      size_t size,
                                      rmm::cuda_stream_view stream) override
  {
    return source->device_read(offset, size, stream);
  }

  size_t size() const override { return source->size(); }

 private:
  datasource *const source;  ///< A non-owning pointer to the user-implemented datasource
};

}  // namespace

std::unique_ptr<datasource> datasource::create(const std::string &filepath,
                                               size_t offset,
                                               size_t size)
{
#ifdef CUFILE_FOUND
  if (detail::cufile_config::instance()->is_required()) {
    // avoid mmap as GDS is expected to be used for most reads
    return std::make_unique<direct_read_source>(filepath.c_str());
  }
#endif
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
