/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "file_io_utilities.hpp"
#include "getenv_or.hpp"

#include <cudf/detail/utilities/logger.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/config_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <kvikio/file_handle.hpp>

#include <rmm/device_buffer.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <regex>
#include <vector>

#ifdef CUDF_KVIKIO_REMOTE_IO
#include <kvikio/remote_handle.hpp>
#endif

namespace cudf {
namespace io {
namespace {

/**
 * @brief Base class for file input. Only implements direct device reads.
 */
class file_source : public datasource {
 public:
  explicit file_source(char const* filepath) : _file(filepath, O_RDONLY)
  {
    detail::force_init_cuda_context();
    if (cufile_integration::is_kvikio_enabled()) {
      cufile_integration::set_up_kvikio();
      _kvikio_file = kvikio::FileHandle(filepath);
      CUDF_LOG_INFO("Reading a file using kvikIO, with compatibility mode {}.",
                    _kvikio_file.is_compat_mode_preferred() ? "on" : "off");
    } else {
      _cufile_in = detail::make_cufile_input(filepath);
    }
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    lseek(_file.desc(), offset, SEEK_SET);

    // Clamp length to available data
    ssize_t const read_size = std::min(size, _file.size() - offset);

    std::vector<uint8_t> v(read_size);
    CUDF_EXPECTS(read(_file.desc(), v.data(), read_size) == read_size, "read failed");
    return buffer::create(std::move(v));
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    lseek(_file.desc(), offset, SEEK_SET);

    // Clamp length to available data
    auto const read_size = std::min(size, _file.size() - offset);

    CUDF_EXPECTS(read(_file.desc(), dst, read_size) == static_cast<ssize_t>(read_size),
                 "read failed");
    return read_size;
  }

  ~file_source() override = default;

  [[nodiscard]] bool supports_device_read() const override
  {
    return !_kvikio_file.closed() || _cufile_in != nullptr;
  }

  [[nodiscard]] bool is_device_read_preferred(size_t size) const override
  {
    if (!supports_device_read()) { return false; }

    // Always prefer device reads if kvikio is enabled
    if (!_kvikio_file.closed()) { return true; }

    return size >= _gds_read_preferred_threshold;
  }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override
  {
    CUDF_EXPECTS(supports_device_read(), "Device reads are not supported for this file.");

    auto const read_size = std::min(size, _file.size() - offset);
    if (!_kvikio_file.closed()) { return _kvikio_file.pread(dst, read_size, offset); }
    return _cufile_in->read_async(offset, read_size, dst, stream);
  }

  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t* dst,
                     rmm::cuda_stream_view stream) override
  {
    return device_read_async(offset, size, dst, stream).get();
  }

  std::unique_ptr<datasource::buffer> device_read(size_t offset,
                                                  size_t size,
                                                  rmm::cuda_stream_view stream) override
  {
    rmm::device_buffer out_data(size, stream);
    size_t read = device_read(offset, size, reinterpret_cast<uint8_t*>(out_data.data()), stream);
    out_data.resize(read, stream);
    return datasource::buffer::create(std::move(out_data));
  }

  [[nodiscard]] size_t size() const override { return _file.size(); }

 protected:
  detail::file_wrapper _file;

 private:
  std::unique_ptr<detail::cufile_input_impl> _cufile_in;
  kvikio::FileHandle _kvikio_file;
  // The read size above which GDS is faster then posix-read + h2d-copy
  static constexpr size_t _gds_read_preferred_threshold = 128 << 10;  // 128KB
};

/**
 * @brief Implementation class for reading from a file using memory mapped access.
 *
 * Unlike Arrow's memory mapped IO class, this implementation allows memory mapping a subset of the
 * file where the starting offset may not be zero.
 */
class memory_mapped_source : public file_source {
 public:
  explicit memory_mapped_source(char const* filepath, size_t offset, size_t max_size_estimate)
    : file_source(filepath)
  {
    if (_file.size() != 0) {
      // Memory mapping is not exclusive, so we can include the whole region we expect to read
      map(_file.desc(), offset, max_size_estimate);
    }
  }

  ~memory_mapped_source() override
  {
    if (_map_addr != nullptr) { unmap(); }
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    // Clamp length to available data
    auto const read_size = std::min(size, +_file.size() - offset);

    // If the requested range is outside of the mapped region, read from the file
    if (offset < _map_offset or offset + read_size > (_map_offset + _map_size)) {
      return file_source::host_read(offset, read_size);
    }

    // If the requested range is only partially within the registered region, copy to a new
    // host buffer to make the data safe to copy to the device
    if (_reg_addr != nullptr and
        (offset < _reg_offset or offset + read_size > (_reg_offset + _reg_size))) {
      auto const src = static_cast<uint8_t*>(_map_addr) + (offset - _map_offset);

      return std::make_unique<owning_buffer<std::vector<uint8_t>>>(
        std::vector<uint8_t>(src, src + read_size));
    }

    return std::make_unique<non_owning_buffer>(
      static_cast<uint8_t*>(_map_addr) + offset - _map_offset, read_size);
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    // Clamp length to available data
    auto const read_size = std::min(size, +_file.size() - offset);

    // If the requested range is outside of the mapped region, read from the file
    if (offset < _map_offset or offset + read_size > (_map_offset + _map_size)) {
      return file_source::host_read(offset, read_size, dst);
    }

    auto const src = static_cast<uint8_t*>(_map_addr) + (offset - _map_offset);
    std::memcpy(dst, src, read_size);
    return read_size;
  }

 private:
  void map(int fd, size_t offset, size_t size)
  {
    CUDF_EXPECTS(offset < _file.size(), "Offset is past end of file", std::overflow_error);

    // Offset for `mmap()` must be page aligned
    _map_offset = offset & ~(sysconf(_SC_PAGESIZE) - 1);

    if (size == 0 || (offset + size) > _file.size()) { size = _file.size() - offset; }

    // Size for `mmap()` needs to include the page padding
    _map_size = size + (offset - _map_offset);
    if (_map_size == 0) { return; }

    // Check if accessing a region within already mapped area
    _map_addr = mmap(nullptr, _map_size, PROT_READ, MAP_PRIVATE, fd, _map_offset);
    CUDF_EXPECTS(_map_addr != MAP_FAILED, "Cannot create memory mapping");
  }

  void unmap()
  {
    if (_map_addr != nullptr) {
      auto const result = munmap(_map_addr, _map_size);
      if (result != 0) { CUDF_LOG_WARN("munmap failed with {}", result); }
      _map_addr = nullptr;
    }
  }

 private:
  size_t _map_offset = 0;
  size_t _map_size   = 0;
  void* _map_addr    = nullptr;

  size_t _reg_offset = 0;
  size_t _reg_size   = 0;
  void* _reg_addr    = nullptr;
};

/**
 * @brief Implementation class for reading from a device buffer source
 */
class device_buffer_source final : public datasource {
 public:
  explicit device_buffer_source(cudf::device_span<std::byte const> d_buffer) : _d_buffer{d_buffer}
  {
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const count  = std::min(size, this->size() - offset);
    auto const stream = cudf::detail::global_cuda_stream_pool().get_stream();
    cudf::detail::cuda_memcpy(host_span<uint8_t>{dst, count},
                              device_span<uint8_t const>{
                                reinterpret_cast<uint8_t const*>(_d_buffer.data() + offset), count},
                              stream);
    return count;
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto const count  = std::min(size, this->size() - offset);
    auto const stream = cudf::detail::global_cuda_stream_pool().get_stream();
    auto h_data       = cudf::detail::make_host_vector_async(
      cudf::device_span<std::byte const>{_d_buffer.data() + offset, count}, stream);
    stream.synchronize();
    return std::make_unique<owning_buffer<cudf::detail::host_vector<std::byte>>>(std::move(h_data));
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override
  {
    auto const count = std::min(size, this->size() - offset);
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(dst, _d_buffer.data() + offset, count, cudaMemcpyDefault, stream.value()));
    return std::async(std::launch::deferred, [count] { return count; });
  }

  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t* dst,
                     rmm::cuda_stream_view stream) override
  {
    return device_read_async(offset, size, dst, stream).get();
  }

  std::unique_ptr<buffer> device_read(size_t offset,
                                      size_t size,
                                      rmm::cuda_stream_view stream) override
  {
    return std::make_unique<non_owning_buffer>(
      reinterpret_cast<uint8_t const*>(_d_buffer.data() + offset), size);
  }

  [[nodiscard]] size_t size() const override { return _d_buffer.size(); }

 private:
  cudf::device_span<std::byte const> _d_buffer;  ///< A non-owning view of the existing device data
};

// zero-copy host buffer source
class host_buffer_source final : public datasource {
 public:
  explicit host_buffer_source(cudf::host_span<std::byte const> h_buffer) : _h_buffer{h_buffer} {}

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const count = std::min(size, this->size() - offset);
    std::memcpy(dst, _h_buffer.data() + offset, count);
    return count;
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto const count = std::min(size, this->size() - offset);
    return std::make_unique<non_owning_buffer>(
      reinterpret_cast<uint8_t const*>(_h_buffer.data() + offset), count);
  }

  [[nodiscard]] bool supports_device_read() const override { return false; }

  [[nodiscard]] size_t size() const override { return _h_buffer.size(); }

 private:
  cudf::host_span<std::byte const> _h_buffer;  ///< A non-owning view of the existing host data
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
  explicit user_datasource_wrapper(datasource* const source) : source(source) {}

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    return source->host_read(offset, size, dst);
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    return source->host_read(offset, size);
  }

  [[nodiscard]] bool supports_device_read() const override
  {
    return source->supports_device_read();
  }

  [[nodiscard]] bool is_device_read_preferred(size_t size) const override
  {
    return source->is_device_read_preferred(size);
  }

  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t* dst,
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

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override
  {
    return source->device_read_async(offset, size, dst, stream);
  }

  [[nodiscard]] size_t size() const override { return source->size(); }

  [[nodiscard]] bool is_empty() const override { return source->is_empty(); }

 private:
  datasource* const source;  ///< A non-owning pointer to the user-implemented datasource
};

#ifdef CUDF_KVIKIO_REMOTE_IO
/**
 * @brief Remote file source backed by KvikIO, which handles S3 filepaths seamlessly.
 */
class remote_file_source : public datasource {
  static std::unique_ptr<kvikio::S3Endpoint> create_s3_endpoint(char const* filepath)
  {
    auto [bucket_name, bucket_object] = kvikio::S3Endpoint::parse_s3_url(filepath);
    return std::make_unique<kvikio::S3Endpoint>(bucket_name, bucket_object);
  }

 public:
  explicit remote_file_source(char const* filepath) : _kvikio_file{create_s3_endpoint(filepath)} {}

  ~remote_file_source() override = default;

  [[nodiscard]] bool supports_device_read() const override { return true; }

  [[nodiscard]] bool is_device_read_preferred(size_t size) const override { return true; }

  [[nodiscard]] size_t size() const override { return _kvikio_file.nbytes(); }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override
  {
    CUDF_EXPECTS(supports_device_read(), "Device reads are not supported for this file.");

    auto const read_size = std::min(size, this->size() - offset);
    return _kvikio_file.pread(dst, read_size, offset);
  }

  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t* dst,
                     rmm::cuda_stream_view stream) override
  {
    return device_read_async(offset, size, dst, stream).get();
  }

  std::unique_ptr<datasource::buffer> device_read(size_t offset,
                                                  size_t size,
                                                  rmm::cuda_stream_view stream) override
  {
    rmm::device_buffer out_data(size, stream);
    size_t read = device_read(offset, size, reinterpret_cast<uint8_t*>(out_data.data()), stream);
    out_data.resize(read, stream);
    return datasource::buffer::create(std::move(out_data));
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const read_size = std::min(size, this->size() - offset);
    return _kvikio_file.pread(dst, read_size, offset).get();
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto const count = std::min(size, this->size() - offset);
    std::vector<uint8_t> h_data(count);
    this->host_read(offset, count, h_data.data());
    return datasource::buffer::create(std::move(h_data));
  }

  /**
   * @brief Is `url` referring to a remote file supported by KvikIO?
   *
   * For now, only S3 urls (urls starting with "s3://") are supported.
   */
  static bool is_supported_remote_url(std::string const& url)
  {
    // Regular expression to match "s3://"
    static std::regex pattern{R"(^s3://)", std::regex_constants::icase};
    return std::regex_search(url, pattern);
  }

 private:
  kvikio::RemoteHandle _kvikio_file;
};
#else
/**
 * @brief When KvikIO remote IO is disabled, `is_supported_remote_url()` return false always.
 */
class remote_file_source : public file_source {
 public:
  explicit remote_file_source(char const* filepath) : file_source(filepath) {}
  static constexpr bool is_supported_remote_url(std::string const&) { return false; }
};
#endif
}  // namespace

std::unique_ptr<datasource> datasource::create(std::string const& filepath,
                                               size_t offset,
                                               size_t max_size_estimate)
{
  auto const use_memory_mapping = [] {
    auto const policy = getenv_or("LIBCUDF_MMAP_ENABLED", std::string{"ON"});

    if (policy == "ON") { return true; }
    if (policy == "OFF") { return false; }

    CUDF_FAIL("Invalid LIBCUDF_MMAP_ENABLED value: " + policy);
  }();
  if (remote_file_source::is_supported_remote_url(filepath)) {
    return std::make_unique<remote_file_source>(filepath.c_str());
  } else if (use_memory_mapping) {
    return std::make_unique<memory_mapped_source>(filepath.c_str(), offset, max_size_estimate);
  } else {
    // `file_source` reads the file directly, without memory mapping
    return std::make_unique<file_source>(filepath.c_str());
  }
}

std::unique_ptr<datasource> datasource::create(host_buffer const& buffer)
{
  return create(
    cudf::host_span<std::byte const>{reinterpret_cast<std::byte const*>(buffer.data), buffer.size});
}

std::unique_ptr<datasource> datasource::create(cudf::host_span<std::byte const> buffer)
{
  return std::make_unique<host_buffer_source>(buffer);
}

std::unique_ptr<datasource> datasource::create(cudf::device_span<std::byte const> buffer)
{
  return std::make_unique<device_buffer_source>(buffer);
}

std::unique_ptr<datasource> datasource::create(datasource* source)
{
  // instantiate a wrapper that forwards the calls to the user implementation
  return std::make_unique<user_datasource_wrapper>(source);
}

}  // namespace io
}  // namespace cudf
