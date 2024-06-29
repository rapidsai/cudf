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

#include <cudf/detail/utilities/logger.hpp>
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

#include <unordered_map>

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
      _kvikio_file = kvikio::FileHandle(filepath);
      CUDF_LOG_INFO("Reading a file using kvikIO, with compatibility mode {}.",
                    _kvikio_file.is_compat_mode_on() ? "on" : "off");
    } else {
      _cufile_in = detail::make_cufile_input(filepath);
    }
  }

  ~file_source() override = default;

  [[nodiscard]] bool supports_device_read() const override
  {
    return !_kvikio_file.closed() || _cufile_in != nullptr;
  }

  [[nodiscard]] bool is_device_read_preferred(size_t size) const override
  {
    if (size < _gds_read_preferred_threshold) { return false; }
    return supports_device_read();
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
 * @brief Memoized pageableMemoryAccessUsesHostPageTables device property.
 */
[[nodiscard]] bool pageableMemoryAccessUsesHostPageTables()
{
  static std::unordered_map<int, bool> result_cache{};

  int deviceId{};
  CUDF_CUDA_TRY(cudaGetDevice(&deviceId));

  if (result_cache.find(deviceId) == result_cache.end()) {
    cudaDeviceProp props{};
    CUDF_CUDA_TRY(cudaGetDeviceProperties(&props, deviceId));
    result_cache[deviceId] = (props.pageableMemoryAccessUsesHostPageTables == 1);
    CUDF_LOG_INFO(
      "Device {} pageableMemoryAccessUsesHostPageTables: {}", deviceId, result_cache[deviceId]);
  }

  return result_cache[deviceId];
}

/**
 * @brief Implementation class for reading from a file using memory mapped access.
 *
 * Unlike Arrow's memory mapped IO class, this implementation allows memory mapping a subset of the
 * file where the starting offset may not be zero.
 */
class memory_mapped_source : public file_source {
 public:
  explicit memory_mapped_source(char const* filepath, size_t offset, size_t size)
    : file_source(filepath)
  {
    if (_file.size() != 0) {
      map(_file.desc(), offset, size);
      register_mmap_buffer();
    }
  }

  ~memory_mapped_source() override
  {
    if (_map_addr != nullptr) {
      munmap(_map_addr, _map_size);
      unregister_mmap_buffer();
    }
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    CUDF_EXPECTS(offset >= _map_offset, "Requested offset is outside mapping");

    // Clamp length to available data in the mapped region
    auto const read_size = std::min(size, _map_size - (offset - _map_offset));

    return std::make_unique<non_owning_buffer>(
      static_cast<uint8_t*>(_map_addr) + (offset - _map_offset), read_size);
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    CUDF_EXPECTS(offset >= _map_offset, "Requested offset is outside mapping");

    // Clamp length to available data in the mapped region
    auto const read_size = std::min(size, _map_size - (offset - _map_offset));

    auto const src = static_cast<uint8_t*>(_map_addr) + (offset - _map_offset);
    std::memcpy(dst, src, read_size);
    return read_size;
  }

 private:
  /**
   * @brief Page-locks (registers) the memory range of the mapped file.
   *
   * Fixes nvbugs/4215160
   */
  void register_mmap_buffer()
  {
    if (_map_addr == nullptr or _map_size == 0 or not pageableMemoryAccessUsesHostPageTables()) {
      return;
    }

    auto const result = cudaHostRegister(_map_addr, _map_size, cudaHostRegisterDefault);
    if (result == cudaSuccess) {
      _is_map_registered = true;
    } else {
      CUDF_LOG_WARN("cudaHostRegister failed with {} ({})",
                    static_cast<int>(result),
                    cudaGetErrorString(result));
    }
  }

  /**
   * @brief Unregisters the memory range of the mapped file.
   */
  void unregister_mmap_buffer()
  {
    if (not _is_map_registered) { return; }

    auto const result = cudaHostUnregister(_map_addr);
    if (result != cudaSuccess) {
      CUDF_LOG_WARN("cudaHostUnregister failed with {} ({})",
                    static_cast<int>(result),
                    cudaGetErrorString(result));
    }
  }

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
  size_t _map_size        = 0;
  size_t _map_offset      = 0;
  void* _map_addr         = nullptr;
  bool _is_map_registered = false;
};

/**
 * @brief Implementation class for reading from a file using `read` calls
 *
 * Potentially faster than `memory_mapped_source` when only a small portion of the file is read
 * through the host.
 */
class direct_read_source : public file_source {
 public:
  explicit direct_read_source(char const* filepath) : file_source(filepath) {}

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
    auto const stream = cudf::get_default_stream();
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(dst, _d_buffer.data() + offset, count, cudaMemcpyDefault, stream.value()));
    stream.synchronize();
    return count;
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto const count  = std::min(size, this->size() - offset);
    auto const stream = cudf::get_default_stream();
    auto h_data       = cudf::detail::make_std_vector_async(
      cudf::device_span<std::byte const>{_d_buffer.data() + offset, count}, stream);
    stream.synchronize();
    return std::make_unique<owning_buffer<std::vector<std::byte>>>(std::move(h_data));
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

}  // namespace

std::unique_ptr<datasource> datasource::create(std::string const& filepath,
                                               size_t offset,
                                               size_t size)
{
#ifdef CUFILE_FOUND
  if (cufile_integration::is_always_enabled()) {
    // avoid mmap as GDS is expected to be used for most reads
    return std::make_unique<direct_read_source>(filepath.c_str());
  }
#endif
  // Use our own memory mapping implementation for direct file reads
  return std::make_unique<memory_mapped_source>(filepath.c_str(), offset, size);
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
