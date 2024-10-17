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
#include <vector>

namespace cudf {
namespace io {
namespace {

/**
 * @brief Helper function to safely check the ssize_t return value from read()
 * against a size_t read_size.
 *
 * @param bytes_read Supplies the return value from a read().  Negative values
 * are assumed to indicate an error.
 *
 * @param read_size Supplies the expected number of bytes to have been read.
 *
 * @return True iff bytes_read is non-negative and equal to read_size, false
 * otherwise.
 */
static inline bool check_read(ssize_t bytes_read, size_t read_size) {
  return (bytes_read >= 0) && (static_cast<size_t>(bytes_read) == read_size);
}

/**
 * @brief Helper macro for wrapping a check_read() call in a CUDF_EXPECTS().
 *
 * @param bytes_read Supplies the return value from a read().
 *
 * @param read_size Supplies the expected number of bytes to have been read.
 */
#define CUDF_EXPECTS_READ_SUCCESS(bytes_read, read_size) \
  CUDF_EXPECTS(check_read(bytes_read, read_size), "read failed")

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

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    lseek(_file.desc(), offset, SEEK_SET);

    // Clamp length to available data
    auto const read_size = get_read_size(size, offset);

    std::vector<uint8_t> v(read_size);
    auto const bytes_read = read(_file.desc(), v.data(), read_size);
    CUDF_EXPECTS_READ_SUCCESS(bytes_read, read_size);
    return buffer::create(std::move(v));
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    lseek(_file.desc(), offset, SEEK_SET);

    // Clamp length to available data
    auto const read_size = get_read_size(size, offset);
    auto const bytes_read = read(_file.desc(), dst, read_size);
    CUDF_EXPECTS_READ_SUCCESS(bytes_read, read_size);
    return read_size;
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

    auto const read_size = get_read_size(size, offset);
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
  explicit memory_mapped_source(char const* filepath,
                                size_t offset,
                                size_t max_size_estimate,
                                size_t min_size_estimate)
    : file_source(filepath)
  {
    if (_file.size() != 0) {
      // Memory mapping is not exclusive, so we can include the whole region we expect to read
      map(_file.desc(), offset, max_size_estimate);
      // Buffer registration is exclusive (can't overlap with other registered buffers) so we
      // register the lower estimate; this avoids issues when reading adjacent ranges from the same
      // file from multiple threads
      register_mmap_buffer(offset, min_size_estimate);
    }
  }

  ~memory_mapped_source() override
  {
    if (_map_addr != nullptr) {
      unmap();
      unregister_mmap_buffer();
    }
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    // Clamp length to available data
    auto const read_size = get_read_size(size, offset);

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
    auto const read_size = get_read_size(size, offset);

    // If the requested range is outside of the mapped region, read from the file
    if (offset < _map_offset or offset + read_size > (_map_offset + _map_size)) {
      return file_source::host_read(offset, read_size, dst);
    }

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
  void register_mmap_buffer(size_t offset, size_t size)
  {
    if (_map_addr == nullptr or not pageableMemoryAccessUsesHostPageTables()) { return; }

    // Registered region must be within the mapped region
    _reg_offset = std::max(offset, _map_offset);
    _reg_size   = std::min(size != 0 ? size : _map_size, (_map_offset + _map_size) - _reg_offset);

    _reg_addr         = static_cast<std::byte*>(_map_addr) - _map_offset + _reg_offset;
    auto const result = cudaHostRegister(_reg_addr, _reg_size, cudaHostRegisterReadOnly);
    if (result != cudaSuccess) {
      _reg_addr = nullptr;
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
    if (_reg_addr == nullptr) { return; }

    auto const result = cudaHostUnregister(_reg_addr);
    if (result == cudaSuccess) {
      _reg_addr = nullptr;
    } else {
      CUDF_LOG_WARN("cudaHostUnregister failed with {} ({})",
                    static_cast<int>(result),
                    cudaGetErrorString(result));
    }
  }

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
    auto const count  = get_read_size(size, offset);
    auto const stream = cudf::get_default_stream();
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(dst, _d_buffer.data() + offset, count, cudaMemcpyDefault, stream.value()));
    stream.synchronize();
    return count;
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto const count  = get_read_size(size, offset);
    auto const stream = cudf::get_default_stream();
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
    auto const count = get_read_size(size, offset);
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
                                               size_t max_size_estimate,
                                               size_t min_size_estimate,
                                               datasource_kind kind,
                                               std::optional<const datasource_params&> params)
{
  CUDF_EXPECTS(max_size_estimate == 0 or min_size_estimate <= max_size_estimate,
               "Invalid min/max size estimates for datasource creation");

  switch (kind) {
    case datasource_kind::KVIKIO:
    case datasource_kind::KVIKIO_COMPAT:
    case datasource_kind::KVIKIO_GDS: {
      kvikio_datasource_params new_params;
      if (params) {
        if (auto kvikio_params = std::get_if<kvikio_datasource_params>(&params.value())) {
          // Copy the user-provided parameters into our local variable.
          new_params = *kvikio_params;
        } else {
          throw cudf::logic_error("Invalid parameters for KVIKIO-based datasource.");
        }
      }
      if (kind == datasource_kind::KVIKIO_COMPAT) {
        // Forcibly-set the compatibility mode to true, regardless of what may
        // already be present in the params.  The `kind` parameter has requested
        // `KVIKIO_COMPAT`, and that takes precedence over the `use_compat_mode`
        // parameter in the `kvikio_datasource_params`.
        new_params.use_compat_mode = true;
      } else if (kind == datasource_kind::KVIKIO_GDS) {
        // GDS is unique in that we are expected to throw a cudf::runtime_error
        // if GDS is not available.  The first chance we have to do this is
        // here, by way of fencing against CUFILE_FOUND.
#ifndef CUFILE_FOUND
        throw cudf::runtime_error("GDS is not available because cuFile is not enabled.");
#endif
        // The next check is done against the `is_gds_enabled()` function in
        // `cufile_integration`.  If GDS is not enabled, we throw a runtime
        // error here as well.
        if (!cufile_integration::is_gds_enabled()) {
          throw cudf::runtime_error("cuFile reports GDS is not available.");
        }
        // Forcibly-set the compatibility mode to false, regardless of what may
        // already be present in the params.  The `kind` parameter has requested
        // `KVIKIO_GDS`, and that takes precedence over the `use_compat_mode`
        // parameter in the `kvikio_datasource_params`.
        new_params.use_compat_mode = false;
      } else {
        CUDF_EXPECTS(kind == datasource_kind::KVIKIO,
                     "Invariant check failed: kind != datasource_kind::KVIKIO");
        // We don't need to do any special handling for `KVIKIO` here.
      }
      return std::make_unique<kvikio_source>(filepath.c_str(), new_params);
    }
    case datasource_kind::HOST:
      return std::make_unique<host_source>(filepath.c_str());
    case datasource_kind::ODIRECT: {
      odirect_datasource_params new_params;
      if (params) {
        if (auto odirect_params = std::get_if<odirect_datasource_params>(&params.value())) {
          // Copy the user-provided parameters into our local variable.
          new_params = *odirect_params;
        } else {
          throw cudf::logic_error("Invalid parameters for O_DIRECT-based datasource.");
        }
      }
      return std::make_unique<odirect_source>(filepath.c_str(), new_params);
    }
    case datasource_kind::HOST_MMAP:
      return std::make_unique<memory_mapped_source>(
        filepath.c_str(), offset, max_size_estimate, min_size_estimate);
    default:
      CUDF_FAIL("Unsupported datasource kind");
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
