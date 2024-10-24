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

#include <cudf/detail/utilities/integer_utils.hpp>
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
#include <sys/stat.h>
#include <unistd.h>

#include <unordered_map>
#include <vector>

namespace cudf {
namespace io {
namespace {

/**
 * @brief Helper routine for determining if a given address is aligned to the
 * specified alignment.

 * @param ptr Supplies the address to check.
 * @param alignment Supplies the alignment to check against.
 *
 * @return True iff ptr is aligned to alignment, false otherwise.
 */
static inline bool is_aligned(void const* ptr, std::uintptr_t alignment)
{
  // N.B. Stolen from io/comp/nvcomp_adapter.cpp.
  return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

/**
 * @brief Helper class to encapsulate an aligned host buffer allocated via
 * posix_memalign().
 */
class aligned_buffer : public datasource::buffer {
 public:
  /**
   * @brief Construct an aligned buffer of the specified size and alignment.
   *
   * @param size Supplies the size of the buffer to allocate.
   * @param alignment Supplies the desired address alignment of the underlying
   * buffer.
   */
  aligned_buffer(size_t size, std::uintptr_t alignment) : _size(size), _alignment(alignment)
  {
    if (posix_memalign(reinterpret_cast<void**>(&_data), static_cast<size_t>(alignment), size) !=
        0) {
      CUDF_LOG_ERROR("posix_memalign(size={}, alignment={}) failed: {} ({})",
                     size,
                     alignment,
                     errno,
                     strerror(errno));
      CUDF_FAIL("Failed to allocate aligned buffer");
    }
  }

  ~aligned_buffer() override
  {
    if (_data != nullptr) {
      free(_data);
      _data = nullptr;
    }
  }

  constexpr aligned_buffer() noexcept = default;

  /**
   * @brief Move constructor.
   *
   * @param other Supplies the aligned buffer to move from.
   */
  aligned_buffer(aligned_buffer&& other) noexcept
    : _data(other._data), _size(other._size), _alignment(other._alignment)
  {
    other._data      = nullptr;
    other._size      = 0;
    other._alignment = 0;
  }

  /**
   * @brief Swap the contents of this aligned buffer with another.
   *
   * @param other Supplies the other aligned buffer with which to swap contents.
   */
  void swap(aligned_buffer& other) noexcept
  {
    std::swap(_data, other._data);
    std::swap(_size, other._size);
    std::swap(_alignment, other._alignment);
  }

  aligned_buffer& operator=(aligned_buffer&& other) noexcept
  {
    if (this != &other) {
      // Use a temporary to ensure we don't leave this object in an
      // inconsistent state if the move assignment fails.
      aligned_buffer tmp(std::move(other));
      swap(tmp);
    }
    return *this;
  }

  // Delete copy constructor and assignment operator.
  aligned_buffer(aligned_buffer const&)            = delete;
  aligned_buffer& operator=(aligned_buffer const&) = delete;

  // Base class overrides
  [[nodiscard]] size_t size() const override { return _size; }
  [[nodiscard]] uint8_t const* data() const override { return _data; }

  // Additional methods
  [[nodiscard]] uint8_t* mutable_data() { return _data; }
  [[nodiscard]] std::uintptr_t alignment() const { return _alignment; }

 private:
  uint8_t* _data{nullptr};  ///< Pointer to the aligned buffer
  size_t _size{0};          ///< Size of the aligned buffer
  size_t _alignment{0};     ///< Alignment of the buffer
};

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
static inline bool check_read(ssize_t bytes_read, size_t read_size)
{
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
 * @brief Host-based data source that issues standard POSIX file I/O calls.
 */
class host_source : public datasource {
 public:
  host_source(std::string const& filepath) : datasource(datasource_kind::HOST), _filepath(filepath)
  {
    // Open the file, then obtain its size by way of fstat().
    _fd = open(filepath.c_str(), O_RDONLY);
    if (_fd < 0) {
      CUDF_LOG_ERROR("Cannot open file {}: {}: {}", filepath, errno, strerror(errno));
      CUDF_FAIL("Cannot open file");
    }

    // File descriptor is valid; now obtain the file size.
    struct stat statbuf;
    if (fstat(_fd, &statbuf) < 0) {
      CUDF_LOG_ERROR("Cannot stat file {}: {}: {}", filepath, errno, strerror(errno));
      CUDF_FAIL("Cannot stat file");
    }
    _size = statbuf.st_size;
  }

  ~host_source() override
  {
    if (_fd >= 0) {
      if (::close(_fd) < 0) {
        CUDF_LOG_ERROR("Cannot close file {}: {}: {}", _filepath, errno, strerror(errno));
      }
    }
    _fd = -1;
  }

  [[nodiscard]] std::unique_ptr<buffer> host_read(size_t offset, size_t size)
  {
    // Clamp length to available data
    auto const read_size = get_read_size(size, offset);

    std::vector<uint8_t> v(read_size);
    auto const bytes_read = host_read(_fd, offset, read_size, v.data());
    CUDF_EXPECTS_READ_SUCCESS(bytes_read, read_size);
    return buffer::create(std::move(v));
  }

  [[nodiscard]] size_t host_read(size_t offset, size_t size, uint8_t* dst)
  {
    // Clamp length to available data
    auto const read_size  = get_read_size(size, offset);
    auto const bytes_read = host_read(_fd, offset, read_size, dst);
    CUDF_EXPECTS_READ_SUCCESS(bytes_read, read_size);
    return read_size;
  }

  [[nodiscard]] size_t size() const override { return _size; }

 protected:
  [[nodiscard]] const std::string& filepath() const { return _filepath; }

  /**
   * @brief Reads a range of bytes from a file descriptor into the supplied
   * buffer.
   *
   * @param fd Supplies the file descriptor from which to read.
   * @param offset Supplies the offset within the file to begin reading.
   * @param read_size Supplies the number of bytes to read.  This should be the
   * clamped read size value obtained from an earlier call to get_read_size().
   * @param dst Supplies the buffer into which to read the data.  This buffer
   * should be at least read_size bytes in length.
   *
   * @return The number of bytes read on success.  An exception is thrown on
   * error.
   */
  [[nodiscard]] size_t host_read(int fd, size_t offset, size_t read_size, uint8_t* dst)
  {
    ssize_t bytes_remaining = read_size;
    size_t current_offset   = offset;
    auto buf                = reinterpret_cast<char*>(dst);
    ssize_t bytes_read;
    size_t total_bytes_read = 0;

    while (bytes_remaining > 0) {
      // Retry the pread() if interrupted by a signal.
      do {
        bytes_read = pread(fd, buf, bytes_remaining, current_offset);
      } while (bytes_read < 0 && errno == EINTR);

      if (bytes_read == 0) {
        // We're at EOF; we should never hit this because get_read_size() clamps
        // our size to the underlying datasource size, meaning we'll never issue
        // a read past EOF.
        CUDF_LOG_ERROR(
          "Encountered unexpected EOF reading {} byte{} at offset {} "
          "from {}: {}, {}",
          bytes_remaining,
          (bytes_remaining == 1) ? "" : "s",
          current_offset,
          filepath(),
          errno,
          strerror(errno));
        CUDF_FAIL("Unexpected EOF reading file");
      }

      if (bytes_read < 0) {
        CUDF_LOG_ERROR("Failed to read {} byte{} at offset {} from file {}: {}, {}",
                       bytes_remaining,
                       (bytes_remaining == 1) ? "" : "s",
                       current_offset,
                       filepath(),
                       errno,
                       strerror(errno));
        CUDF_FAIL("Cannot read from file");
      }

      // Update the buffer pointer, counters, offsets, and remaining byte count.
      total_bytes_read += static_cast<size_t>(bytes_read);
      bytes_remaining -= bytes_read;
      current_offset += bytes_read;
      buf += bytes_read;

      // Invariant check: bytes_remaining should always be non-negative.
      CUDF_EXPECTS(bytes_remaining >= 0, "Invariant check failed: bytes_remaining >= 0");
    }

    return total_bytes_read;
  }

 private:
  std::string _filepath;  ///< The path to the file
  int _fd{-1};            ///< File descriptor
  size_t _size{0};        ///< Size of the file, in bytes.
};

/**
 * @brief O_DIRECT-based data source derived from host_source.
 */
class odirect_source : public host_source {
 public:
  odirect_source(std::string const& filepath, odirect_datasource_params const& params)
    : host_source(filepath), _params(params)
  {
    // Verify the caller provided something sane for the sector size.
    if (!params.is_valid_sector_size()) {
      CUDF_LOG_ERROR("Invalid sector size: {}", params.sector_size);
      CUDF_FAIL("Invalid sector size");
    }
    _sector_size = _params.sector_size;

    set_datasource_kind(datasource_kind::ODIRECT);

    // Open the file with O_DIRECT.
    _fd_o_direct = open(filepath.c_str(), O_RDONLY | O_DIRECT);
    if (_fd_o_direct < 0) {
      CUDF_LOG_ERROR("Cannot open file {}: {}: {}", filepath, errno, strerror(errno));
      CUDF_FAIL("Cannot open file");
    }
  }

  ~odirect_source() override
  {
    if (_fd_o_direct >= 0) {
      if (::close(_fd_o_direct) < 0) {
        CUDF_LOG_ERROR("Cannot close file {}: {}: {}", filepath(), errno, strerror(errno));
      }
    }
    _fd_o_direct = -1;
  }

  [[nodiscard]] size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    // Clamp length to available data
    auto const read_size = get_read_size(size, offset);

    bool use_o_direct = false;

    // In order to read from a file descriptor opened with O_DIRECT, the
    // following three elements must all be aligned to the sector size:
    //
    //  1. The offset at which to start reading.
    //  2. The number of bytes to read.
    //  3. The buffer into which the data is read.
    //
    // If all three conditions are met, we can use O_DIRECT to read the data.
    // As a caller will rarely pass us offsets and sizes that are perfectly
    // aligned to the sector size, we typically have to massage the read
    // parameters first prior to issuing the read.
    //
    // The exception to this rule is when the caller has requested a read
    // of the final bytes of the file, such that an aligned-up sector size
    // read would exceed the file size.  In this case, we fall back to a
    // normal pread() call against a non-O_DIRECT file descriptor (by way
    // of simply deferring to the `host_source` base class's `host_read()`).

    // Calculate the sector-aligned sizes for offset and read size.  We round
    // down for offset, which means we need to track the bytes to skip at the
    // beginning of the read buffer.
    size_t aligned_offset = util::round_down_safe(offset, _sector_size);
    size_t bytes_to_skip  = offset - aligned_offset;

    // For the read size, we add an additional sector size to the read size,
    // and then round that value up to the nearest sector size.  This is done
    // to ensure that we always read enough data to cover the requested read
    // size.  As we're adding an extra sector size and rounding up, we need
    // to track the bytes to ignore at the end of the read buffer.
    size_t aligned_read_size = util::round_up_safe(read_size + _sector_size, _sector_size);
    size_t bytes_to_ignore   = aligned_read_size - read_size;

    // We can use O_DIRECT as long as the final aligned read size from the
    // aligned offset does not exceed the file size.
    if ((aligned_offset + aligned_read_size) <= this->size()) { use_o_direct = true; }

    if (!use_o_direct) {
      // We can't use O_DIRECT for this read, so we fall back to a normal
      // pread() call against a non-O_DIRECT file descriptor.  Note that we
      // use the original offset and read size, not the aligned values.
      return host_source::host_read(offset, read_size, dst);
    }

    // If we get here, we're going to use O_DIRECT for the read, which means
    // the buffer we read into needs to be sector-aligned.  If the caller has
    // already provided a sector-aligned buffer, as well as sector-aligned
    // offsets and read size (i.e. bytes_to_skip and bytes_to_ignore are zero),
    // we can use the caller's buffer as-is.
    uint8_t* buf;
    aligned_buffer aligned_buf;
    const bool use_caller_buffer =
      ((bytes_to_skip == 0) && (bytes_to_ignore == 0) && is_aligned(dst, _sector_size));

    if (use_caller_buffer) {
      buf = dst;
    } else {
      // Allocate an aligned buffer to read into.
      aligned_buf = aligned_buffer(aligned_read_size, _sector_size);
      buf         = aligned_buf.mutable_data();
    }

    // We can now issue the read against our O_DIRECT file descriptor using
    // the base class's `host_read()` implementation.
    auto const total_bytes_read =
      host_source::host_read(_fd_o_direct, aligned_offset, aligned_read_size, buf);

    // We can't do the usual `CUDF_EXPECTS_READ_SUCCESS(bytes_read, read_size)`
    // post-read check here as we probably read more data than originally
    // requested, due to the sector alignment.  Determine the actual bytes
    // read by subtracting the bytes to skip and ignore from the total bytes
    // read.
    size_t actual_bytes_read = total_bytes_read - bytes_to_skip - bytes_to_ignore;
    CUDF_EXPECTS_READ_SUCCESS(actual_bytes_read, read_size);

    // Fast-path exit for the case where the caller provided sector-aligned
    // values for everything.
    if (use_caller_buffer) { return actual_bytes_read; }

    // Invariant check: the number of readable bytes left in the aligned
    // buffer after accounting for the bytes to skip should be equal to or
    // greater than the read size (which should be the allocated size of the
    // caller's buffer).
    auto const remaining_bytes = aligned_read_size - bytes_to_skip;
    if (remaining_bytes < read_size) {
      CUDF_LOG_ERROR("Invariant check failed: remaining_bytes ({}) >= read_size ({})",
                     remaining_bytes,
                     read_size);
      CUDF_FAIL("Invariant check failed: remaining_bytes >= read_size");
    }

    // We can now safely copy the requested data from the aligned buffer to
    // the caller's buffer, skipping the bytes at the beginning and ignoring
    // the bytes at the end (by way of the read size possibly being less than
    // the aligned buffer size).
    std::memcpy(dst, buf + bytes_to_skip, read_size);

    // Finally, return the actual bytes we read back to the caller.
    return actual_bytes_read;
  }

 private:
  odirect_datasource_params _params;  ///< O_DIRECT parameters
  int _fd_o_direct{-1};               ///< O_DIRECT file descriptor
  size_t _sector_size;                ///< Sector size for O_DIRECT I/O
};

/**
 * @brief Kvikio-based datasource.
 */
class kvikio_source : public host_source {
 public:
  kvikio_source(std::string const& filepath, kvikio_datasource_params const& params)
    : host_source(filepath), _params(params), _kvikio_file(filepath)
  {
    datasource_kind kind =
      (params.use_compat_mode) ? datasource_kind::KVIKIO_COMPAT : datasource_kind::KVIKIO_GDS;
    set_datasource_kind(kind);
  }

  [[nodiscard]] size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const read_size = get_read_size(size, offset);

    auto future =
      _kvikio_file.pread(dst, read_size, offset, _params.task_size, _params.device_read_threshold);
    return future.get();
  }

  [[nodiscard]] std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto const read_size = get_read_size(size, offset);

    std::vector<uint8_t> v(read_size);
    auto future = _kvikio_file.pread(
      v.data(), read_size, offset, _params.task_size, _params.device_read_threshold);
    future.get();
    return buffer::create(std::move(v));
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  [[nodiscard]] bool is_device_read_preferred(size_t size) const override
  {
    return size >= _params.device_read_threshold;
  }

  [[nodiscard]] std::unique_ptr<datasource::buffer> device_read(
    size_t offset, size_t size, rmm::cuda_stream_view stream) override
  {
    auto const read_size = get_read_size(size, offset);
    rmm::device_buffer out_data(read_size, stream);
    auto dst          = reinterpret_cast<uint8_t*>(out_data.data());
    size_t bytes_read = device_read_async(offset, size, dst, stream).get();
    out_data.resize(bytes_read, stream);
    return datasource::buffer::create(std::move(out_data));
  }

  [[nodiscard]] std::future<size_t> device_read_async(size_t offset,
                                                      size_t size,
                                                      uint8_t* dst,
                                                      rmm::cuda_stream_view stream) override
  {
    return _kvikio_file.pread(dst, size, offset, _params.task_size, _params.device_read_threshold);
  }

  [[nodiscard]] size_t size() const override { return _kvikio_file.nbytes(); }

 private:
  std::string _filepath;             ///< The path to the file
  kvikio_datasource_params _params;  ///< Kvikio parameters
  kvikio::FileHandle _kvikio_file;   ///< Kvikio file handle
};

/**
 * @brief Base class for file input. Only implements direct device reads.
 */
class file_source : public datasource {
 public:
  explicit file_source(char const* filepath)
    : datasource(datasource_kind::KVIKIO), _file(filepath, O_RDONLY)
  {
    detail::force_init_cuda_context();
    if (cufile_integration::is_kvikio_enabled()) {
      cufile_integration::set_thread_pool_nthreads_from_env();
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
    auto const read_size  = get_read_size(size, offset);
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
    set_datasource_kind(datasource_kind::HOST_MMAP);
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
  explicit device_buffer_source(cudf::device_span<std::byte const> d_buffer)
    : datasource(datasource_kind::OTHER), _d_buffer{d_buffer}
  {
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const count  = get_read_size(size, offset);
    auto const stream = cudf::detail::global_cuda_stream_pool().get_stream();
    cudf::detail::cuda_memcpy(host_span<uint8_t>{dst, count},
                              device_span<uint8_t const>{
                                reinterpret_cast<uint8_t const*>(_d_buffer.data() + offset), count},
                              stream);
    return count;
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto const count  = get_read_size(size, offset);
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
  explicit host_buffer_source(cudf::host_span<std::byte const> h_buffer)
    : datasource(datasource_kind::OTHER), _h_buffer{h_buffer}
  {
  }

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
 * @brief Implementation class that wraps a GDS-enabled cuFile input object.
 *
 * N.B. Named `cufile_source` instead of `gds_source` as cuFile is more
 * descriptive of the underlying implementation.
 */
class cufile_source : public host_source {
 public:
  cufile_source(std::string const& filepath, gds_datasource_params const& params)
    : host_source(filepath), _params(params)
  {
    set_datasource_kind(datasource_kind::GDS);
    _cufile_in = detail::make_cufile_input(filepath);
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  [[nodiscard]] bool is_device_read_preferred(size_t size) const override
  {
    return size >= _params.device_read_threshold;
  }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override
  {
    auto const read_size = get_read_size(size, offset);
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

 private:
  gds_datasource_params _params;                          ///< GDS parameters
  std::unique_ptr<detail::cufile_input_impl> _cufile_in;  ///< cuFile input obj
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
  explicit user_datasource_wrapper(datasource* const source)
    : datasource(datasource_kind::OTHER), source(source)
  {
  }

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
                                               datasource_kind kind,
                                               std::optional<const datasource_params> params)
{
  auto const use_memory_mapping = [] {
    auto const policy = getenv_or("LIBCUDF_MMAP_ENABLED", std::string{"ON"});

    if (policy == "ON") { return true; }
    if (policy == "OFF") { return false; }

    CUDF_FAIL("Invalid LIBCUDF_MMAP_ENABLED value: " + policy);
  }();

  if (use_memory_mapping) { kind = datasource_kind::HOST_MMAP; }

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
          CUDF_FAIL("Invalid parameters for KVIKIO-based datasource.");
        }
      }
      if (kind == datasource_kind::KVIKIO_COMPAT) {
        // Forcibly-set the compatibility mode to true, regardless of what may
        // already be present in the params.  The `kind` parameter has requested
        // `KVIKIO_COMPAT`, and that takes precedence over the `use_compat_mode`
        // parameter in the `kvikio_datasource_params`.
        new_params.use_compat_mode = true;
      } else if (kind == datasource_kind::KVIKIO_GDS) {
        // GDS is unique in that we are expected to throw a cudf::logic_error
        // if GDS is not available.  The first chance we have to do this is
        // here, by way of fencing against CUFILE_FOUND.
#ifndef CUFILE_FOUND
        CUDF_FAIL("GDS is not available because cuFile is not enabled.");
#endif
        // The next check is done against the `is_gds_enabled()` function in
        // `cufile_integration`.  If GDS is not enabled, we balk here too.
        CUDF_EXPECTS(cufile_integration::is_gds_enabled(), "cuFile reports GDS is not available.");
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
    case datasource_kind::GDS: {
      gds_datasource_params new_params;
      if (params) {
        if (auto gds_params = std::get_if<gds_datasource_params>(&params.value())) {
          // Copy the user-provided parameters into our local variable.
          new_params = *gds_params;
        } else {
          CUDF_FAIL("Invalid parameters for GDS-based datasource.");
        }
      }
      return std::make_unique<cufile_source>(filepath.c_str(), new_params);
    }
    case datasource_kind::HOST: return std::make_unique<host_source>(filepath);
    case datasource_kind::ODIRECT: {
      odirect_datasource_params new_params;
      if (params) {
        if (auto odirect_params = std::get_if<odirect_datasource_params>(&params.value())) {
          // Copy the user-provided parameters into our local variable.
          new_params = *odirect_params;
        } else {
          CUDF_FAIL("Invalid parameters for O_DIRECT-based datasource.");
        }
      }
      return std::make_unique<odirect_source>(filepath.c_str(), new_params);
    }
    case datasource_kind::HOST_MMAP:
      return std::make_unique<memory_mapped_source>(filepath.c_str(), offset, max_size_estimate);
    default: CUDF_FAIL("Unsupported datasource kind");
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
