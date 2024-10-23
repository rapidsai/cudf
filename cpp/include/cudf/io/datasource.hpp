/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/io/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <future>
#include <memory>
#include <variant>

namespace CUDF_EXPORT cudf {
//! IO interfaces
namespace io {

/**
 * @addtogroup io_datasources
 * @{
 * @file
 */

/**
 * @brief Kind of data source to create when calling
 * `cudf::io::datasource::create()`.
 *
 * @see cudf::io::datasource::create()
 * @see cudf::io::datasource_params
 *
 * N.B. GDS = GPUDirect Storage
 */
enum class datasource_kind {
  /**
   * @brief Kvikio-based data source (default).
   *
   * This data source is the default for cuDF, and should be the most performant
   * option for most use cases.  It supports GDS where possible, falling back to
   * multi-threaded host-based reads when GDS is not available.
   *
   * It supports asynchronous reads, and will use the provided CUDA stream for
   * all I/O operations when possible.
   */
  KVIKIO  = 0,
  DEFAULT = KVIKIO,

  /**
   * @brief Kvikio-based data source that does not attempt to use GDS, instead
   * falling back to multi-threaded host-based reads.
   *
   * It supports asynchronous reads, but does not do any stream synchronization,
   * as the reads are all performed on the host.
   */
  KVIKIO_COMPAT,

  /**
   * @brief Kvikio-based data source that will fail if GDS is not available.
   * Specifically, `cudf::io::datasource::create()` when called with this kind
   * of data source will throw a `cudf::logic_error` if GDS is not available.
   */
  KVIKIO_GDS,

  /**
   * @brief Host-based data source that does not support any device or async
   * operations.
   *
   * All reads are performed via standard POSIX pread() calls.  No
   * multi-threading or asynchronous operations are supported.
   *
   * The primary purpose of this datasource type is to be a base class for the
   * `O_DIRECT` implementation, which needs to issue pread() calls against a
   * file descriptor that *hasn't* been opened with `O_DIRECT` if certain
   * constraints aren't met (specifically: when reading the final bytes of a
   * file that isn't perfectly aligned to a sector-size boundary).
   *
   * The time required to service reads from this data source will be affected
   * by the presence or absence of the desired data in the Linux page cache.
   * Thus, back-to-back runs of the same file will have significantly different
   * performance characteristics, depending on whether the data is in the page
   * cache or not.
   *
   * Generally, this data source should be avoided in favor of the `KVIKIO`
   * data source, which will be more performant in most cases.  Thus, it can
   * be used as a baseline for which improved `KVIKIO` performance can be
   * empirically measured.
   */
  HOST,

  /**
   * @brief Host-based data source that issues reads against a file descriptor
   * opened with `O_DIRECT`, where possible, bypassing the Linux page cache.
   *
   * This data source will always result in the slowest possible read times,
   * as all reads are serviced directly from the underlying device.  However,
   * it will be consistently slow, and that consistency can be critical when
   * benchmarking or profiling changes purporting to improve performance in
   * unrelated areas.
   *
   * Thus, the primary use case for this data source is for benchmarking and
   * profiling purposes, where you want to eliminate any runtime variance in
   * back-to-back runs that would be caused by the presence or absence of data
   * in the host's page cache.
   *
   * A secondary use case for this data source is when you specifically do not
   * want to pollute the host's page cache with the data being read, either
   * because it won't be read again soon, or you want to remove the memory
   * pressure (or small but non-trivial amount of compute overhead) that would
   * otherwise be introduced by servicing I/O through the page cache.  In some
   * scenarios, this can yield a net performance improvement, despite a higher
   * per-read latency.
   *
   * A real-life example of how this can manifest is when doing very large TPC-H
   * or TPC-DS runs, where the data set is orders of magnitude larger than the
   * available host memory, e.g. 30TB or 100TB runs on hosts with <= 4TB of RAM.
   *
   * For certain queries--typically read-heavy, join-heavy ones--doing `O_DIRECT`
   * reads can result in a net performance improvement, as the host's page cache
   * won't be polluted with data that will never be read again, and compute
   * overhead associated with cache thrashing when memory is tight is
   * eliminated.
   */
  ODIRECT,

  /**
   * @brief Host-based data source that uses memory mapped files to satisfy
   * read requests.
   *
   * Note that this can result in pathological performance problems in certain
   * environments, such as when small reads are done against files residing on
   * a network file system (including accelerated file systems like WekaFS).
   */
  HOST_MMAP,

  /**
   * @brief This is a special sentinel value that is used to indicate the
   * datasource is not one of the publicly available types above.
   *
   * N.B. You cannot create a datasource of this kind directly via create().
   */
  OTHER,
};

/**
 * @brief Parameters for the kvikio data source.
 */
struct kvikio_datasource_params {
  /**
   * @brief When set, explicitly disables any attempts at using GDS, resulting
   * resulting in kvikio falling back to its "compat" mode using multi-threaded
   * host-based reads.
   *
   * Defaults to false.
   *
   * N.B. Compat mode will still be used if GDS isn't available, regardless
   *      of the value of this parameter.
   */
  bool use_compat_mode{false};

  /**
   * @brief The threshold at which the data source will switch from using
   * host-based reads to device-based (GDS) reads, if GDS is available.
   *
   * This parameter should represent the read size where GDS is faster than
   * a posix read() plus the overhead of a host-to-device memcpy.
   *
   * Defaults to 128KB.
   */
  size_t device_read_threshold{128 << 10};

  /**
   * @brief The number of threads in the kvikio thread pool.
   *
   * This parameter only applies to the kvikio data source when GDS is not
   * available and it is in compat mode.
   *
   * Defaults to 0, which defers the thread pool sizing to kvikio.
   */
  uint16_t num_threads{0};

  /**
   * @brief The size in bytes into which I/O operations will be split.
   *
   * Defaults to 1MB.
   */
  size_t task_size{1 << 20};
};

/**
 * @brief Parameters for the `O_DIRECT` data source.
 */
struct odirect_datasource_params {
  /**
   * @brief The sector size, in bytes, to use for alignment when issuing
   * `O_DIRECT` reads.  This size dictates the alignment used for three things:
   * the file offset, the buffer address, and the buffer size.  It *must* be a
   * multiple of the underlying device's sector size, which is typically 512
   * bytes.  A larger size is fine as long as it's a multiple of the device's
   * sector size.
   *
   * Defaults to 4096.
   *
   * N.B. On Linux, you can determine the sector size of a device with the
   *      the `blockdev` command, e.g.: `sudo blockdev --getss /dev/sda`.
   */
  size_t sector_size{4096};

  /**
   * @brief The minimum permissible sector size.  All sector sizes must be a
   * multiple of this value.  This is hardcoded to 512 bytes as a simple means
   * to catch misconfigurations.  The underlying device's sector size may be
   * larger, but it will certainly be a multiple of this value.
   */
  static constexpr size_t min_sector_size{512};

  /**
   * @brief Returns true iff the sector size is a multiple of the minimum sector
   * size.
   */
  [[nodiscard]] bool is_valid_sector_size() const
  {
    return ((sector_size > 0) && ((sector_size % min_sector_size) == 0));
  }
};

/**
 * @brief Union of parameters for different data sources.
 */
using datasource_params = std::variant<kvikio_datasource_params, odirect_datasource_params>;

/**
 * @brief Interface class for providing input data to the readers.
 */
class datasource {
 public:
  template <typename Container>
  class owning_buffer;  // forward declaration
  /**
   * @brief Interface class for buffers that the datasource returns to the caller.
   *
   * Provides a basic interface to return the data address and size.
   */
  class buffer {
   public:
    /**
     * @pure @brief Returns the buffer size in bytes.
     *
     * @return Buffer size in bytes
     */
    [[nodiscard]] virtual size_t size() const = 0;

    /**
     * @pure @brief Returns the address of the data in the buffer.
     *
     * @return Address of the data in the buffer
     */
    [[nodiscard]] virtual uint8_t const* data() const = 0;

    /**
     * @brief Base class destructor
     */
    virtual ~buffer() {}

    /**
     * @brief Factory to construct a datasource buffer object from a container.
     *
     * @tparam Container Type of the container to construct the buffer from
     * @param data_owner The container to construct the buffer from (ownership is transferred)
     * @return Constructed buffer object
     */
    template <typename Container>
    static std::unique_ptr<buffer> create(Container&& data_owner)
    {
      return std::make_unique<owning_buffer<Container>>(std::move(data_owner));
    }
  };

  /**
   * @brief Creates a source from a file path.
   *
   * Parameters `offset` and `max_size_estimate` are hints to the `datasource` implementation about
   * the expected range of the data that will be read. The implementation may use these hints to
   * optimize the read operation. These parameters are usually based on the byte range option. In
   * this case, `max_size_estimate` can include padding after the byte range, to include additional
   * data that may be needed for processing.
   *
   * @throws cudf::logic_error if `KVIKIO_GDS` is specified as the desired kind of data source,
   * and GDS is not available for the file.
   *
   * @throws cudf::logic_error if `params` are supplied but do not match the
   * kind of data source being created as indicated by the `kind` parameter.
   *
   * @param[in] filepath Path to the file to use
   * @param[in] offset Starting byte offset from which data will be read (the default is zero)
   * @param[in] max_size_estimate Upper estimate of the data range that will be read (the default is
   * zero, which means the whole file after `offset`)
   * @param[in] kind Optionally supplies the kind of data source to create
   * @param[in] params Optionally supplies parameters for the data source
   * @return Constructed datasource object
   */
  static std::unique_ptr<datasource> create(
    std::string const& filepath,
    size_t offset                                 = 0,
    size_t max_size_estimate                      = 0,
    datasource_kind kind                          = datasource_kind::DEFAULT,
    std::optional<const datasource_params> params = std::nullopt);

  /**
   * @brief Creates a source from a host memory buffer.
   *
   # @deprecated Since 23.04
   *
   * @param[in] buffer Host buffer object
   * @return Constructed datasource object
   */
  static std::unique_ptr<datasource> create(host_buffer const& buffer);

  /**
   * @brief Creates a source from a host memory buffer.
   *
   * @param[in] buffer Host buffer object
   * @return Constructed datasource object
   */
  static std::unique_ptr<datasource> create(cudf::host_span<std::byte const> buffer);

  /**
   * @brief Creates a source from a device memory buffer.
   *
   * @param buffer Device buffer object
   * @return Constructed datasource object
   */
  static std::unique_ptr<datasource> create(cudf::device_span<std::byte const> buffer);

  /**
   * @brief Creates a source from an user implemented datasource object.
   *
   * @param[in] source Non-owning pointer to the datasource object
   * @return Constructed datasource object
   */
  static std::unique_ptr<datasource> create(datasource* source);

  /**
   * @brief Creates a vector of datasources, one per element in the input vector.
   *
   * @param[in] args vector of parameters
   * @return Constructed vector of datasource objects
   */
  template <typename T>
  static std::vector<std::unique_ptr<datasource>> create(std::vector<T> const& args)
  {
    std::vector<std::unique_ptr<datasource>> sources;
    sources.reserve(args.size());
    std::transform(args.cbegin(), args.cend(), std::back_inserter(sources), [](auto const& arg) {
      return datasource::create(arg);
    });
    return sources;
  }

  /**
   * @brief Base class destructor
   */
  virtual ~datasource(){};

  /**
   * @brief Returns a buffer with a subset of data from the source.
   *
   * @param[in] offset Bytes from the start
   * @param[in] size Bytes to read
   *
   * @return The data buffer (can be smaller than size)
   */
  virtual std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size) = 0;

  /**
   * @brief Reads a selected range into a preallocated buffer.
   *
   * @param[in] offset Bytes from the start
   * @param[in] size Bytes to read
   * @param[in] dst Address of the existing host memory
   *
   * @return The number of bytes read (can be smaller than size)
   */
  virtual size_t host_read(size_t offset, size_t size, uint8_t* dst) = 0;

  /**
   * @brief Whether or not this source supports reading directly into device memory.
   *
   * If this function returns true, the datasource will receive calls to device_read() instead of
   * host_read() when the reader processes the data on the device. Most readers will still make
   * host_read() calls, for the parts of input that are processed on the host (e.g. metadata).
   *
   * Data source implementations that don't support direct device reads don't need to override this
   * function. The implementations that do should override it to return false.
   *
   * @return bool Whether this source supports device_read() calls
   */
  [[nodiscard]] virtual bool supports_device_read() const { return false; }

  /**
   * @brief Estimates whether a direct device read would be more optimal for the given size.
   *
   * @param size Number of bytes to read
   * @return whether the device read is expected to be more performant for the given size
   */
  [[nodiscard]] virtual bool is_device_read_preferred(size_t size) const
  {
    return supports_device_read();
  }

  /**
   * @brief Returns a device buffer with a subset of data from the source.
   *
   * For optimal performance, should only be called when `is_device_read_preferred` returns `true`.
   * Data source implementations that don't support direct device reads don't need to override this
   * function.
   *
   *  @throws cudf::logic_error the object does not support direct device reads, i.e.
   * `supports_device_read` returns `false`.
   *
   * @param offset Number of bytes from the start
   * @param size Number of bytes to read
   * @param stream CUDA stream to use
   *
   * @return The data buffer in the device memory
   */
  virtual std::unique_ptr<datasource::buffer> device_read(size_t offset,
                                                          size_t size,
                                                          rmm::cuda_stream_view stream)
  {
    CUDF_FAIL("datasource classes that support device_read must override it.");
  }

  /**
   * @brief Reads a selected range into a preallocated device buffer
   *
   * For optimal performance, should only be called when `is_device_read_preferred` returns `true`.
   * Data source implementations that don't support direct device reads don't need to override this
   * function.
   *
   *  @throws cudf::logic_error when the object does not support direct device reads, i.e.
   * `supports_device_read` returns `false`.
   *
   * @param offset Number of bytes from the start
   * @param size Number of bytes to read
   * @param dst Address of the existing device memory
   * @param stream CUDA stream to use
   *
   * @return The number of bytes read (can be smaller than size)
   */
  virtual size_t device_read(size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream)
  {
    CUDF_FAIL("datasource classes that support device_read must override it.");
  }

  /**
   * @brief Asynchronously reads a selected range into a preallocated device buffer
   *
   * Returns a future value that contains the number of bytes read. Calling `get()` method of the
   * return value synchronizes this function.
   *
   * For optimal performance, should only be called when `is_device_read_preferred` returns `true`.
   * Data source implementations that don't support direct device reads don't need to override this
   * function.
   *
   *  @throws cudf::logic_error when the object does not support direct device reads, i.e.
   * `supports_device_read` returns `false`.
   *
   * @param offset Number of bytes from the start
   * @param size Number of bytes to read
   * @param dst Address of the existing device memory
   * @param stream CUDA stream to use
   *
   * @return The number of bytes read as a future value (can be smaller than size)
   */
  virtual std::future<size_t> device_read_async(size_t offset,
                                                size_t size,
                                                uint8_t* dst,
                                                rmm::cuda_stream_view stream)
  {
    CUDF_FAIL("datasource classes that support device_read_async must override it.");
  }

  /**
   * @brief Returns the size of the data in the source.
   *
   * @return The size of the source data in bytes
   */
  [[nodiscard]] virtual size_t size() const = 0;

  /**
   * @brief Returns whether the source contains any data.
   *
   * @return True if there is data, False otherwise
   */
  [[nodiscard]] virtual bool is_empty() const { return size() == 0; }

  /**
   * @brief Returns the appropriate size, in bytes, of a read request, given
   * the supplied requested size and offset.
   *
   * The returned size is clamped to ensure it does not exceed the total size
   * of the data source, once the requested size and offset are taken into
   * account.
   *
   * @param requested_size[in] Supplies the desired size of the read request,
   * in bytes.
   *
   * @param offset[in] Supplies the offset, in bytes, from the start of the
   * data.
   *
   * @return The size of the read request in bytes.  This will be the minimum
   * of the requested size and the remaining size of the data source after the
   * offset.  If the offset is beyond the end of the data source, this will
   * return 0.
   */
  [[nodiscard]] size_t get_read_size(size_t requested_size, size_t offset) const
  {
    return std::min(requested_size, size() > offset ? size() - offset : 0);
  }

  /**
   * @brief Returns the kind of data source.
   *
   * @return The kind of data source.
   */
  [[nodiscard]] datasource_kind kind() const { return _kind; }

  /**
   * @brief Implementation for non owning buffer where datasource holds buffer until destruction.
   */
  class non_owning_buffer : public buffer {
   public:
    non_owning_buffer() {}

    /**
     * @brief Construct a new non owning buffer object
     *
     * @param data The data buffer
     * @param size The size of the data buffer
     */
    non_owning_buffer(uint8_t const* data, size_t size) : _data(data), _size(size) {}

    /**
     * @brief Returns the size of the buffer.
     *
     * @return The size of the buffer in bytes
     */
    [[nodiscard]] size_t size() const override { return _size; }

    /**
     * @brief Returns the pointer to the buffer.
     *
     * @return Pointer to the buffer
     */
    [[nodiscard]] uint8_t const* data() const override { return _data; }

   private:
    uint8_t const* _data{nullptr};
    size_t _size{0};
  };

  /**
   * @brief Derived implementation of `buffer` that owns the data.
   *
   * Can use different container types to hold the data buffer.
   *
   * @tparam Container Type of the container object that owns the data
   */
  template <typename Container>
  class owning_buffer : public buffer {
   public:
    /**
     * @brief Moves the input container into the newly created object.
     *
     * @param data_owner The container to construct the buffer from (ownership is transferred)
     */
    owning_buffer(Container&& data_owner)
      : _data(std::move(data_owner)), _data_ptr(_data.data()), _size(_data.size())
    {
    }

    /**
     * @brief Moves the input container into the newly created object, and exposes a subspan of the
     * buffer.
     *
     * @param data_owner The container to construct the buffer from (ownership is transferred)
     * @param data_ptr Pointer to the start of the subspan
     * @param size The size of the subspan
     */
    owning_buffer(Container&& data_owner, uint8_t const* data_ptr, size_t size)
      : _data(std::move(data_owner)), _data_ptr(data_ptr), _size(size)
    {
    }

    /**
     * @brief Returns the size of the buffer.
     *
     * @return The size of the buffer in bytes
     */
    [[nodiscard]] size_t size() const override { return _size; }

    /**
     * @brief Returns the pointer to the data in the buffer.
     *
     * @return Pointer to the data in the buffer
     */
    [[nodiscard]] uint8_t const* data() const override
    {
      return static_cast<uint8_t const*>(_data_ptr);
    }

   private:
    Container _data;
    void const* _data_ptr;
    size_t _size;
  };

 protected:
  /**
   * @brief Constructor for the datasource object.
   *
   * @param kind The kind of data source
   */
  datasource(datasource_kind kind) : _kind(kind) {}

  /**
   * @brief Sets the kind of data source.
   *
   * @note This is intended for use by derived classes that need to change the
   * kind of data source after construction.
   */
  void set_datasource_kind(datasource_kind kind) { _kind = kind; }

 private:
  datasource_kind _kind{datasource_kind::DEFAULT};
};

/** @} */  // end of group
}  // namespace io
}  // namespace CUDF_EXPORT cudf
