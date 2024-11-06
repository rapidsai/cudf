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

namespace CUDF_EXPORT cudf {
//! IO interfaces
namespace io {

/**
 * @addtogroup io_datasources
 * @{
 * @file
 */

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
      return std::make_unique<owning_buffer<Container>>(std::forward<Container>(data_owner));
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
   * @param[in] filepath Path to the file to use
   * @param[in] offset Starting byte offset from which data will be read (the default is zero)
   * @param[in] max_size_estimate Upper estimate of the data range that will be read (the default is
   * zero, which means the whole file after `offset`)
   * @return Constructed datasource object
   */
  static std::unique_ptr<datasource> create(std::string const& filepath,
                                            size_t offset            = 0,
                                            size_t max_size_estimate = 0);

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
    // Require that the argument passed to the constructor be an rvalue (Container&& being an rvalue
    // reference).
    static_assert(std::is_rvalue_reference_v<Container&&>,
                  "The container argument passed to the constructor must be an rvalue.");

    /**
     * @brief Moves the input container into the newly created object.
     *
     * @param moved_data_owner The container to construct the buffer from. Callers should explicitly
     * pass std::move(data_owner) to this function to transfer the ownership.
     */
    owning_buffer(Container&& moved_data_owner)
      : _data(std::move(moved_data_owner)), _data_ptr(_data.data()), _size(_data.size())
    {
    }

    /**
     * @brief Moves the input container into the newly created object, and exposes a subspan of the
     * buffer.
     *
     * @param moved_data_owner The container to construct the buffer from. Callers should explicitly
     * pass std::move(data_owner) to this function to transfer the ownership.
     * @param data_ptr Pointer to the start of the subspan
     * @param size The size of the subspan
     */
    owning_buffer(Container&& moved_data_owner, uint8_t const* data_ptr, size_t size)
      : _data(std::move(moved_data_owner)), _data_ptr(data_ptr), _size(size)
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
};

/** @} */  // end of group
}  // namespace io
}  // namespace CUDF_EXPORT cudf
