/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>

#include <arrow/buffer.h>

// We disable warning 611 because some Arrow subclasses of
// `arrow::fs::FileSystem` only partially override the `Equals` method,
// triggering warning 611-D from nvcc.
#pragma nv_diag_suppress 611
#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/s3fs.h>
#pragma nv_diag_default 611

#include <arrow/io/file.h>
#include <arrow/io/interfaces.h>
#include <arrow/io/memory.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include <future>
#include <memory>

namespace cudf {
//! IO interfaces
namespace io {

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
   * @param[in] filepath Path to the file to use
   * @param[in] offset Bytes from the start of the file (the default is zero)
   * @param[in] size Bytes from the offset; use zero for entire file (the default is zero)
   * @return Constructed datasource object
   */
  static std::unique_ptr<datasource> create(const std::string& filepath,
                                            size_t offset = 0,
                                            size_t size   = 0);

  /**
   * @brief Creates a source from a memory buffer.
   *
   * @param[in] buffer Host buffer object
   * @return Constructed datasource object
   */
  static std::unique_ptr<datasource> create(host_buffer const& buffer);

  /**
   * @brief Creates a source from a from an Arrow file.
   *
   * @param[in] arrow_file RandomAccessFile to which the API calls are forwarded
   * @return Constructed datasource object
   */
  static std::unique_ptr<datasource> create(
    std::shared_ptr<arrow::io::RandomAccessFile> arrow_file);

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
   * @return size_t The size of the source data in bytes
   */
  [[nodiscard]] virtual size_t size() const = 0;

  /**
   * @brief Returns whether the source contains any data.
   *
   * @return bool True if there is data, False otherwise
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
    non_owning_buffer(uint8_t* data, size_t size) : _data(data), _size(size) {}

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
    uint8_t* const _data{nullptr};
    size_t const _size{0};
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
};

/**
 * @brief Implementation class for reading from an Apache Arrow file. The file
 * could be a memory-mapped file or other implementation supported by Arrow.
 */
class arrow_io_source : public datasource {
  /**
   * @brief Implementation for an owning buffer where `arrow::Buffer` holds the data.
   */
  class arrow_io_buffer : public buffer {
    std::shared_ptr<arrow::Buffer> arrow_buffer;

   public:
    explicit arrow_io_buffer(std::shared_ptr<arrow::Buffer> arrow_buffer)
      : arrow_buffer(arrow_buffer)
    {
    }
    [[nodiscard]] size_t size() const override { return arrow_buffer->size(); }
    [[nodiscard]] uint8_t const* data() const override { return arrow_buffer->data(); }
  };

 public:
  /**
   * @brief Constructs an object from an Apache Arrow Filesystem URI
   *
   * @param arrow_uri Apache Arrow Filesystem URI
   */
  explicit arrow_io_source(std::string_view arrow_uri)
  {
    const std::string uri_start_delimiter = "//";
    const std::string uri_end_delimiter   = "?";

    arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> result =
      arrow::fs::FileSystemFromUri(static_cast<std::string>(arrow_uri));
    CUDF_EXPECTS(result.ok(), "Failed to generate Arrow Filesystem instance from URI.");
    filesystem = result.ValueOrDie();

    // Parse the path from the URI
    size_t start          = arrow_uri.find(uri_start_delimiter) == std::string::npos
                              ? 0
                              : arrow_uri.find(uri_start_delimiter) + uri_start_delimiter.size();
    size_t end            = arrow_uri.find(uri_end_delimiter) - start;
    std::string_view path = arrow_uri.substr(start, end);

    arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> in_stream =
      filesystem->OpenInputFile(static_cast<std::string>(path).c_str());
    CUDF_EXPECTS(in_stream.ok(), "Failed to open Arrow RandomAccessFile");
    arrow_file = in_stream.ValueOrDie();
  }

  /**
   * @brief Constructs an object from an `arrow` source object.
   *
   * @param file The `arrow` object from which the data is read
   */
  explicit arrow_io_source(std::shared_ptr<arrow::io::RandomAccessFile> file) : arrow_file(file) {}

  /**
   * @brief Returns a buffer with a subset of data from the `arrow` source.
   *
   * @param offset The offset in bytes from which to read
   * @param size The number of bytes to read
   * @return A buffer with the read data
   */
  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto result = arrow_file->ReadAt(offset, size);
    CUDF_EXPECTS(result.ok(), "Cannot read file data");
    return std::make_unique<arrow_io_buffer>(result.ValueOrDie());
  }

  /**
   * @brief Reads a selected range from the `arrow` source into a preallocated buffer.
   *
   * @param[in] offset The offset in bytes from which to read
   * @param[in] size The number of bytes to read
   * @param[out] dst The preallocated buffer to read into
   * @return The number of bytes read
   */
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto result = arrow_file->ReadAt(offset, size, dst);
    CUDF_EXPECTS(result.ok(), "Cannot read file data");
    return result.ValueOrDie();
  }

  /**
   * @brief Returns the size of the data in the `arrow` source.
   *
   * @return The size of the data in the `arrow` source
   */
  [[nodiscard]] size_t size() const override
  {
    auto result = arrow_file->GetSize();
    CUDF_EXPECTS(result.ok(), "Cannot get file size");
    return result.ValueOrDie();
  }

 private:
  std::shared_ptr<arrow::fs::FileSystem> filesystem;
  std::shared_ptr<arrow::io::RandomAccessFile> arrow_file;
};

}  // namespace io
}  // namespace cudf
