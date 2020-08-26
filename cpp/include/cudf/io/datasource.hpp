/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <arrow/buffer.h>
#include <arrow/io/file.h>
#include <arrow/io/interfaces.h>
#include <arrow/io/memory.h>

#include <memory>

#include <cudf/io/types.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {
//! IO interfaces
namespace io {

/**
 * @brief Interface class for providing input data to the readers.
 */
class datasource {
 public:
  /**
   * @brief Interface class for buffers that the datasource returns to the caller.
   *
   * Provides a basic interface to return the data address and size.
   */
  class buffer {
   public:
    /**
     * @brief Returns the buffer size in bytes.
     */
    virtual size_t size() const = 0;

    /**
     * @brief Returns the address of the data in the buffer.
     */
    virtual const uint8_t* data() const = 0;

    /**
     * @brief Base class destructor
     */
    virtual ~buffer() {}
  };

  /**
   * @brief Creates a source from a file path.
   *
   * @param[in] filepath Path to the file to use
   * @param[in] offset Bytes from the start of the file (the default is zero)
   * @param[in] size Bytes from the offset; use zero for entire file (the default is zero)
   */
  static std::unique_ptr<datasource> create(const std::string& filepath,
                                            size_t offset = 0,
                                            size_t size   = 0);

  /**
   * @brief Creates a source from a memory buffer.
   *
   * @param[in] buffer Host buffer object
   */
  static std::unique_ptr<datasource> create(host_buffer const& buffer);

  /**
   * @brief Creates a source from a from an Arrow file.
   *
   * @param[in] arrow_file RandomAccessFile to which the API calls are forwarded
   */
  static std::unique_ptr<datasource> create(
    std::shared_ptr<arrow::io::RandomAccessFile> arrow_file);

  /**
   * @brief Creates a source from an user implemented datasource object.
   *
   * @param[in] source Non-owning pointer to the datasource object
   */
  static std::unique_ptr<datasource> create(datasource* source);

  /**
   * @brief Creates a vector of datasources, one per element in the input vector.
   *
   * @param[in] args vector of parameters
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
   * @return The data buffer
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
  virtual bool supports_device_read() const { return false; }

  /**
   * @brief Returns a device buffer with a subset of data from the source.
   *
   * Data source implementations that don't support direct device reads don't need to override this
   * function.
   *
   * @param[in] offset Bytes from the start
   * @param[in] size Bytes to read
   *
   * @return The data buffer in the device memory
   */
  virtual std::unique_ptr<datasource::buffer> device_read(size_t offset, size_t size)
  {
    CUDF_FAIL("datasource classes that support device_read must override this function.");
  }

  /**
   * @brief Reads a selected range into a preallocated device buffer
   *
   * Data source implementations that don't support direct device reads don't need to override this
   * function.
   *
   * @param[in] offset Bytes from the start
   * @param[in] size Bytes to read
   * @param[in] dst Address of the existing device memory
   *
   * @return The number of bytes read (can be smaller than size)
   */
  virtual size_t device_read(size_t offset, size_t size, uint8_t* dst)
  {
    CUDF_FAIL("datasource classes that support device_read must override this function.");
  }

  /**
   * @brief Returns the size of the data in the source.
   *
   * @return size_t The size of the source data in bytes
   */
  virtual size_t size() const = 0;

  /**
   * @brief Returns whether the source contains any data.
   *
   * @return bool True if there is data, False otherwise
   */
  virtual bool is_empty() const { return size() == 0; }

  /**
   * @brief Implementation for non owning buffer where datasource holds buffer until destruction.
   */
  class non_owning_buffer : public buffer {
   public:
    non_owning_buffer() : _data(0), _size(0) {}

    non_owning_buffer(uint8_t* data, size_t size) : _data(data), _size(size) {}

    size_t size() const override { return _size; }

    const uint8_t* data() const override { return _data; }

   private:
    uint8_t* const _data;
    size_t const _size;
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
    size_t size() const override { return arrow_buffer->size(); }
    const uint8_t* data() const override { return arrow_buffer->data(); }
  };

 public:
  /**
   * @brief Constructs an object from an `arrow` source object.
   *
   * @param file The `arrow` object from which the data is read
   */
  explicit arrow_io_source(std::shared_ptr<arrow::io::RandomAccessFile> file) : arrow_file(file) {}

  /**
   * @brief Returns a buffer with a subset of data from the `arrow` source.
   */
  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto result = arrow_file->ReadAt(offset, size);
    CUDF_EXPECTS(result.ok(), "Cannot read file data");
    return std::make_unique<arrow_io_buffer>(result.ValueOrDie());
  }

  /**
   * @brief Reads a selected range from the `arrow` source into a preallocated buffer.
   */
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto result = arrow_file->ReadAt(offset, size, dst);
    CUDF_EXPECTS(result.ok(), "Cannot read file data");
    return result.ValueOrDie();
  }

  /**
   * @brief Returns the size of the data in the `arrow` source.
   */
  size_t size() const override
  {
    auto result = arrow_file->GetSize();
    CUDF_EXPECTS(result.ok(), "Cannot get file size");
    return result.ValueOrDie();
  }

 private:
  std::shared_ptr<arrow::io::RandomAccessFile> arrow_file;
};

}  // namespace io
}  // namespace cudf
