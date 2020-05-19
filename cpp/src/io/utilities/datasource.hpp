/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/utilities/error.hpp>

#include <arrow/buffer.h>
#include <arrow/io/file.h>
#include <arrow/io/interfaces.h>
#include <arrow/io/memory.h>

#include <algorithm>
#include <memory>
#include <string>

namespace cudf {
namespace io {

class buffer {
 public:
  virtual size_t size() const = 0;

  virtual const uint8_t *data() const = 0;
};

/**
 * @brief Class for reading from a file or memory source
 **/
class datasource {
 public:
  /**
   * @brief Create a source from a file path
   *
   * @param[in] filepath Path to the file to use
   * @param[in] offset Bytes from the start of the file
   * @param[in] size Bytes from the offset; use zero for entire file
   **/
  static std::unique_ptr<datasource> create(const std::string filepath,
                                            size_t offset = 0,
                                            size_t size   = 0);

  /**
   * @brief Create a source from a memory buffer
   *
   * @param[in] filepath Path to the file to use
   **/
  static std::unique_ptr<datasource> create(const char *data, size_t length);

  /**
   * @brief Create a source from a from an Arrow file
   *
   * @param[in] filepath Path to the file to use
   **/
  static std::unique_ptr<datasource> create(std::shared_ptr<arrow::io::RandomAccessFile> file);

  /**
   * @brief Base class destructor
   **/
  virtual ~datasource(){};

  /**
   * @brief Returns a buffer with a subset of data from the source
   *
   * @param[in] offset Bytes from the start
   * @param[in] size Bytes to read
   *
   * @return The data buffer
   **/
  virtual std::unique_ptr<buffer> host_read(size_t offset, size_t size) = 0;

  /**
   * @brief Read a selected range into a preallocated buffer
   *
   * @param[in] offset Bytes from the start
   * @param[in] size Bytes to read
   * @param[in] dst Address of the existing host memory
   *
   * @return The number of bytes read (can be smaller than size)
   **/
  virtual size_t host_read(size_t offset, size_t size, uint8_t *dst) = 0;

  /**
   * @brief Whether or not this source supports reading directly into device memory
   *
   * If this function returns true, the data_sink will receive calls to device_read()
   * instead of host_read() when possible.  However, it is still possible to receive
   * host_read() calls as well.
   *
   * @return bool Whether this source supports device_read() calls.
   **/
  virtual bool supports_device_write() const { return false; }

  /**
   * @brief Returns a device buffer with a subset of data from the source
   *
   * @param[in] offset Bytes from the start
   * @param[in] size Bytes to read
   *
   * @return The data buffer in the device memory
   **/
  virtual std::unique_ptr<buffer> device_read(size_t offset, size_t size)
  {
    CUDF_FAIL("datasource classes that support device_read must override this function.");
  }

  /**
   * @brief Read a selected range into a preallocated device buffer
   *
   * @param[in] offset Bytes from the start
   * @param[in] size Bytes to read
   * @param[in] dst Address of the existing device memory
   *
   * @return The number of bytes read (can be smaller than size)
   **/
  virtual size_t device_read(size_t offset, size_t size, uint8_t *dst)
  {
    CUDF_FAIL("datasource classes that support device_read must override this function.");
  }

  /**
   * @brief Returns the size of the data in the source
   *
   * @return size_t The size of the source data in bytes
   **/
  virtual size_t size() const = 0;

  /**
   * @brief Returns whether the data source contains any actual data
   *
   * @return bool True if there is data, False otherwise
   **/
  virtual bool empty() const { return size() == 0; }
};

}  // namespace io
}  // namespace cudf
