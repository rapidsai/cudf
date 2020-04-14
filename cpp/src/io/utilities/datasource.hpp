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

#include <arrow/buffer.h>
#include <arrow/io/file.h>
#include <arrow/io/interfaces.h>
#include <arrow/io/memory.h>

#include <algorithm>
#include <memory>
#include <string>

#include <rmm/device_buffer.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace io {

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
                                            size_t offset = 0, size_t size = 0);

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
  static std::unique_ptr<datasource> create(
      std::shared_ptr<arrow::io::RandomAccessFile> file);

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
   * @return std::shared_ptr<arrow::Buffer> The data buffer
   **/
  virtual const std::shared_ptr<arrow::Buffer> get_buffer(size_t offset,
                                                          size_t size) = 0;

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

  /**
   * @brief Returns a device buffer with a subset of data from the source
   *
   * @param[in] offset Bytes from the start
   * @param[in] size Bytes to read
   * @param[in] stream cuda stream to use
   *
   * @return rmm::device_buffer the data buffer
   **/
  virtual rmm::device_buffer get_device_buffer(size_t offset, size_t size, cudaStream_t stream = 0) {
    auto host_buffer = get_buffer(offset, size);
    auto device_buffer = rmm::device_buffer(host_buffer->data(), host_buffer->size(), stream);
    CUDA_TRY(cudaStreamSynchronize(stream)); // Due to host_buffer scope
    return std::move(device_buffer);
  }

  /**
   * @brief Reads bytes to device memory
   *
   * @param[in] offset Bytes from the start
   * @param[in] size Bytes to read
   * @param[out] dst Destination
   * @param[in] stream cuda stream to use
   *
   **/
  virtual void device_read(size_t offset, size_t size, void *dst, cudaStream_t stream = 0) {
    auto host_buffer = get_buffer(offset, size);
    CUDA_TRY(cudaMemcpyAsync(dst, host_buffer->data(), host_buffer->size(), cudaMemcpyHostToDevice, stream));
  }
};

}  // namespace io
}  // namespace cudf
