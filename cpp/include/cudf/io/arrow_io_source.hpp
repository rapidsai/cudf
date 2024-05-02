/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "datasource.hpp"

#include <arrow/filesystem/filesystem.h>
#include <arrow/io/interfaces.h>

#include <memory>
#include <string>

namespace cudf::io {
/**
 * @addtogroup io_datasources
 * @{
 * @file
 */

/**
 * @brief Implementation class for reading from an Apache Arrow file. The file
 * could be a memory-mapped file or other implementation supported by Arrow.
 */
class arrow_io_source : public datasource {
 public:
  /**
   * @brief Constructs an object from an Apache Arrow Filesystem URI
   *
   * @param arrow_uri Apache Arrow Filesystem URI
   */
  explicit arrow_io_source(std::string const& arrow_uri);

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
  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override;

  /**
   * @brief Reads a selected range from the `arrow` source into a preallocated buffer.
   *
   * @param[in] offset The offset in bytes from which to read
   * @param[in] size The number of bytes to read
   * @param[out] dst The preallocated buffer to read into
   * @return The number of bytes read
   */
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;
  /**
   * @brief Returns the size of the data in the `arrow` source.
   *
   * @return The size of the data in the `arrow` source
   */
  [[nodiscard]] size_t size() const override;

 private:
  std::shared_ptr<arrow::fs::FileSystem> filesystem;
  std::shared_ptr<arrow::io::RandomAccessFile> arrow_file;
};

/** @} */  // end of group
}  // namespace cudf::io
