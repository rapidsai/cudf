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

#include "cudf.h"
#include "utilities/error_utils.hpp"

#include <arrow/buffer.h>
#include <arrow/io/file.h>
#include <arrow/io/interfaces.h>
#include <arrow/io/memory.h>

#include <memory>

/**
 * @brief Class for reading from a file or buffer source
 **/
class DataSource {
 public:
  /**
   * @brief Constructor for creating source from a file path
   *
   * @param[in] filepath Path to the file to use
   **/
  explicit DataSource(const char *filepath) {
    std::shared_ptr<arrow::io::ReadableFile> file;
    CUDF_EXPECTS(
        arrow::io::ReadableFile::Open(std::string(filepath), &file).ok(),
        "Cannot open file");
    rd_file = file;
  }

  /**
   * @brief Constructor for creating a source from a memory buffer
   *
   * @param[in] bufferpath Memory buffer to use
   * @param[in] length Length in bytes of the buffer
   **/
  explicit DataSource(const char *bufferpath, size_t length) {
    rd_file = std::make_shared<arrow::io::BufferReader>(
        reinterpret_cast<const uint8_t *>(bufferpath), length);
  }

  /**
   * @brief Constructor for creating a source from an existing Arrow file
   *
   * @param[in] file Arrow file instance
   **/
  explicit DataSource(std::shared_ptr<arrow::io::RandomAccessFile> file)
      : rd_file(file) {}

  /**
   * @brief Returns a buffer with a subset of data from the source
   *
   * @param[in] position Starting offset
   * @param[in] length Number of bytes to read
   *
   * @return std::shared_ptr<arrow::Buffer The data buffer
   **/
  const std::shared_ptr<arrow::Buffer> get_buffer(size_t position,
                                                  size_t length) const {
    std::shared_ptr<arrow::Buffer> out;
    CUDF_EXPECTS(rd_file->ReadAt(position, length, &out).ok(),
                 "Cannot read file data");
    return out;
  }

  /**
   * @brief Returns the size of the data in the source
   *
   * @return size_t The size of the source data in bytes
   **/
  size_t size() const {
    int64_t size;
    CUDF_EXPECTS(rd_file->GetSize(&size).ok(), "Cannot get file size");
    return size;
  }

 private:
  std::shared_ptr<arrow::io::RandomAccessFile> rd_file;
};
