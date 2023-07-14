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

#include <arrow/buffer.h>

// We disable warning 611 because some Arrow subclasses of
// `arrow::fs::FileSystem` only partially override the `Equals` method,
// triggering warning 611-D from nvcc.
#ifdef __CUDACC__
#pragma nv_diag_suppress 611
#endif
#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/s3fs.h>
#ifdef __CUDACC__
#pragma nv_diag_default 611
#endif

// We disable warning 2810 to workaround the compile issue (warning treated as error):
// result.h(263): error #2810-D: ignoring return value type with "nodiscard" attribute
#ifdef __CUDACC__
#pragma nv_diag_suppress 2810
#endif
#include <arrow/result.h>
#ifdef __CUDACC__
#pragma nv_diag_default 2810
#endif

#include <arrow/io/file.h>
#include <arrow/io/interfaces.h>
#include <arrow/io/memory.h>
#include <arrow/status.h>

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
    std::string const uri_start_delimiter = "//";
    std::string const uri_end_delimiter   = "?";

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

/** @} */  // end of group
}  // namespace cudf::io
