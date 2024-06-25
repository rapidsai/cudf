/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/io/arrow_io_source.hpp>

#include <arrow/buffer.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/result.h>

#include <memory>
#include <string>
#include <utility>

namespace cudf::io {

/**
 * @brief Implementation for an owning buffer where `arrow::Buffer` holds the data.
 */
class arrow_io_buffer : public datasource::buffer {
  std::shared_ptr<arrow::Buffer> arrow_buffer;

 public:
  explicit arrow_io_buffer(std::shared_ptr<arrow::Buffer> arrow_buffer)
    : arrow_buffer(std::move(arrow_buffer))
  {
  }
  [[nodiscard]] size_t size() const override { return arrow_buffer->size(); }
  [[nodiscard]] uint8_t const* data() const override { return arrow_buffer->data(); }
};

arrow_io_source::arrow_io_source(std::string const& arrow_uri)
{
  std::string const uri_start_delimiter = "//";
  std::string const uri_end_delimiter   = "?";

  auto const result = arrow::fs::FileSystemFromUri(arrow_uri);
  CUDF_EXPECTS(result.ok(), "Failed to generate Arrow Filesystem instance from URI.");
  filesystem = result.ValueOrDie();

  // Parse the path from the URI
  auto const start = [&]() {
    auto const delim_start = arrow_uri.find(uri_start_delimiter);
    return delim_start == std::string::npos ? 0 : delim_start + uri_start_delimiter.size();
  }();
  auto const end  = arrow_uri.find(uri_end_delimiter) - start;
  auto const path = arrow_uri.substr(start, end);

  auto const in_stream = filesystem->OpenInputFile(path);
  CUDF_EXPECTS(in_stream.ok(), "Failed to open Arrow RandomAccessFile");
  arrow_file = in_stream.ValueOrDie();
}

std::unique_ptr<datasource::buffer> arrow_io_source::host_read(size_t offset, size_t size)
{
  auto const result = arrow_file->ReadAt(offset, size);
  CUDF_EXPECTS(result.ok(), "Cannot read file data");
  return std::make_unique<arrow_io_buffer>(result.ValueOrDie());
}

size_t arrow_io_source::host_read(size_t offset, size_t size, uint8_t* dst)
{
  auto const result = arrow_file->ReadAt(offset, size, dst);
  CUDF_EXPECTS(result.ok(), "Cannot read file data");
  return result.ValueOrDie();
}

[[nodiscard]] size_t arrow_io_source::size() const
{
  auto const result = arrow_file->GetSize();
  CUDF_EXPECTS(result.ok(), "Cannot get file size");
  return result.ValueOrDie();
}

}  // namespace cudf::io
