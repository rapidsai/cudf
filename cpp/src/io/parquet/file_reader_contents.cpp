/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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

#include "file_reader_contents.h"

#include "row_group_reader_contents.h"

#include "file_reader.h"

namespace gdf {
namespace parquet {
namespace internal {

FileReaderContents::FileReaderContents(
  std::unique_ptr<::parquet::RandomAccessSource> source,
  const ::parquet::ReaderProperties &            properties)
  : source_(std::move(source)), properties_(properties) {}

FileReaderContents::~FileReaderContents() {
    try {
        Close();
    } catch (...) {}
}

void
FileReaderContents::Close() {
    source_->Close();
}

std::shared_ptr<::parquet::RowGroupReader>
FileReaderContents::GetRowGroup(int i) {
    std::unique_ptr<internal::RowGroupReaderContents> contents(
      new internal::RowGroupReaderContents(
        source_.get(), file_metadata_.get(), i, properties_));
    return std::static_pointer_cast<::parquet::RowGroupReader>(
      std::make_shared<GdfRowGroupReader>(std::move(contents)));
}

std::shared_ptr<::parquet::FileMetaData>
FileReaderContents::metadata() const {
    return file_metadata_;
}

void
FileReaderContents::ParseMetaData() {
    std::int64_t file_size = source_->Size();

    if (file_size < FOOTER_SIZE) {
        throw ::parquet::ParquetException(
          "Corrupted file, smaller than file footer");
    }

    std::uint8_t footer_buffer[DEFAULT_FOOTER_READ_SIZE];
    std::int64_t footer_read_size =
      std::min(file_size, DEFAULT_FOOTER_READ_SIZE);
    std::int64_t bytes_read = source_->ReadAt(
      file_size - footer_read_size, footer_read_size, footer_buffer);

    if (bytes_read != footer_read_size
        || std::memcmp(footer_buffer + footer_read_size - 4, PARQUET_MAGIC, 4)
             != 0) {
        throw ::parquet::ParquetException(
          "Invalid parquet file. Corrupt footer.");
    }

    std::uint32_t metadata_len = *reinterpret_cast<std::uint32_t *>(
      footer_buffer + footer_read_size - FOOTER_SIZE);
    std::int64_t metadata_start = file_size - FOOTER_SIZE - metadata_len;
    if (FOOTER_SIZE + metadata_len > file_size) {
        throw ::parquet::ParquetException(
          "Invalid parquet file. File is less than "
          "file metadata size.");
    }

    std::shared_ptr<::parquet::ResizableBuffer> metadata_buffer =
      ::parquet::AllocateBuffer(properties_.memory_pool(), metadata_len);

    if (footer_read_size >= (metadata_len + FOOTER_SIZE)) {
        std::memcpy(metadata_buffer->mutable_data(),
                    footer_buffer
                      + (footer_read_size - metadata_len - FOOTER_SIZE),
                    metadata_len);
    } else {
        bytes_read = source_->ReadAt(
          metadata_start, metadata_len, metadata_buffer->mutable_data());
        if (bytes_read != metadata_len) {
            throw ::parquet::ParquetException(
              "Invalid parquet file. Could not read metadata bytes.");
        }
    }

    file_metadata_ =
      ::parquet::FileMetaData::Make(metadata_buffer->data(), &metadata_len);
}

}  // namespace internal
}  // namespace parquet
}  // namespace gdf
