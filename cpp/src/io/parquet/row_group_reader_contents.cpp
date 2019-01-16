/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
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

#include "row_group_reader_contents.h"

#include "page_reader.h"

namespace gdf {
namespace parquet {
namespace internal {

RowGroupReaderContents::RowGroupReaderContents(
  ::parquet::RandomAccessSource *    source,
  ::parquet::FileMetaData *          file_metadata,
  int                                row_group_number,
  const ::parquet::ReaderProperties &props)
  : source_(source), file_metadata_(file_metadata), properties_(props) {
    row_group_metadata_ = file_metadata->RowGroup(row_group_number);
}

const ::parquet::RowGroupMetaData *
RowGroupReaderContents::metadata() const {
    return row_group_metadata_.get();
}

const ::parquet::ReaderProperties *
RowGroupReaderContents::properties() const {
    return &properties_;
}

std::unique_ptr<::parquet::PageReader>
RowGroupReaderContents::GetColumnPageReader(int i) {
    auto col = row_group_metadata_->ColumnChunk(i);

    int64_t col_start = col->data_page_offset();
    if (col->has_dictionary_page()
        && col_start > col->dictionary_page_offset()) {
        col_start = col->dictionary_page_offset();
    }

    std::int64_t col_length = col->total_compressed_size();
    std::unique_ptr<::parquet::InputStream> stream;

    const ::parquet::ApplicationVersion &version =
      file_metadata_->writer_version();
    if (version.VersionLt(
          ::parquet::ApplicationVersion::PARQUET_816_FIXED_VERSION())) {
        std::int64_t bytes_remaining =
          source_->Size() - (col_start + col_length);
        std::int64_t padding =
          std::min<std::int64_t>(kMaxDictHeaderSize, bytes_remaining);
        col_length += padding;
    }

    stream = properties_.GetStream(source_, col_start, col_length);

    return std::unique_ptr<::parquet::PageReader>(
      new internal::PageReader(std::move(stream),
                               col->num_values(),
                               col->compression(),
                               properties_.memory_pool()));
}

}  // namespace internal
}  // namespace parquet
}  // namespace gdf
