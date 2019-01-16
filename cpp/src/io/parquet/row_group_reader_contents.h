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

#include <parquet/file_reader.h>

namespace gdf {
namespace parquet {
namespace internal {

class RowGroupReaderContents : public ::parquet::RowGroupReader::Contents {
public:
    RowGroupReaderContents(::parquet::RandomAccessSource *    source,
                           ::parquet::FileMetaData *          file_metadata,
                           int                                row_group_number,
                           const ::parquet::ReaderProperties &props);

    const ::parquet::RowGroupMetaData *metadata() const final;
    const ::parquet::ReaderProperties *properties() const final;
    virtual std::unique_ptr< ::parquet::PageReader>
    GetColumnPageReader(int i) final;

private:
    ::parquet::RandomAccessSource *               source_;
    ::parquet::FileMetaData *                     file_metadata_;
    std::unique_ptr< ::parquet::RowGroupMetaData> row_group_metadata_;
    ::parquet::ReaderProperties                   properties_;

    const std::int64_t kMaxDictHeaderSize = 100;
};

}  // namespace internal
}  // namespace parquet
}  // namespace gdf
