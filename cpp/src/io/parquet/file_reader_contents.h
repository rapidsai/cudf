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

class FileReaderContents : public ::parquet::ParquetFileReader::Contents {
public:
    FileReaderContents(std::unique_ptr< ::parquet::RandomAccessSource> source,
                       const ::parquet::ReaderProperties &properties =
                         ::parquet::default_reader_properties());

    ~FileReaderContents() final;
    void                                        Close() final;
    std::shared_ptr< ::parquet::RowGroupReader> GetRowGroup(int i) final;
    std::shared_ptr< ::parquet::FileMetaData>   metadata() const final;

    void ParseMetaData();

private:
    std::unique_ptr< ::parquet::RandomAccessSource> source_;
    std::shared_ptr< ::parquet::FileMetaData>       file_metadata_;
    ::parquet::ReaderProperties                     properties_;

    const int64_t  DEFAULT_FOOTER_READ_SIZE = 64 * 1024;
    const uint32_t FOOTER_SIZE              = 8;
    const uint8_t  PARQUET_MAGIC[4]         = {'P', 'A', 'R', '1'};
};

}  // namespace internal
}  // namespace parquet
}  // namespace gdf
