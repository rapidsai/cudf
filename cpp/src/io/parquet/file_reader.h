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

#ifndef _GDF_PARQUET_FILE_READER_H
#define _GDF_PARQUET_FILE_READER_H

#include <arrow/io/file.h>
#include <parquet/file_reader.h>

namespace gdf {
namespace parquet {

class GdfRowGroupReader : public ::parquet::RowGroupReader {
public:
    explicit GdfRowGroupReader(std::unique_ptr<Contents> contents);

    // Returns the rowgroup metadata
    const ::parquet::RowGroupMetaData *metadata() const;

    // Construct a ColumnReader for the indicated row group-relative
    // column. Ownership is shared with the RowGroupReader.
    std::shared_ptr< ::parquet::ColumnReader> Column(int i);

    std::unique_ptr< ::parquet::PageReader> GetColumnPageReader(int i);

private:
    // Holds a pointer to an instance of Contents implementation
    std::unique_ptr<Contents> contents_;
};

class FileReader {
public:
    static std::unique_ptr<FileReader>
    OpenFile(const std::string &                path,
             const ::parquet::ReaderProperties &properties =
               ::parquet::default_reader_properties());

    static std::unique_ptr<FileReader>
    OpenFile(std::shared_ptr< ::arrow::io::RandomAccessFile> file,
             const ::parquet::ReaderProperties &             properties =
               ::parquet::default_reader_properties());

    std::shared_ptr<GdfRowGroupReader>        RowGroup(int i);
    std::shared_ptr< ::parquet::FileMetaData> metadata() const;

private:
    std::unique_ptr< ::parquet::ParquetFileReader> parquetFileReader_;
};

}  // namespace parquet
}  // namespace gdf

#endif
