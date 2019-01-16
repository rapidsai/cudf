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

#include <arrow/io/file.h>
#include <arrow/util/logging.h>

#include "column_reader.h"
#include "file_reader.h"
#include "file_reader_contents.h"

namespace gdf {
namespace parquet {

// ----------------------------------------------------------------------
// RowGroupReader public API

GdfRowGroupReader::GdfRowGroupReader(std::unique_ptr<::parquet::RowGroupReader::Contents> contents)
        : ::parquet::RowGroupReader(nullptr), contents_(std::move(contents)) {}


static std::shared_ptr<::parquet::ColumnReader> GdfColumnReaderMake(const ::parquet::ColumnDescriptor* descr,
                                                 std::unique_ptr<::parquet::PageReader> pager,
                                                             ::arrow::MemoryPool* pool) {
    switch (descr->physical_type()) {
        case ::parquet::Type::BOOLEAN:
            return std::static_pointer_cast<::parquet::ColumnReader>(std::make_shared<BoolReader>(descr, std::move(pager), pool));
        case ::parquet::Type::INT32:
            return std::static_pointer_cast<::parquet::ColumnReader>(std::make_shared<Int32Reader>(descr, std::move(pager), pool));
            break;
        case ::parquet::Type::INT64:
            return std::static_pointer_cast<::parquet::ColumnReader>(std::make_shared<Int64Reader>(descr, std::move(pager), pool));
        case ::parquet::Type::FLOAT:
            return std::static_pointer_cast<::parquet::ColumnReader>(std::make_shared<FloatReader>(descr, std::move(pager), pool));
        case ::parquet::Type::DOUBLE:
            return std::static_pointer_cast<::parquet::ColumnReader>(std::make_shared<DoubleReader>(descr, std::move(pager), pool));
        default:
            ::parquet::ParquetException::NYI("type reader not implemented");
    }
    // Unreachable code, but supress compiler warning
    return std::shared_ptr<::parquet::ColumnReader>(nullptr);
}


std::shared_ptr<::parquet::ColumnReader> GdfRowGroupReader::Column(int i) {
    DCHECK(i < metadata()->num_columns()) << "The RowGroup only has "
                                          << metadata()->num_columns()
                                          << "columns, requested column: " << i;
    const ::parquet::ColumnDescriptor* descr = metadata()->schema()->Column(i);

    std::unique_ptr<::parquet::PageReader> page_reader = contents_->GetColumnPageReader(i);
    return GdfColumnReaderMake(
            descr, std::move(page_reader),
            const_cast<::parquet::ReaderProperties*>(contents_->properties())->memory_pool());
}



std::unique_ptr<::parquet::PageReader> GdfRowGroupReader::GetColumnPageReader(int i) {
    DCHECK(i < metadata()->num_columns()) << "The RowGroup only has "
                                          << metadata()->num_columns()
                                          << "columns, requested column: " << i;
    return contents_->GetColumnPageReader(i);
}

// Returns the rowgroup metadata
const ::parquet::RowGroupMetaData* GdfRowGroupReader::metadata() const { return contents_->metadata(); }

// ----------------------------------------------------------------------

std::unique_ptr<FileReader>
FileReader::OpenFile(const std::string &                path,
                     const ::parquet::ReaderProperties &properties) {

	FileReader *const reader = new FileReader();
	reader->parquetFileReader_.reset(new ::parquet::ParquetFileReader());

	std::shared_ptr<::arrow::io::ReadableFile> file;

	PARQUET_THROW_NOT_OK(
			::arrow::io::ReadableFile::Open(path, properties.memory_pool(), &file));

	std::unique_ptr<::parquet::RandomAccessSource> source(
			new ::parquet::ArrowInputFile(file));

	std::unique_ptr<::parquet::ParquetFileReader::Contents> contents(
			new internal::FileReaderContents(std::move(source), properties));

	static_cast<internal::FileReaderContents *>(contents.get())
	    		   ->ParseMetaData();

	reader->parquetFileReader_->Open(std::move(contents));

	return std::unique_ptr<FileReader>(reader);
}

std::unique_ptr<FileReader>
FileReader::OpenFile(std::shared_ptr<::arrow::io::RandomAccessFile> file,
		const ::parquet::ReaderProperties &properties) {

	FileReader *const reader = new FileReader();
	reader->parquetFileReader_.reset(new ::parquet::ParquetFileReader());

	std::unique_ptr<::parquet::RandomAccessSource> source(
			new ::parquet::ArrowInputFile(file));

	std::unique_ptr<::parquet::ParquetFileReader::Contents> contents(
			new internal::FileReaderContents(std::move(source), properties));

	static_cast<internal::FileReaderContents *>(contents.get())
		    				   ->ParseMetaData();

	reader->parquetFileReader_->Open(std::move(contents));


	return std::unique_ptr<FileReader>(reader);
}

std::shared_ptr<GdfRowGroupReader>
FileReader::RowGroup(int i) {
    return std::static_pointer_cast< GdfRowGroupReader >(parquetFileReader_->RowGroup(i));
}

std::shared_ptr<::parquet::FileMetaData>
FileReader::metadata() const {
    return parquetFileReader_->metadata();
}

}  // namespace parquet
}  // namespace gdf
