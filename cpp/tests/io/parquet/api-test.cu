/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
 *     Copyright 2018 William Malpica <william@blazingdb.com>
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

#include <cudf/io_functions.hpp>

#include <arrow/io/file.h>
#include <arrow/util/logging.h>

#include <boost/filesystem.hpp>

#include <parquet/column_writer.h>
#include <parquet/file_writer.h>
#include <parquet/properties.h>
#include <parquet/schema.h>
#include <parquet/types.h>

#include <gtest/gtest.h>

#include "utils.cuh"

// NOTE c.gonzales percy para el parquet file reader aqui los detalles del file 
//file:reader-test.parquet
//Total RowCount: 50000

/* NOTE Meta info for file:reader-test.parquet
file:          file:reader-test.parquet 
creator:       parquet-cpp version 1.4.0 

file schema:   schema 
--------------------------------------------------------------------------------
boolean_field: REQUIRED BOOLEAN R:0 D:0
int64_field:   REQUIRED INT64 R:0 D:0
double_field:  REQUIRED DOUBLE R:0 D:0

row group 1:   RC:50000 TS:767525 OFFSET:4 
--------------------------------------------------------------------------------
boolean_field:  BOOLEAN SNAPPY DO:0 FPO:4 SZ:329/6281/19.09 VC:50000 ENC:PLAIN,RLE
int64_field:    INT64 SNAPPY DO:393 FPO:354318 SZ:454082/500168/1.10 VC:50000 ENC:PLAIN,RLE,PLAIN_DICTIONARY
double_field:   DOUBLE SNAPPY DO:454570 FPO:667527 SZ:313114/500168/1.60 VC:50000 ENC:PLAIN,RLE,PLAIN_DICTIONARY
*/

/* NOTE Schema for file:reader-test.parquet
message schema {
  required boolean boolean_field;
  required int64 int64_field;
  required double double_field;
}

*/

class ParquetReaderAPITest : public testing::Test {
protected:
    ParquetReaderAPITest()
      : filename(boost::filesystem::unique_path().native()) {}

    std::int32_t
    genInt32(int i) {
        if (i >= 100 && i < 150) {
            return 10000;
        } else if (i >= 200 && i < 300) {
            return 20000;
        } else if (i >= 310 && i < 350) {
            return 30000;
        } else if (i >= 450 && i < 550) {
            return 40000;
        } else if (i >= 800 && i < 950) {
            return 50000;
        } else {
            return i * 100;
        }
    }

    std::int64_t
    genInt64(int i) {
        if (i >= 100 && i < 150) {
            return 10000;
        } else if (i >= 200 && i < 300) {
            return 20000;
        } else if (i >= 310 && i < 350) {
            return 30000;
        } else if (i >= 450 && i < 550) {
            return 40000;
        } else if (i >= 800 && i < 950) {
            return 50000;
        } else {
            return i * 100000;
        }
    }

    void
    SetUp() final {
        static constexpr std::size_t kGroups       = 3;
        static constexpr std::size_t kRowsPerGroup = 499;
        try {

            std::shared_ptr<::arrow::io::FileOutputStream> stream;
            PARQUET_THROW_NOT_OK(
              ::arrow::io::FileOutputStream::Open(filename, &stream));

            std::shared_ptr<::parquet::schema::GroupNode> schema =
              CreateSchema();

            ::parquet::WriterProperties::Builder builder;
            builder.compression(::parquet::Compression::SNAPPY);
            std::shared_ptr<::parquet::WriterProperties> properties =
              builder.build();

            std::shared_ptr<::parquet::ParquetFileWriter> file_writer =
              ::parquet::ParquetFileWriter::Open(stream, schema, properties);

            std::int16_t repetition_level = 0;

            for (std::size_t i = 0; i < kGroups; i++) {
                ::parquet::RowGroupWriter *row_group_writer =
                  file_writer->AppendRowGroup(kRowsPerGroup);

                ::parquet::BoolWriter *bool_writer =
                  static_cast<::parquet::BoolWriter *>(
                    row_group_writer->NextColumn());
                for (std::size_t j = 0; j < kRowsPerGroup; j++) {
                    int          ind              = i * kRowsPerGroup + j;
                    std::int16_t definition_level = ind % 3 > 0 ? 1 : 0;
                    bool         bool_value       = true;
                    bool_writer->WriteBatch(
                      1, &definition_level, &repetition_level, &bool_value);
                }

                ::parquet::Int32Writer *int32_writer =
                  static_cast<::parquet::Int32Writer *>(
                    row_group_writer->NextColumn());
                for (std::size_t j = 0; j < kRowsPerGroup; j++) {
                    int          ind              = i * kRowsPerGroup + j;
                    std::int16_t definition_level = ind % 3 > 0 ? 1 : 0;
                    std::int32_t int32_value      = genInt32(ind);
                    int32_writer->WriteBatch(
                      1, &definition_level, &repetition_level, &int32_value);
                }

                ::parquet::Int64Writer *int64_writer =
                  static_cast<::parquet::Int64Writer *>(
                    row_group_writer->NextColumn());
                for (std::size_t j = 0; j < kRowsPerGroup; j++) {
                    int          ind              = i * kRowsPerGroup + j;
                    std::int16_t definition_level = ind % 3 > 0 ? 1 : 0;
                    std::int64_t int64_value      = genInt64(ind);
                    int64_writer->WriteBatch(
                      1, &definition_level, &repetition_level, &int64_value);
                }

                ::parquet::DoubleWriter *double_writer =
                  static_cast<::parquet::DoubleWriter *>(
                    row_group_writer->NextColumn());
                for (std::size_t j = 0; j < kRowsPerGroup; j++) {
                    int          ind              = i * kRowsPerGroup + j;
                    std::int16_t definition_level = ind % 3 > 0 ? 1 : 0;
                    double       double_value     = (double) ind;
                    double_writer->WriteBatch(
                      1, &definition_level, &repetition_level, &double_value);
                }
            }

            file_writer->Close();

            DCHECK(stream->Close().ok());
        } catch (const std::exception &e) {
            FAIL() << "Generate file" << e.what();
        }
    }

    std ::shared_ptr<::parquet::schema::GroupNode>
    CreateSchema() {
        return std::static_pointer_cast<::parquet::schema::GroupNode>(
          ::parquet::schema::GroupNode::Make(
            "schema",
            ::parquet::Repetition::REQUIRED,
            ::parquet::schema::NodeVector{
              ::parquet::schema::PrimitiveNode::Make(
                "boolean_field",
                ::parquet::Repetition::OPTIONAL,
                ::parquet::Type::BOOLEAN,
                ::parquet::LogicalType::NONE),
              ::parquet::schema::PrimitiveNode::Make(
                "int32_field",
                ::parquet::Repetition::OPTIONAL,
                ::parquet::Type::INT32,
                ::parquet::LogicalType::NONE),
              ::parquet::schema::PrimitiveNode::Make(
                "int64_field",
                ::parquet::Repetition::OPTIONAL,
                ::parquet::Type::INT64,
                ::parquet::LogicalType::NONE),
              ::parquet::schema::PrimitiveNode::Make(
                "double_field",
                ::parquet::Repetition::OPTIONAL,
                ::parquet::Type::DOUBLE,
                ::parquet::LogicalType::NONE),
            }));
    }

    void
    TearDown() final {
        if (std::remove(filename.c_str())) { FAIL() << "Remove file"; }
    }

    void
    checkNulls(/*const */ gdf_column &column) {

        const std::size_t valid_size =
          arrow::BitUtil::BytesForBits(column.size);
        const std::size_t valid_last = valid_size - 1;

        int fails = 0;
        for (std::size_t i = 0; i < valid_last; i++) {

            if (i % 3 == 0) {
                std::uint8_t valid    = column.valid[i];
                std::uint8_t expected = 0b10110110;
                EXPECT_EQ(expected, valid);
                if (expected != valid) {
                    std::cout << "fail at checkNulls i: " << i << std::endl;
                    fails++;
                    if (fails > 5) break;
                }
            } else if (i % 3 == 1) {
                std::uint8_t valid    = column.valid[i];
                std::uint8_t expected = 0b01101101;
                EXPECT_EQ(expected, valid);
                if (expected != valid) {
                    std::cout << "fail at checkNulls i: " << i << std::endl;
                    fails++;
                    if (fails > 5) break;
                }
            } else {
                std::uint8_t valid    = column.valid[i];
                std::uint8_t expected = 0b11011011;
                EXPECT_EQ(expected, valid);
                if (expected != valid) {
                    std::cout << "fail at checkNulls i: " << i << std::endl;
                    fails++;
                    if (fails > 5) break;
                }
            }
        }
        //        EXPECT_EQ(0b00101101, 0b00101101 & column.valid[valid_last]);
    }

    void
    checkBoolean(/*const */ gdf_column &column) {

        gdf_column boolean_column =
          convert_to_host_gdf_column<::parquet::BooleanType::c_type>(&column);

        int fails = 0;

        for (gdf_size_type i = 0; i < boolean_column.size; i++) {
            if (i % 3 > 0) {
                bool expected = true;
                bool value    = static_cast<bool *>(boolean_column.data)[i];

                EXPECT_EQ(expected, value);

                if (expected != value) {
                    std::cout << "fail at checkBoolean row: " << i << std::endl;
                    fails++;
                    if (fails > 5) { break; }
                }
            }
        }
        checkNulls(boolean_column);
    }

    void
    checkInt32(/*const */ gdf_column &column) {

        gdf_column int32_column =
          convert_to_host_gdf_column<::parquet::Int32Type::c_type>(&column);

        int fails = 0;

        for (gdf_size_type i = 0; i < int32_column.size; i++) {
            if (i % 3 > 0) {
                std::int32_t expected = genInt32(i);
                std::int32_t value =
                  static_cast<std::int32_t *>(int32_column.data)[i];

                EXPECT_EQ(expected, value);

                if (expected != value) {
                    std::cout << "fail at checkInt32 row: " << i << std::endl;
                    fails++;
                    if (fails > 5) { break; }
                }
            }
        }

        checkNulls(int32_column);
    }

    void
    checkInt64(/*const */ gdf_column &column) {
        gdf_column int64_column =
          convert_to_host_gdf_column<::parquet::Int64Type::c_type>(&column);

        int fails = 0;

        for (gdf_size_type i = 0; i < int64_column.size; i++) {
            if (i % 3 > 0) {
                std::int64_t expected = genInt64(i);
                std::int64_t value =
                  static_cast<std::int64_t *>(int64_column.data)[i];

                EXPECT_EQ(expected, value);

                if (expected != value) {
                    std::cout << "fail at checkInt64 row: " << i << std::endl;
                    fails++;
                    if (fails > 5) { break; }
                }
            }
        }

        checkNulls(int64_column);
    }

    void
    checkDouble(/*const */ gdf_column &column) {
        gdf_column double_column =
          convert_to_host_gdf_column<::parquet::DoubleType::c_type>(&column);

        int fails = 0;

        for (gdf_size_type i = 0; i < double_column.size; i++) {
            if (i % 3 > 0) {
                double expected = static_cast<double>(i);
                double value    = static_cast<double *>(double_column.data)[i];

                EXPECT_EQ(expected, value);

                if (expected != value) {
                    std::cout << "fail at checkDouble row: " << i << std::endl;
                    fails++;
                    if (fails > 5) { break; }
                }
            }
        }

        checkNulls(double_column);
    }

    const std::string filename;

    gdf_column *columns        = nullptr;
    std::size_t columns_length = 0;
};

TEST_F(ParquetReaderAPITest, ReadAll) {

    gdf_error error_code = gdf::parquet::read_parquet(
      filename.c_str(), nullptr, &columns, &columns_length);

    EXPECT_EQ(GDF_SUCCESS, error_code);

    EXPECT_EQ(4U, columns_length);

    EXPECT_EQ(columns[0].size, columns[1].size);
    EXPECT_EQ(columns[1].size, columns[2].size);

    checkBoolean(columns[0]);
    checkInt32(columns[1]);
    checkInt64(columns[2]);
    checkDouble(columns[3]);
}

TEST_F(ParquetReaderAPITest, ReadSomeColumns) {
    const char *const column_names[] = {"double_field", "int64_field", nullptr};

    gdf_error error_code = gdf::parquet::read_parquet(
      filename.c_str(), column_names, &columns, &columns_length);

    EXPECT_EQ(GDF_SUCCESS, error_code);

    EXPECT_EQ(2U, columns_length);

    checkDouble(columns[0]);
    checkInt64(columns[1]);
}

TEST_F(ParquetReaderAPITest, ByIdsInOrder) {
    const std::vector<std::size_t> row_group_indices = {0, 1};
    const std::vector<std::size_t> column_indices    = {0, 1, 2, 3};

    std::vector<gdf_column *> columns;

    gdf_error error_code = gdf::parquet::read_parquet_by_ids(
      filename, row_group_indices, column_indices, columns);

    EXPECT_EQ(GDF_SUCCESS, error_code);

    EXPECT_EQ(4U, columns.size());

    checkBoolean(*columns[0]);
    checkInt32(*columns[1]);
    checkInt64(*columns[2]);
    checkDouble(*columns[3]);
}

TEST_F(ParquetReaderAPITest, ByIdsOutOfOrder) {
    const std::vector<std::size_t> row_group_indices = {0, 1};
    const std::vector<std::size_t> column_indices    = {1, 3, 2, 0};

    std::vector<gdf_column *> columns;

    gdf_error error_code = gdf::parquet::read_parquet_by_ids(
      filename, row_group_indices, column_indices, columns);

    EXPECT_EQ(GDF_SUCCESS, error_code);

    EXPECT_EQ(4U, columns.size());

    checkBoolean(*columns[3]);
    checkInt32(*columns[0]);
    checkInt64(*columns[2]);
    checkDouble(*columns[1]);
}

TEST_F(ParquetReaderAPITest, ByIdsInFromInterface) {
    const std::vector<std::size_t> row_group_indices = {0, 1};
    const std::vector<std::size_t> column_indices    = {0, 1, 2, 3};

    std::vector<gdf_column *> columns;

    std::shared_ptr<::arrow::io::ReadableFile> file;
    const ::parquet::ReaderProperties          properties =
      ::parquet::default_reader_properties();
    ::arrow::io::ReadableFile::Open(filename, properties.memory_pool(), &file);

    gdf_error error_code = gdf::parquet::read_parquet_by_ids(
      file, row_group_indices, column_indices, columns);

    EXPECT_EQ(GDF_SUCCESS, error_code);

    EXPECT_EQ(4U, columns.size());

    checkBoolean(*columns[0]);
    checkInt32(*columns[1]);
    checkInt64(*columns[2]);
    checkDouble(*columns[3]);
}
