/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <src/io/parquet/compact_protocol_reader.hpp>
#include <src/io/parquet/parquet.hpp>
#include <src/io/parquet/parquet_gpu.hpp>

#include <random>
#include <type_traits>

template <typename T, typename SourceElementT = T>
using column_wrapper =
  typename std::conditional<std::is_same_v<T, cudf::string_view>,
                            cudf::test::strings_column_wrapper,
                            cudf::test::fixed_width_column_wrapper<T, SourceElementT>>::type;
using column     = cudf::column;
using table      = cudf::table;
using table_view = cudf::table_view;

// Global environment for temporary files
extern cudf::test::TempDirTestEnvironment* const temp_env;

//////////////////////////////////////////////////////////////////////
// Test fixtures

// Base test fixture for tests
struct ParquetWriterTest : public cudf::test::BaseFixture {};

// Base test fixture for tests
struct ParquetReaderTest : public cudf::test::BaseFixture {};

// Base test fixture for "stress" tests
struct ParquetWriterStressTest : public cudf::test::BaseFixture {};

// Typed test fixture for numeric type tests
template <typename T>
struct ParquetWriterNumericTypeTest : public ParquetWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Typed test fixture for comparable type tests
template <typename T>
struct ParquetWriterComparableTypeTest : public ParquetWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Typed test fixture for timestamp type tests
template <typename T>
struct ParquetWriterChronoTypeTest : public ParquetWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Typed test fixture for timestamp type tests
template <typename T>
struct ParquetWriterTimestampTypeTest : public ParquetWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Typed test fixture for all types
template <typename T>
struct ParquetWriterSchemaTest : public ParquetWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

template <typename T>
struct ParquetReaderSourceTest : public ParquetReaderTest {};

template <typename T>
struct ParquetWriterDeltaTest : public ParquetWriterTest {};

// Base test fixture for chunked writer tests
struct ParquetChunkedWriterTest : public cudf::test::BaseFixture {};

// Typed test fixture for numeric type tests
template <typename T>
struct ParquetChunkedWriterNumericTypeTest : public ParquetChunkedWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Base test fixture for size-parameterized tests
class ParquetSizedTest : public ::cudf::test::BaseFixtureWithParam<int> {};

// Base test fixture for V2 header tests
class ParquetV2Test : public ::cudf::test::BaseFixtureWithParam<bool> {};

// Test fixture for metadata tests
struct ParquetMetadataReaderTest : public cudf::test::BaseFixture {
  std::string print(cudf::io::parquet_column_schema schema, int depth = 0)
  {
    std::string child_str;
    for (auto const& child : schema.children()) {
      child_str += print(child, depth + 1);
    }
    return std::string(depth, ' ') + schema.name() + "\n" + child_str;
  }
};

// Test for Types - numeric, chrono, string.
template <typename T>
struct ParquetReaderPredicatePushdownTest : public ParquetReaderTest {};

////////////////////////////////////////////////////////////////////

// Generates a vector of uniform random values of type T
template <typename T>
std::vector<T> random_values(size_t size);

template <typename T>
std::unique_ptr<cudf::table> create_random_fixed_table(cudf::size_type num_columns,
                                                       cudf::size_type num_rows,
                                                       bool include_validity);

template <typename T>
std::unique_ptr<cudf::table> create_compressible_fixed_table(cudf::size_type num_columns,
                                                             cudf::size_type num_rows,
                                                             cudf::size_type period,
                                                             bool include_validity);

// this function replicates the "list_gen" function in
// python/cudf/cudf/tests/test_parquet.py
template <typename T>
std::unique_ptr<cudf::column> make_parquet_list_list_col(
  int skip_rows, int num_rows, int lists_per_row, int list_size, bool include_validity);

// given a datasource pointing to a parquet file, read the footer
// of the file to populate the FileMetaData pointed to by file_meta_data.
// throws cudf::logic_error if the file or metadata is invalid.
void read_footer(std::unique_ptr<cudf::io::datasource> const& source,
                 cudf::io::parquet::detail::FileMetaData* file_meta_data);

// returns the number of bits used for dictionary encoding data at the given page location.
// this assumes the data is uncompressed.
// throws cudf::logic_error if the page_loc data is invalid.
int read_dict_bits(std::unique_ptr<cudf::io::datasource> const& source,
                   cudf::io::parquet::detail::PageLocation const& page_loc);

// read column index from datasource at location indicated by chunk,
// parse and return as a ColumnIndex struct.
// throws cudf::logic_error if the chunk data is invalid.
cudf::io::parquet::detail::ColumnIndex read_column_index(
  std::unique_ptr<cudf::io::datasource> const& source,
  cudf::io::parquet::detail::ColumnChunk const& chunk);

// read offset index from datasource at location indicated by chunk,
// parse and return as an OffsetIndex struct.
// throws cudf::logic_error if the chunk data is invalid.
cudf::io::parquet::detail::OffsetIndex read_offset_index(
  std::unique_ptr<cudf::io::datasource> const& source,
  cudf::io::parquet::detail::ColumnChunk const& chunk);

// Return as a Statistics from the column chunk
cudf::io::parquet::detail::Statistics const& get_statistics(
  cudf::io::parquet::detail::ColumnChunk const& chunk);

// read page header from datasource at location indicated by page_loc,
// parse and return as a PageHeader struct.
// throws cudf::logic_error if the page_loc data is invalid.
cudf::io::parquet::detail::PageHeader read_page_header(
  std::unique_ptr<cudf::io::datasource> const& source,
  cudf::io::parquet::detail::PageLocation const& page_loc);

// make a random validity iterator
inline auto random_validity(std::mt19937& engine)
{
  static std::bernoulli_distribution bn(0.7f);
  return cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(engine); });
}

// make a random list<T> column
template <typename T>
std::unique_ptr<cudf::column> make_parquet_list_col(std::mt19937& engine,
                                                    int num_rows,
                                                    int max_vals_per_row,
                                                    bool include_validity);

// return vector of random strings
std::vector<std::string> string_values(std::mt19937& engine, int num_rows, int max_string_len);

// make a random list<string> column, with random string lengths of 0..max_string_len,
// and up to max_vals_per_row strings in each list.
std::unique_ptr<cudf::column> make_parquet_string_list_col(std::mt19937& engine,
                                                           int num_rows,
                                                           int max_vals_per_row,
                                                           int max_string_len,
                                                           bool include_validity);

template <typename T>
std::pair<cudf::table, std::string> create_parquet_typed_with_stats(std::string const& filename);

int32_t compare_binary(std::vector<uint8_t> const& v1,
                       std::vector<uint8_t> const& v2,
                       cudf::io::parquet::detail::Type ptype,
                       thrust::optional<cudf::io::parquet::detail::ConvertedType> const& ctype);

void expect_compression_stats_empty(std::shared_ptr<cudf::io::writer_compression_statistics> stats);

// =============================================================================
// ---- test data for stats sort order tests
// need at least 3 pages, and min page count is 5000, so need at least 15000 values.
// use 20000 to be safe.
static constexpr int num_ordered_rows            = 20000;
static constexpr int page_size_for_ordered_tests = 5000;

namespace testdata {

// ----- most numerics
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>,
                 cudf::test::fixed_width_column_wrapper<T>>
ascending();

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>,
                 cudf::test::fixed_width_column_wrapper<T>>
descending();

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>,
                 cudf::test::fixed_width_column_wrapper<T>>
unordered();

// ----- bool

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, cudf::test::fixed_width_column_wrapper<bool>> ascending();

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, cudf::test::fixed_width_column_wrapper<bool>>
descending();

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, cudf::test::fixed_width_column_wrapper<bool>> unordered();

// ----- fixed point types

template <typename T>
std::enable_if_t<cudf::is_fixed_point<T>(), cudf::test::fixed_width_column_wrapper<T>> ascending();

template <typename T>
std::enable_if_t<cudf::is_fixed_point<T>(), cudf::test::fixed_width_column_wrapper<T>> descending();

template <typename T>
std::enable_if_t<cudf::is_fixed_point<T>(), cudf::test::fixed_width_column_wrapper<T>> unordered();

// ----- chrono types
// ----- timstamp

template <typename T>
std::enable_if_t<cudf::is_timestamp<T>(), cudf::test::fixed_width_column_wrapper<T>> ascending();

template <typename T>
std::enable_if_t<cudf::is_timestamp<T>(), cudf::test::fixed_width_column_wrapper<T>> descending();

template <typename T>
std::enable_if_t<cudf::is_timestamp<T>(), cudf::test::fixed_width_column_wrapper<T>> unordered();

// ----- duration

template <typename T>
std::enable_if_t<cudf::is_duration<T>(), cudf::test::fixed_width_column_wrapper<T>> ascending();

template <typename T>
std::enable_if_t<cudf::is_duration<T>(), cudf::test::fixed_width_column_wrapper<T>> descending();

template <typename T>
std::enable_if_t<cudf::is_duration<T>(), cudf::test::fixed_width_column_wrapper<T>> unordered();

// ----- string_view

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, cudf::test::strings_column_wrapper>
ascending();

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, cudf::test::strings_column_wrapper>
descending();

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, cudf::test::strings_column_wrapper>
unordered();

}  // namespace testdata
