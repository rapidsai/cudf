/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <src/io/parquet/compact_protocol_reader.hpp>
#include <src/io/parquet/parquet.hpp>
#include <src/io/parquet/parquet_gpu.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <fstream>
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
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

template <typename T, typename Elements>
std::unique_ptr<cudf::table> create_fixed_table(cudf::size_type num_columns,
                                                cudf::size_type num_rows,
                                                bool include_validity,
                                                Elements elements)
{
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });
  std::vector<cudf::test::fixed_width_column_wrapper<T>> src_cols(num_columns);
  for (int idx = 0; idx < num_columns; idx++) {
    if (include_validity) {
      src_cols[idx] =
        cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_rows, valids);
    } else {
      src_cols[idx] = cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_rows);
    }
  }
  std::vector<std::unique_ptr<cudf::column>> columns(num_columns);
  std::transform(src_cols.begin(),
                 src_cols.end(),
                 columns.begin(),
                 [](cudf::test::fixed_width_column_wrapper<T>& in) {
                   auto ret = in.release();
                   // pre-cache the null count
                   [[maybe_unused]] auto const nulls = ret->has_nulls();
                   return ret;
                 });
  return std::make_unique<cudf::table>(std::move(columns));
}

template <typename T>
std::unique_ptr<cudf::table> create_random_fixed_table(cudf::size_type num_columns,
                                                       cudf::size_type num_rows,
                                                       bool include_validity)
{
  auto rand_elements =
    cudf::detail::make_counting_transform_iterator(0, [](T i) { return rand(); });
  return create_fixed_table<T>(num_columns, num_rows, include_validity, rand_elements);
}

template <typename T>
std::unique_ptr<cudf::table> create_compressible_fixed_table(cudf::size_type num_columns,
                                                             cudf::size_type num_rows,
                                                             cudf::size_type period,
                                                             bool include_validity)
{
  auto compressible_elements =
    cudf::detail::make_counting_transform_iterator(0, [period](T i) { return i / period; });
  return create_fixed_table<T>(num_columns, num_rows, include_validity, compressible_elements);
}

// this function replicates the "list_gen" function in
// python/cudf/cudf/tests/test_parquet.py
template <typename T>
std::unique_ptr<cudf::column> make_parquet_list_list_col(
  int skip_rows, int num_rows, int lists_per_row, int list_size, bool include_validity)
{
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0 ? 1 : 0; });

  // root list
  std::vector<int> row_offsets(num_rows + 1);
  int row_offset_count = 0;
  {
    int offset = 0;
    for (int idx = 0; idx < (num_rows) + 1; idx++) {
      row_offsets[row_offset_count] = offset;
      if (!include_validity || valids[idx]) { offset += lists_per_row; }
      row_offset_count++;
    }
  }
  cudf::test::fixed_width_column_wrapper<int> offsets(row_offsets.begin(),
                                                      row_offsets.begin() + row_offset_count);

  // child list
  std::vector<int> child_row_offsets((num_rows * lists_per_row) + 1);
  int child_row_offset_count = 0;
  {
    int offset = 0;
    for (int idx = 0; idx < (num_rows * lists_per_row); idx++) {
      int row_index = idx / lists_per_row;
      if (include_validity && !valids[row_index]) { continue; }

      child_row_offsets[child_row_offset_count] = offset;
      offset += list_size;
      child_row_offset_count++;
    }
    child_row_offsets[child_row_offset_count++] = offset;
  }
  cudf::test::fixed_width_column_wrapper<int> child_offsets(
    child_row_offsets.begin(), child_row_offsets.begin() + child_row_offset_count);

  // child values
  std::vector<T> child_values(num_rows * lists_per_row * list_size);
  T first_child_value_index = skip_rows * lists_per_row * list_size;
  int child_value_count     = 0;
  {
    for (int idx = 0; idx < (num_rows * lists_per_row * list_size); idx++) {
      int row_index = idx / (lists_per_row * list_size);

      int val = first_child_value_index;
      first_child_value_index++;

      if (include_validity && !valids[row_index]) { continue; }

      child_values[child_value_count] = val;
      child_value_count++;
    }
  }
  // validity by value instead of index
  auto valids2 = cudf::detail::make_counting_transform_iterator(
    0, [list_size](auto i) { return (i % list_size) % 2 == 0 ? 1 : 0; });
  auto child_data = include_validity
                      ? cudf::test::fixed_width_column_wrapper<T>(
                          child_values.begin(), child_values.begin() + child_value_count, valids2)
                      : cudf::test::fixed_width_column_wrapper<T>(
                          child_values.begin(), child_values.begin() + child_value_count);

  int child_offsets_size = static_cast<cudf::column_view>(child_offsets).size() - 1;
  auto child             = cudf::make_lists_column(
    child_offsets_size, child_offsets.release(), child_data.release(), 0, rmm::device_buffer{});

  int offsets_size             = static_cast<cudf::column_view>(offsets).size() - 1;
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids, valids + offsets_size);
  return include_validity
           ? cudf::make_lists_column(
               offsets_size, offsets.release(), std::move(child), null_count, std::move(null_mask))
           : cudf::make_lists_column(
               offsets_size, offsets.release(), std::move(child), 0, rmm::device_buffer{});
}

// given a datasource pointing to a parquet file, read the footer
// of the file to populate the FileMetaData pointed to by file_meta_data.
// throws cudf::logic_error if the file or metadata is invalid.
void read_footer(std::unique_ptr<cudf::io::datasource> const& source,
                 cudf::io::parquet::FileMetaData* file_meta_data)
{
  constexpr auto header_len = sizeof(cudf::io::parquet::file_header_s);
  constexpr auto ender_len  = sizeof(cudf::io::parquet::file_ender_s);

  auto const len           = source->size();
  auto const header_buffer = source->host_read(0, header_len);
  auto const header =
    reinterpret_cast<cudf::io::parquet::file_header_s const*>(header_buffer->data());
  auto const ender_buffer = source->host_read(len - ender_len, ender_len);
  auto const ender = reinterpret_cast<cudf::io::parquet::file_ender_s const*>(ender_buffer->data());

  // checks for valid header, footer, and file length
  ASSERT_GT(len, header_len + ender_len);
  ASSERT_TRUE(header->magic == cudf::io::parquet::parquet_magic &&
              ender->magic == cudf::io::parquet::parquet_magic);
  ASSERT_TRUE(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len));

  // parquet files end with 4-byte footer_length and 4-byte magic == "PAR1"
  // seek backwards from the end of the file (footer_length + 8 bytes of ender)
  auto const footer_buffer =
    source->host_read(len - ender->footer_len - ender_len, ender->footer_len);
  cudf::io::parquet::CompactProtocolReader cp(footer_buffer->data(), ender->footer_len);

  // returns true on success
  bool res = cp.read(file_meta_data);
  ASSERT_TRUE(res);
}

// returns the number of bits used for dictionary encoding data at the given page location.
// this assumes the data is uncompressed.
// throws cudf::logic_error if the page_loc data is invalid.
int read_dict_bits(std::unique_ptr<cudf::io::datasource> const& source,
                   cudf::io::parquet::PageLocation const& page_loc)
{
  CUDF_EXPECTS(page_loc.offset > 0, "Cannot find page header");
  CUDF_EXPECTS(page_loc.compressed_page_size > 0, "Invalid page header length");

  cudf::io::parquet::PageHeader page_hdr;
  auto const page_buf = source->host_read(page_loc.offset, page_loc.compressed_page_size);
  cudf::io::parquet::CompactProtocolReader cp(page_buf->data(), page_buf->size());
  bool res = cp.read(&page_hdr);
  CUDF_EXPECTS(res, "Cannot parse page header");

  // cp should be pointing at the start of page data now. the first byte
  // should be the encoding bit size
  return cp.getb();
}

// read column index from datasource at location indicated by chunk,
// parse and return as a ColumnIndex struct.
// throws cudf::logic_error if the chunk data is invalid.
cudf::io::parquet::ColumnIndex read_column_index(
  std::unique_ptr<cudf::io::datasource> const& source, cudf::io::parquet::ColumnChunk const& chunk)
{
  CUDF_EXPECTS(chunk.column_index_offset > 0, "Cannot find column index");
  CUDF_EXPECTS(chunk.column_index_length > 0, "Invalid column index length");

  cudf::io::parquet::ColumnIndex colidx;
  auto const ci_buf = source->host_read(chunk.column_index_offset, chunk.column_index_length);
  cudf::io::parquet::CompactProtocolReader cp(ci_buf->data(), ci_buf->size());
  bool res = cp.read(&colidx);
  CUDF_EXPECTS(res, "Cannot parse column index");
  return colidx;
}

// read offset index from datasource at location indicated by chunk,
// parse and return as an OffsetIndex struct.
// throws cudf::logic_error if the chunk data is invalid.
cudf::io::parquet::OffsetIndex read_offset_index(
  std::unique_ptr<cudf::io::datasource> const& source, cudf::io::parquet::ColumnChunk const& chunk)
{
  CUDF_EXPECTS(chunk.offset_index_offset > 0, "Cannot find offset index");
  CUDF_EXPECTS(chunk.offset_index_length > 0, "Invalid offset index length");

  cudf::io::parquet::OffsetIndex offidx;
  auto const oi_buf = source->host_read(chunk.offset_index_offset, chunk.offset_index_length);
  cudf::io::parquet::CompactProtocolReader cp(oi_buf->data(), oi_buf->size());
  bool res = cp.read(&offidx);
  CUDF_EXPECTS(res, "Cannot parse offset index");
  return offidx;
}

// Return as a Statistics from the column chunk
cudf::io::parquet::Statistics const& get_statistics(cudf::io::parquet::ColumnChunk const& chunk)
{
  return chunk.meta_data.statistics;
}

// read page header from datasource at location indicated by page_loc,
// parse and return as a PageHeader struct.
// throws cudf::logic_error if the page_loc data is invalid.
cudf::io::parquet::PageHeader read_page_header(std::unique_ptr<cudf::io::datasource> const& source,
                                               cudf::io::parquet::PageLocation const& page_loc)
{
  CUDF_EXPECTS(page_loc.offset > 0, "Cannot find page header");
  CUDF_EXPECTS(page_loc.compressed_page_size > 0, "Invalid page header length");

  cudf::io::parquet::PageHeader page_hdr;
  auto const page_buf = source->host_read(page_loc.offset, page_loc.compressed_page_size);
  cudf::io::parquet::CompactProtocolReader cp(page_buf->data(), page_buf->size());
  bool res = cp.read(&page_hdr);
  CUDF_EXPECTS(res, "Cannot parse page header");
  return page_hdr;
}

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

// Declare typed test cases
// TODO: Replace with `NumericTypes` when unsigned support is added. Issue #5352
using SupportedTypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, bool, float, double>;
TYPED_TEST_SUITE(ParquetWriterNumericTypeTest, SupportedTypes);
using ComparableAndFixedTypes =
  cudf::test::Concat<cudf::test::ComparableTypes, cudf::test::FixedPointTypes>;
TYPED_TEST_SUITE(ParquetWriterComparableTypeTest, ComparableAndFixedTypes);
TYPED_TEST_SUITE(ParquetWriterChronoTypeTest, cudf::test::ChronoTypes);
using SupportedTimestampTypes =
  cudf::test::Types<cudf::timestamp_ms, cudf::timestamp_us, cudf::timestamp_ns>;
TYPED_TEST_SUITE(ParquetWriterTimestampTypeTest, SupportedTimestampTypes);
TYPED_TEST_SUITE(ParquetWriterSchemaTest, cudf::test::AllTypes);
using ByteLikeTypes = cudf::test::Types<int8_t, char, uint8_t, unsigned char, std::byte>;
TYPED_TEST_SUITE(ParquetReaderSourceTest, ByteLikeTypes);

// Base test fixture for chunked writer tests
struct ParquetChunkedWriterTest : public cudf::test::BaseFixture {};

// Typed test fixture for numeric type tests
template <typename T>
struct ParquetChunkedWriterNumericTypeTest : public ParquetChunkedWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Declare typed test cases
TYPED_TEST_SUITE(ParquetChunkedWriterNumericTypeTest, SupportedTypes);

// Base test fixture for size-parameterized tests
class ParquetSizedTest : public ::cudf::test::BaseFixtureWithParam<int> {};

// test the allowed bit widths for dictionary encoding
// values chosen to trigger 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, and 24 bit dictionaries
INSTANTIATE_TEST_SUITE_P(ParquetDictionaryTest,
                         ParquetSizedTest,
                         testing::Range(1, 25),
                         testing::PrintToStringParamName());

// Base test fixture for V2 header tests
class ParquetV2Test : public ::cudf::test::BaseFixtureWithParam<bool> {};
INSTANTIATE_TEST_SUITE_P(ParquetV2ReadWriteTest,
                         ParquetV2Test,
                         testing::Bool(),
                         testing::PrintToStringParamName());

namespace {
// Generates a vector of uniform random values of type T
template <typename T>
inline auto random_values(size_t size)
{
  std::vector<T> values(size);

  using T1 = T;
  using uniform_distribution =
    typename std::conditional_t<std::is_same_v<T1, bool>,
                                std::bernoulli_distribution,
                                std::conditional_t<std::is_floating_point_v<T1>,
                                                   std::uniform_real_distribution<T1>,
                                                   std::uniform_int_distribution<T1>>>;

  static constexpr auto seed = 0xf00d;
  static std::mt19937 engine{seed};
  static uniform_distribution dist{};
  std::generate_n(values.begin(), size, [&]() { return T{dist(engine)}; });

  return values;
}

}  // namespace

TYPED_TEST(ParquetWriterNumericTypeTest, SingleColumn)
{
  auto sequence =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i % 400); });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  constexpr auto num_rows = 800;
  column_wrapper<TypeParam> col(sequence, sequence + num_rows, validity);

  auto expected = table_view{{col}};

  auto filepath = temp_env->get_temp_filepath("SingleColumn.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(ParquetWriterNumericTypeTest, SingleColumnWithNulls)
{
  auto sequence =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i); });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i % 2); });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam> col(sequence, sequence + num_rows, validity);

  auto expected = table_view{{col}};

  auto filepath = temp_env->get_temp_filepath("SingleColumnWithNulls.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

template <typename mask_op_t>
void test_durations(mask_op_t mask_op)
{
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution_d(0, 30);
  auto sequence_d = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) { return distribution_d(generator); });

  std::uniform_int_distribution<int> distribution_s(0, 86400);
  auto sequence_s = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) { return distribution_s(generator); });

  std::uniform_int_distribution<int> distribution(0, 86400 * 1000);
  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) { return distribution(generator); });

  auto mask = cudf::detail::make_counting_transform_iterator(0, mask_op);

  constexpr auto num_rows = 100;
  // Durations longer than a day are not exactly valid, but cudf should be able to round trip
  auto durations_d = cudf::test::fixed_width_column_wrapper<cudf::duration_D, int64_t>(
    sequence_d, sequence_d + num_rows, mask);
  auto durations_s = cudf::test::fixed_width_column_wrapper<cudf::duration_s, int64_t>(
    sequence_s, sequence_s + num_rows, mask);
  auto durations_ms = cudf::test::fixed_width_column_wrapper<cudf::duration_ms, int64_t>(
    sequence, sequence + num_rows, mask);
  auto durations_us = cudf::test::fixed_width_column_wrapper<cudf::duration_us, int64_t>(
    sequence, sequence + num_rows, mask);
  auto durations_ns = cudf::test::fixed_width_column_wrapper<cudf::duration_ns, int64_t>(
    sequence, sequence + num_rows, mask);

  auto expected = table_view{{durations_d, durations_s, durations_ms, durations_us, durations_ns}};

  auto filepath = temp_env->get_temp_filepath("Durations.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  auto durations_d_got =
    cudf::cast(result.tbl->view().column(0), cudf::data_type{cudf::type_id::DURATION_DAYS});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(durations_d, durations_d_got->view());

  auto durations_s_got =
    cudf::cast(result.tbl->view().column(1), cudf::data_type{cudf::type_id::DURATION_SECONDS});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(durations_s, durations_s_got->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(durations_ms, result.tbl->view().column(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(durations_us, result.tbl->view().column(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(durations_ns, result.tbl->view().column(4));
}

TEST_F(ParquetWriterTest, Durations)
{
  test_durations([](auto i) { return true; });
  test_durations([](auto i) { return (i % 2) != 0; });
  test_durations([](auto i) { return (i % 3) != 0; });
  test_durations([](auto i) { return false; });
}

TYPED_TEST(ParquetWriterTimestampTypeTest, Timestamps)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return ((std::rand() / 10000) * 1000); });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + num_rows, validity);

  auto expected = table_view{{col}};

  auto filepath = temp_env->get_temp_filepath("Timestamps.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .timestamp_type(this->type());
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(ParquetWriterTimestampTypeTest, TimestampsWithNulls)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return ((std::rand() / 10000) * 1000); });
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i > 30) && (i < 60); });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + num_rows, validity);

  auto expected = table_view{{col}};

  auto filepath = temp_env->get_temp_filepath("TimestampsWithNulls.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .timestamp_type(this->type());
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(ParquetWriterTimestampTypeTest, TimestampOverflow)
{
  constexpr int64_t max = std::numeric_limits<int64_t>::max();
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return max - i; });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + num_rows, validity);
  table_view expected({col});

  auto filepath = temp_env->get_temp_filepath("ParquetTimestampOverflow.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .timestamp_type(this->type());
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_P(ParquetV2Test, MultiColumn)
{
  constexpr auto num_rows = 100000;
  auto const is_v2        = GetParam();

  // auto col0_data = random_values<bool>(num_rows);
  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<float>(num_rows);
  auto col5_data = random_values<double>(num_rows);
  auto col6_vals = random_values<int16_t>(num_rows);
  auto col7_vals = random_values<int32_t>(num_rows);
  auto col8_vals = random_values<int64_t>(num_rows);
  auto col6_data = cudf::detail::make_counting_transform_iterator(0, [col6_vals](auto i) {
    return numeric::decimal32{col6_vals[i], numeric::scale_type{5}};
  });
  auto col7_data = cudf::detail::make_counting_transform_iterator(0, [col7_vals](auto i) {
    return numeric::decimal64{col7_vals[i], numeric::scale_type{-5}};
  });
  auto col8_data = cudf::detail::make_counting_transform_iterator(0, [col8_vals](auto i) {
    return numeric::decimal128{col8_vals[i], numeric::scale_type{-6}};
  });
  auto validity  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  // column_wrapper<bool> col0{
  //    col0_data.begin(), col0_data.end(), validity};
  column_wrapper<int8_t> col1{col1_data.begin(), col1_data.end(), validity};
  column_wrapper<int16_t> col2{col2_data.begin(), col2_data.end(), validity};
  column_wrapper<int32_t> col3{col3_data.begin(), col3_data.end(), validity};
  column_wrapper<float> col4{col4_data.begin(), col4_data.end(), validity};
  column_wrapper<double> col5{col5_data.begin(), col5_data.end(), validity};
  column_wrapper<numeric::decimal32> col6{col6_data, col6_data + num_rows, validity};
  column_wrapper<numeric::decimal64> col7{col7_data, col7_data + num_rows, validity};
  column_wrapper<numeric::decimal128> col8{col8_data, col8_data + num_rows, validity};

  auto expected = table_view{{col1, col2, col3, col4, col5, col6, col7, col8}};

  cudf::io::table_input_metadata expected_metadata(expected);
  // expected_metadata.column_metadata[0].set_name( "bools");
  expected_metadata.column_metadata[0].set_name("int8s");
  expected_metadata.column_metadata[1].set_name("int16s");
  expected_metadata.column_metadata[2].set_name("int32s");
  expected_metadata.column_metadata[3].set_name("floats");
  expected_metadata.column_metadata[4].set_name("doubles");
  expected_metadata.column_metadata[5].set_name("decimal32s").set_decimal_precision(10);
  expected_metadata.column_metadata[6].set_name("decimal64s").set_decimal_precision(20);
  expected_metadata.column_metadata[7].set_name("decimal128s").set_decimal_precision(40);

  auto filepath = temp_env->get_temp_filepath("MultiColumn.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(is_v2)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, MultiColumnWithNulls)
{
  constexpr auto num_rows = 100;
  auto const is_v2        = GetParam();

  // auto col0_data = random_values<bool>(num_rows);
  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<float>(num_rows);
  auto col5_data = random_values<double>(num_rows);
  auto col6_vals = random_values<int32_t>(num_rows);
  auto col7_vals = random_values<int64_t>(num_rows);
  auto col6_data = cudf::detail::make_counting_transform_iterator(0, [col6_vals](auto i) {
    return numeric::decimal32{col6_vals[i], numeric::scale_type{-2}};
  });
  auto col7_data = cudf::detail::make_counting_transform_iterator(0, [col7_vals](auto i) {
    return numeric::decimal64{col7_vals[i], numeric::scale_type{-8}};
  });
  // auto col0_mask = cudf::detail::make_counting_transform_iterator(
  //    0, [](auto i) { return (i % 2); });
  auto col1_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i < 10); });
  auto col2_mask = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });
  auto col3_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i == (num_rows - 1)); });
  auto col4_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i >= 40 && i <= 60); });
  auto col5_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i > 80); });
  auto col6_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i % 5); });
  auto col7_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i != 55); });

  // column_wrapper<bool> col0{
  //    col0_data.begin(), col0_data.end(), col0_mask};
  column_wrapper<int8_t> col1{col1_data.begin(), col1_data.end(), col1_mask};
  column_wrapper<int16_t> col2{col2_data.begin(), col2_data.end(), col2_mask};
  column_wrapper<int32_t> col3{col3_data.begin(), col3_data.end(), col3_mask};
  column_wrapper<float> col4{col4_data.begin(), col4_data.end(), col4_mask};
  column_wrapper<double> col5{col5_data.begin(), col5_data.end(), col5_mask};
  column_wrapper<numeric::decimal32> col6{col6_data, col6_data + num_rows, col6_mask};
  column_wrapper<numeric::decimal64> col7{col7_data, col7_data + num_rows, col7_mask};

  auto expected = table_view{{/*col0, */ col1, col2, col3, col4, col5, col6, col7}};

  cudf::io::table_input_metadata expected_metadata(expected);
  // expected_metadata.column_names.emplace_back("bools");
  expected_metadata.column_metadata[0].set_name("int8s");
  expected_metadata.column_metadata[1].set_name("int16s");
  expected_metadata.column_metadata[2].set_name("int32s");
  expected_metadata.column_metadata[3].set_name("floats");
  expected_metadata.column_metadata[4].set_name("doubles");
  expected_metadata.column_metadata[5].set_name("decimal32s").set_decimal_precision(9);
  expected_metadata.column_metadata[6].set_name("decimal64s").set_decimal_precision(20);

  auto filepath = temp_env->get_temp_filepath("MultiColumnWithNulls.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(is_v2)
      .metadata(expected_metadata);

  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  // TODO: Need to be able to return metadata in tree form from reader so they can be compared.
  // Unfortunately the closest thing to a hierarchical schema is column_name_info which does not
  // have any tests for it c++ or python.
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, Strings)
{
  auto const is_v2 = GetParam();

  std::vector<char const*> strings{
    "Monday", "Wȅdnȅsday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  auto const num_rows = strings.size();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  column_wrapper<int> col0{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<cudf::string_view> col1{strings.begin(), strings.end()};
  column_wrapper<float> col2{seq_col2.begin(), seq_col2.end(), validity};

  auto expected = table_view{{col0, col1, col2}};

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_other");
  expected_metadata.column_metadata[1].set_name("col_string");
  expected_metadata.column_metadata[2].set_name("col_another");

  auto filepath = temp_env->get_temp_filepath("Strings.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(is_v2)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(ParquetWriterTest, StringsAsBinary)
{
  std::vector<char const*> unicode_strings{
    "Monday", "Wȅdnȅsday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  std::vector<char const*> ascii_strings{
    "Monday", "Wednesday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};

  column_wrapper<cudf::string_view> col0{ascii_strings.begin(), ascii_strings.end()};
  column_wrapper<cudf::string_view> col1{unicode_strings.begin(), unicode_strings.end()};
  column_wrapper<cudf::string_view> col2{ascii_strings.begin(), ascii_strings.end()};
  cudf::test::lists_column_wrapper<uint8_t> col3{{'M', 'o', 'n', 'd', 'a', 'y'},
                                                 {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
                                                 {'F', 'r', 'i', 'd', 'a', 'y'},
                                                 {'M', 'o', 'n', 'd', 'a', 'y'},
                                                 {'F', 'r', 'i', 'd', 'a', 'y'},
                                                 {'F', 'r', 'i', 'd', 'a', 'y'},
                                                 {'F', 'r', 'i', 'd', 'a', 'y'},
                                                 {'F', 'u', 'n', 'd', 'a', 'y'}};
  cudf::test::lists_column_wrapper<uint8_t> col4{
    {'M', 'o', 'n', 'd', 'a', 'y'},
    {'W', 200, 133, 'd', 'n', 200, 133, 's', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'M', 'o', 'n', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'u', 'n', 'd', 'a', 'y'}};

  auto write_tbl = table_view{{col0, col1, col2, col3, col4}};

  cudf::io::table_input_metadata expected_metadata(write_tbl);
  expected_metadata.column_metadata[0].set_name("col_single").set_output_as_binary(true);
  expected_metadata.column_metadata[1].set_name("col_string").set_output_as_binary(true);
  expected_metadata.column_metadata[2].set_name("col_another").set_output_as_binary(true);
  expected_metadata.column_metadata[3].set_name("col_binary");
  expected_metadata.column_metadata[4].set_name("col_binary");

  auto filepath = temp_env->get_temp_filepath("BinaryStrings.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, write_tbl)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema(
        {cudf::io::reader_column_schema().set_convert_binary_to_strings(false),
         cudf::io::reader_column_schema().set_convert_binary_to_strings(false),
         cudf::io::reader_column_schema().set_convert_binary_to_strings(false),
         cudf::io::reader_column_schema().add_child(cudf::io::reader_column_schema()),
         cudf::io::reader_column_schema().add_child(cudf::io::reader_column_schema())});
  auto result   = cudf::io::read_parquet(in_opts);
  auto expected = table_view{{col3, col4, col3, col3, col4}};

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, SlicedTable)
{
  // This test checks for writing zero copy, offsetted views into existing cudf tables

  std::vector<char const*> strings{
    "Monday", "Wȅdnȅsday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  auto const num_rows = strings.size();
  auto const is_v2    = GetParam();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 3 != 0; });

  column_wrapper<int> col0{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<cudf::string_view> col1{strings.begin(), strings.end()};
  column_wrapper<float> col2{seq_col2.begin(), seq_col2.end(), validity};

  using lcw = cudf::test::lists_column_wrapper<uint64_t>;
  lcw col3{{9, 8}, {7, 6, 5}, {}, {4}, {3, 2, 1, 0}, {20, 21, 22, 23, 24}, {}, {66, 666}};

  // [[[NULL,2,NULL,4]], [[NULL,6,NULL], [8,9]]]
  // [NULL, [[13],[14,15,16]],  NULL]
  // [NULL, [], NULL, [[]]]
  // NULL
  // [[[NULL,2,NULL,4]], [[NULL,6,NULL], [8,9]]]
  // [NULL, [[13],[14,15,16]],  NULL]
  // [[[]]]
  // [NULL, [], NULL, [[]]]
  auto valids  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; });
  lcw col4{{
             {{{{1, 2, 3, 4}, valids}}, {{{5, 6, 7}, valids}, {8, 9}}},
             {{{{10, 11}, {12}}, {{13}, {14, 15, 16}}, {{17, 18}}}, valids},
             {{lcw{lcw{}}, lcw{}, lcw{}, lcw{lcw{}}}, valids},
             lcw{lcw{lcw{}}},
             {{{{1, 2, 3, 4}, valids}}, {{{5, 6, 7}, valids}, {8, 9}}},
             {{{{10, 11}, {12}}, {{13}, {14, 15, 16}}, {{17, 18}}}, valids},
             lcw{lcw{lcw{}}},
             {{lcw{lcw{}}, lcw{}, lcw{}, lcw{lcw{}}}, valids},
           },
           valids2};

  // Struct column
  auto ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
    {48, 27, 25, 31, 351, 351, 29, 15}, {1, 1, 1, 1, 1, 0, 1, 1}};

  auto col5 = cudf::test::structs_column_wrapper{{ages_col}, {1, 1, 1, 1, 0, 1, 1, 1}};

  // Struct/List mixed column

  // []
  // [NULL, 2, NULL]
  // [4, 5]
  // NULL
  // []
  // [7, 8, 9]
  // [10]
  // [11, 12]
  lcw land{{{}, {{1, 2, 3}, valids}, {4, 5}, {}, {}, {7, 8, 9}, {10}, {11, 12}}, valids2};

  // []
  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8], []]
  // [[]]
  // [[]]
  // [[], [], []]
  // [[10]]
  // [[13, 14], [15]]
  lcw flats{lcw{},
            {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
            {{7, 8}, {}},
            lcw{lcw{}},
            lcw{lcw{}},
            lcw{lcw{}, lcw{}, lcw{}},
            {lcw{10}},
            {{13, 14}, {15}}};

  auto struct_1 = cudf::test::structs_column_wrapper{land, flats};
  auto is_human = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, true, false, true, false}};
  auto col6 = cudf::test::structs_column_wrapper{{is_human, struct_1}};

  auto expected = table_view({col0, col1, col2, col3, col4, col5, col6});

  // auto expected_slice = expected;
  auto expected_slice = cudf::slice(expected, {2, static_cast<cudf::size_type>(num_rows) - 1});

  cudf::io::table_input_metadata expected_metadata(expected_slice);
  expected_metadata.column_metadata[0].set_name("col_other");
  expected_metadata.column_metadata[1].set_name("col_string");
  expected_metadata.column_metadata[2].set_name("col_another");
  expected_metadata.column_metadata[3].set_name("col_list");
  expected_metadata.column_metadata[4].set_name("col_multi_level_list");
  expected_metadata.column_metadata[5].set_name("col_struct");
  expected_metadata.column_metadata[5].set_name("col_struct_list");
  expected_metadata.column_metadata[6].child(0).set_name("human?");
  expected_metadata.column_metadata[6].child(1).set_name("particulars");
  expected_metadata.column_metadata[6].child(1).child(0).set_name("land");
  expected_metadata.column_metadata[6].child(1).child(1).set_name("flats");

  auto filepath = temp_env->get_temp_filepath("SlicedTable.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected_slice)
      .write_v2_headers(is_v2)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_slice, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, ListColumn)
{
  auto const is_v2 = GetParam();

  auto valids  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // [NULL, 2, NULL]
  // []
  // [4, 5]
  // NULL
  lcw col0{{{{1, 2, 3}, valids}, {}, {4, 5}, {}}, valids2};

  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8]]
  // []
  // [[]]
  lcw col1{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, {{7, 8}}, lcw{}, lcw{lcw{}}};

  // [[1, 2, 3], [], [4, 5], NULL, [0, 6, 0]]
  // [[7, 8]]
  // []
  // [[]]
  lcw col2{{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, valids2}, {{7, 8}}, lcw{}, lcw{lcw{}}};

  // [[1, 2, 3], [], [4, 5], NULL, [NULL, 6, NULL]]
  // [[7, 8]]
  // []
  // [[]]
  using dlcw = cudf::test::lists_column_wrapper<double>;
  dlcw col3{{{{1., 2., 3.}, {}, {4., 5.}, {}, {{0., 6., 0.}, valids}}, valids2},
            {{7., 8.}},
            dlcw{},
            dlcw{dlcw{}}};

  // TODO: uint16_t lists are not read properly in parquet reader
  // [[1, 2, 3], [], [4, 5], NULL, [0, 6, 0]]
  // [[7, 8]]
  // []
  // NULL
  // using ui16lcw = cudf::test::lists_column_wrapper<uint16_t>;
  // cudf::test::lists_column_wrapper<uint16_t> col4{
  //   {{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, valids2}, {{7, 8}}, ui16lcw{}, ui16lcw{ui16lcw{}}},
  //   valids2};

  // [[1, 2, 3], [], [4, 5], NULL, [NULL, 6, NULL]]
  // [[7, 8]]
  // []
  // NULL
  lcw col5{
    {{{{1, 2, 3}, {}, {4, 5}, {}, {{0, 6, 0}, valids}}, valids2}, {{7, 8}}, lcw{}, lcw{lcw{}}},
    valids2};

  using strlcw = cudf::test::lists_column_wrapper<cudf::string_view>;
  cudf::test::lists_column_wrapper<cudf::string_view> col6{
    {{"Monday", "Monday", "Friday"}, {}, {"Monday", "Friday"}, {}, {"Sunday", "Funday"}},
    {{"bee", "sting"}},
    strlcw{},
    strlcw{strlcw{}}};

  // [[[NULL,2,NULL,4]], [[NULL,6,NULL], [8,9]]]
  // [NULL, [[13],[14,15,16]],  NULL]
  // [NULL, [], NULL, [[]]]
  // NULL
  lcw col7{{
             {{{{1, 2, 3, 4}, valids}}, {{{5, 6, 7}, valids}, {8, 9}}},
             {{{{10, 11}, {12}}, {{13}, {14, 15, 16}}, {{17, 18}}}, valids},
             {{lcw{lcw{}}, lcw{}, lcw{}, lcw{lcw{}}}, valids},
             lcw{lcw{lcw{}}},
           },
           valids2};

  table_view expected({col0, col1, col2, col3, /* col4, */ col5, col6, col7});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_list_int_0");
  expected_metadata.column_metadata[1].set_name("col_list_list_int_1");
  expected_metadata.column_metadata[2].set_name("col_list_list_int_nullable_2");
  expected_metadata.column_metadata[3].set_name("col_list_list_nullable_double_nullable_3");
  // expected_metadata.column_metadata[0].set_name("col_list_list_uint16_4");
  expected_metadata.column_metadata[4].set_name("col_list_nullable_list_nullable_int_nullable_5");
  expected_metadata.column_metadata[5].set_name("col_list_list_string_6");
  expected_metadata.column_metadata[6].set_name("col_list_list_list_7");

  auto filepath = temp_env->get_temp_filepath("ListColumn.parquet");
  auto out_opts = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
                    .write_v2_headers(is_v2)
                    .metadata(expected_metadata)
                    .compression(cudf::io::compression_type::NONE);

  cudf::io::write_parquet(out_opts);

  auto in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result  = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(ParquetWriterTest, MultiIndex)
{
  constexpr auto num_rows = 100;

  auto col0_data = random_values<int8_t>(num_rows);
  auto col1_data = random_values<int16_t>(num_rows);
  auto col2_data = random_values<int32_t>(num_rows);
  auto col3_data = random_values<float>(num_rows);
  auto col4_data = random_values<double>(num_rows);
  auto validity  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  column_wrapper<int8_t> col0{col0_data.begin(), col0_data.end(), validity};
  column_wrapper<int16_t> col1{col1_data.begin(), col1_data.end(), validity};
  column_wrapper<int32_t> col2{col2_data.begin(), col2_data.end(), validity};
  column_wrapper<float> col3{col3_data.begin(), col3_data.end(), validity};
  column_wrapper<double> col4{col4_data.begin(), col4_data.end(), validity};

  auto expected = table_view{{col0, col1, col2, col3, col4}};

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("int8s");
  expected_metadata.column_metadata[1].set_name("int16s");
  expected_metadata.column_metadata[2].set_name("int32s");
  expected_metadata.column_metadata[3].set_name("floats");
  expected_metadata.column_metadata[4].set_name("doubles");

  auto filepath = temp_env->get_temp_filepath("MultiIndex.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(expected_metadata)
      .key_value_metadata(
        {{{"pandas", "\"index_columns\": [\"int8s\", \"int16s\"], \"column1\": [\"int32s\"]"}}});
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .use_pandas_metadata(true)
      .columns({"int32s", "floats", "doubles"});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(ParquetWriterTest, BufferSource)
{
  constexpr auto num_rows = 100 << 10;
  auto const seq_col      = random_values<int>(num_rows);
  auto const validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });
  column_wrapper<int> col{seq_col.begin(), seq_col.end(), validity};

  auto const expected = table_view{{col}};

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_other");

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer), expected)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  // host buffer
  {
    cudf::io::parquet_reader_options in_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info(out_buffer.data(), out_buffer.size()));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }

  // device buffer
  {
    auto const d_input = cudf::detail::make_device_uvector_sync(
      cudf::host_span<uint8_t const>{reinterpret_cast<uint8_t const*>(out_buffer.data()),
                                     out_buffer.size()},
      cudf::get_default_stream(),
      rmm::mr::get_current_device_resource());
    auto const d_buffer = cudf::device_span<std::byte const>(
      reinterpret_cast<std::byte const*>(d_input.data()), d_input.size());
    cudf::io::parquet_reader_options in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(d_buffer));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }
}

TEST_F(ParquetWriterTest, ManyFragments)
{
  srand(31337);
  auto const expected = create_random_fixed_table<int>(10, 6'000'000, false);

  auto const filepath = temp_env->get_temp_filepath("ManyFragments.parquet");
  cudf::io::parquet_writer_options const args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected)
      .max_page_size_bytes(8 * 1024);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options const read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetWriterTest, NonNullable)
{
  srand(31337);
  auto expected = create_random_fixed_table<int>(9, 9, false);

  auto filepath = temp_env->get_temp_filepath("NonNullable.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetWriterTest, Struct)
{
  // Struct<is_human:bool, Struct<names:string, ages:int>>

  auto names = {"Samuel Vimes",
                "Carrot Ironfoundersson",
                "Angua von Uberwald",
                "Cheery Littlebottom",
                "Detritus",
                "Mr Slant"};

  // `Name` column has all valid values.
  auto names_col = cudf::test::strings_column_wrapper{names.begin(), names.end()};

  auto ages_col =
    cudf::test::fixed_width_column_wrapper<int32_t>{{48, 27, 25, 31, 351, 351}, {1, 1, 1, 1, 1, 0}};

  auto struct_1 = cudf::test::structs_column_wrapper{{names_col, ages_col}, {1, 1, 1, 1, 0, 1}};

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {1, 1, 0, 1, 1, 0}};

  auto struct_2 =
    cudf::test::structs_column_wrapper{{is_human_col, struct_1}, {0, 1, 1, 1, 1, 1}}.release();

  auto expected = table_view({*struct_2});

  auto filepath = temp_env->get_temp_filepath("Struct.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath));
  cudf::io::read_parquet(read_args);
}

TEST_P(ParquetV2Test, StructOfList)
{
  auto const is_v2 = GetParam();

  // Struct<is_human:bool,
  //        Struct<weight:float,
  //               ages:int,
  //               land_unit:List<int>>,
  //               flats:List<List<int>>
  //              >
  //       >

  auto weights_col = cudf::test::fixed_width_column_wrapper<float>{1.1, 2.4, 5.3, 8.0, 9.6, 6.9};

  auto ages_col =
    cudf::test::fixed_width_column_wrapper<int32_t>{{48, 27, 25, 31, 351, 351}, {1, 1, 1, 1, 1, 0}};

  auto valids  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // []
  // [NULL, 2, NULL]
  // [4, 5]
  // NULL
  // []
  // [7, 8, 9]
  lcw land_unit{{{}, {{1, 2, 3}, valids}, {4, 5}, {}, {}, {7, 8, 9}}, valids2};

  // []
  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8], []]
  // [[]]
  // [[]]
  // [[], [], []]
  lcw flats{lcw{},
            {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
            {{7, 8}, {}},
            lcw{lcw{}},
            lcw{lcw{}},
            lcw{lcw{}, lcw{}, lcw{}}};

  auto struct_1 = cudf::test::structs_column_wrapper{{weights_col, ages_col, land_unit, flats},
                                                     {1, 1, 1, 1, 0, 1}};

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {1, 1, 0, 1, 1, 0}};

  auto struct_2 =
    cudf::test::structs_column_wrapper{{is_human_col, struct_1}, {0, 1, 1, 1, 1, 1}}.release();

  auto expected = table_view({*struct_2});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("being");
  expected_metadata.column_metadata[0].child(0).set_name("human?");
  expected_metadata.column_metadata[0].child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("age");
  expected_metadata.column_metadata[0].child(1).child(2).set_name("land_unit");
  expected_metadata.column_metadata[0].child(1).child(3).set_name("flats");

  auto filepath = temp_env->get_temp_filepath("StructOfList.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(is_v2)
      .metadata(expected_metadata);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath));
  auto const result = cudf::io::read_parquet(read_args);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, ListOfStruct)
{
  auto const is_v2 = GetParam();

  // List<Struct<is_human:bool,
  //             Struct<weight:float,
  //                    ages:int,
  //                   >
  //            >
  //     >

  auto weight_col = cudf::test::fixed_width_column_wrapper<float>{1.1, 2.4, 5.3, 8.0, 9.6, 6.9};

  auto ages_col =
    cudf::test::fixed_width_column_wrapper<int32_t>{{48, 27, 25, 31, 351, 351}, {1, 1, 1, 1, 1, 0}};

  auto struct_1 = cudf::test::structs_column_wrapper{{weight_col, ages_col}, {1, 1, 1, 1, 0, 1}};

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {1, 1, 0, 1, 1, 0}};

  auto struct_2 =
    cudf::test::structs_column_wrapper{{is_human_col, struct_1}, {0, 1, 1, 1, 1, 1}}.release();

  auto list_offsets_column =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 5, 5, 6}.release();
  auto num_list_rows = list_offsets_column->size() - 1;

  auto list_col = cudf::make_lists_column(
    num_list_rows, std::move(list_offsets_column), std::move(struct_2), 0, {});

  auto expected = table_view({*list_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("family");
  expected_metadata.column_metadata[0].child(1).child(0).set_name("human?");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).child(1).set_name("age");

  auto filepath = temp_env->get_temp_filepath("ListOfStruct.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(is_v2)
      .metadata(expected_metadata);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath));
  auto const result = cudf::io::read_parquet(read_args);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

// custom data sink that supports device writes. uses plain file io.
class custom_test_data_sink : public cudf::io::data_sink {
 public:
  explicit custom_test_data_sink(std::string const& filepath)
  {
    outfile_.open(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
    CUDF_EXPECTS(outfile_.is_open(), "Cannot open output file");
  }

  virtual ~custom_test_data_sink() { flush(); }

  void host_write(void const* data, size_t size) override
  {
    outfile_.write(static_cast<char const*>(data), size);
  }

  [[nodiscard]] bool supports_device_write() const override { return true; }

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    this->device_write_async(gpu_data, size, stream).get();
  }

  std::future<void> device_write_async(void const* gpu_data,
                                       size_t size,
                                       rmm::cuda_stream_view stream) override
  {
    return std::async(std::launch::deferred, [=] {
      char* ptr = nullptr;
      CUDF_CUDA_TRY(cudaMallocHost(&ptr, size));
      CUDF_CUDA_TRY(cudaMemcpyAsync(ptr, gpu_data, size, cudaMemcpyDefault, stream.value()));
      stream.synchronize();
      outfile_.write(ptr, size);
      CUDF_CUDA_TRY(cudaFreeHost(ptr));
    });
  }

  void flush() override { outfile_.flush(); }

  size_t bytes_written() override { return outfile_.tellp(); }

 private:
  std::ofstream outfile_;
};

TEST_F(ParquetWriterTest, CustomDataSink)
{
  auto filepath = temp_env->get_temp_filepath("CustomDataSink.parquet");
  custom_test_data_sink custom_sink(filepath);

  srand(31337);
  auto expected = create_random_fixed_table<int>(5, 10, false);

  // write out using the custom sink
  {
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
    cudf::io::write_parquet(args);
  }

  // write out using a memmapped sink
  std::vector<char> buf_sink;
  {
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buf_sink}, *expected);
    cudf::io::write_parquet(args);
  }

  // read them back in and make sure everything matches

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());

  cudf::io::parquet_reader_options buf_args = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{buf_sink.data(), buf_sink.size()});
  auto buf_tbl = cudf::io::read_parquet(buf_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(buf_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterTest, DeviceWriteLargeishFile)
{
  auto filepath = temp_env->get_temp_filepath("DeviceWriteLargeishFile.parquet");
  custom_test_data_sink custom_sink(filepath);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_random_fixed_table<int>(4, 4 * 1024 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterTest, PartitionedWrite)
{
  auto source = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 1000, false);

  auto filepath1 = temp_env->get_temp_filepath("PartitionedWrite1.parquet");
  auto filepath2 = temp_env->get_temp_filepath("PartitionedWrite2.parquet");

  auto partition1 = cudf::io::partition_info{10, 1024 * 1024};
  auto partition2 = cudf::io::partition_info{20 * 1024 + 7, 3 * 1024 * 1024};

  auto expected1 =
    cudf::slice(*source, {partition1.start_row, partition1.start_row + partition1.num_rows});
  auto expected2 =
    cudf::slice(*source, {partition2.start_row, partition2.start_row + partition2.num_rows});

  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(
      cudf::io::sink_info(std::vector<std::string>{filepath1, filepath2}), *source)
      .partitions({partition1, partition2})
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(args);

  auto result1 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath1)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected1, result1.tbl->view());

  auto result2 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath2)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected2, result2.tbl->view());
}

TEST_P(ParquetV2Test, PartitionedWriteEmptyPartitions)
{
  auto const is_v2 = GetParam();

  auto source = create_random_fixed_table<int>(4, 4, false);

  auto filepath1 = temp_env->get_temp_filepath("PartitionedWrite1.parquet");
  auto filepath2 = temp_env->get_temp_filepath("PartitionedWrite2.parquet");

  auto partition1 = cudf::io::partition_info{1, 0};
  auto partition2 = cudf::io::partition_info{1, 0};

  auto expected1 =
    cudf::slice(*source, {partition1.start_row, partition1.start_row + partition1.num_rows});
  auto expected2 =
    cudf::slice(*source, {partition2.start_row, partition2.start_row + partition2.num_rows});

  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(
      cudf::io::sink_info(std::vector<std::string>{filepath1, filepath2}), *source)
      .partitions({partition1, partition2})
      .write_v2_headers(is_v2)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(args);

  auto result1 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath1)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected1, result1.tbl->view());

  auto result2 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath2)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected2, result2.tbl->view());
}

TEST_P(ParquetV2Test, PartitionedWriteEmptyColumns)
{
  auto const is_v2 = GetParam();

  auto source = create_random_fixed_table<int>(0, 4, false);

  auto filepath1 = temp_env->get_temp_filepath("PartitionedWrite1.parquet");
  auto filepath2 = temp_env->get_temp_filepath("PartitionedWrite2.parquet");

  auto partition1 = cudf::io::partition_info{1, 0};
  auto partition2 = cudf::io::partition_info{1, 0};

  auto expected1 =
    cudf::slice(*source, {partition1.start_row, partition1.start_row + partition1.num_rows});
  auto expected2 =
    cudf::slice(*source, {partition2.start_row, partition2.start_row + partition2.num_rows});

  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(
      cudf::io::sink_info(std::vector<std::string>{filepath1, filepath2}), *source)
      .partitions({partition1, partition2})
      .write_v2_headers(is_v2)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(args);

  auto result1 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath1)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected1, result1.tbl->view());

  auto result2 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath2)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected2, result2.tbl->view());
}

template <typename T>
std::string create_parquet_file(int num_cols)
{
  srand(31337);
  auto const table = create_random_fixed_table<T>(num_cols, 10, true);
  auto const filepath =
    temp_env->get_temp_filepath(typeid(T).name() + std::to_string(num_cols) + ".parquet");
  cudf::io::parquet_writer_options const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table->view());
  cudf::io::write_parquet(out_opts);
  return filepath;
}

TEST_F(ParquetWriterTest, MultipleMismatchedSources)
{
  auto const int5file = create_parquet_file<int>(5);
  {
    auto const float5file = create_parquet_file<float>(5);
    std::vector<std::string> files{int5file, float5file};
    cudf::io::parquet_reader_options const read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{files});
    EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
  }
  {
    auto const int10file = create_parquet_file<int>(10);
    std::vector<std::string> files{int5file, int10file};
    cudf::io::parquet_reader_options const read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{files});
    EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
  }
}

TEST_F(ParquetWriterTest, Slice)
{
  auto col =
    cudf::test::fixed_width_column_wrapper<int>{{1, 2, 3, 4, 5}, {true, true, true, false, true}};
  std::vector<cudf::size_type> indices{2, 5};
  std::vector<cudf::column_view> result = cudf::slice(col, indices);
  cudf::table_view tbl{result};

  auto filepath = temp_env->get_temp_filepath("Slice.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto read_table = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(read_table.tbl->view(), tbl);
}

TEST_F(ParquetChunkedWriterTest, SingleTable)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedSingle.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(*table1);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *table1);
}

TEST_F(ParquetChunkedWriterTest, SimpleTable)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, true);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedSimple.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(*table1).write(*table2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, LargeTables)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(512, 4096, true);
  auto table2 = create_random_fixed_table<int>(512, 8192, true);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedLarge.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  auto md = cudf::io::parquet_chunked_writer(args).write(*table1).write(*table2).close();
  ASSERT_EQ(md, nullptr);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, ManyTables)
{
  srand(31337);
  std::vector<std::unique_ptr<table>> tables;
  std::vector<table_view> table_views;
  constexpr int num_tables = 96;
  for (int idx = 0; idx < num_tables; idx++) {
    auto tbl = create_random_fixed_table<int>(16, 64, true);
    table_views.push_back(*tbl);
    tables.push_back(std::move(tbl));
  }

  auto expected = cudf::concatenate(table_views);

  auto filepath = temp_env->get_temp_filepath("ChunkedManyTables.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer writer(args);
  std::for_each(table_views.begin(), table_views.end(), [&writer](table_view const& tbl) {
    writer.write(tbl);
  });
  auto md = writer.close({"dummy/path"});
  ASSERT_NE(md, nullptr);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetChunkedWriterTest, Strings)
{
  std::vector<std::unique_ptr<cudf::column>> cols;

  bool mask1[] = {true, true, false, true, true, true, true};
  std::vector<char const*> h_strings1{"four", "score", "and", "seven", "years", "ago", "abcdefgh"};
  cudf::test::strings_column_wrapper strings1(h_strings1.begin(), h_strings1.end(), mask1);
  cols.push_back(strings1.release());
  cudf::table tbl1(std::move(cols));

  bool mask2[] = {false, true, true, true, true, true, true};
  std::vector<char const*> h_strings2{"ooooo", "ppppppp", "fff", "j", "cccc", "bbb", "zzzzzzzzzzz"};
  cudf::test::strings_column_wrapper strings2(h_strings2.begin(), h_strings2.end(), mask2);
  cols.push_back(strings2.release());
  cudf::table tbl2(std::move(cols));

  auto expected = cudf::concatenate(std::vector<table_view>({tbl1, tbl2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedStrings.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(tbl1).write(tbl2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetChunkedWriterTest, ListColumn)
{
  auto valids  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // COL0 (Same nullability) ====================
  // [NULL, 2, NULL]
  // []
  // [4, 5]
  // NULL
  lcw col0_tbl0{{{{1, 2, 3}, valids}, {}, {4, 5}, {}}, valids2};

  // [7, 8, 9]
  // []
  // [NULL, 11]
  // NULL
  lcw col0_tbl1{{{7, 8, 9}, {}, {{10, 11}, valids}, {}}, valids2};

  // COL1 (Nullability different in different chunks, test of merging nullability in writer)
  // [NULL, 2, NULL]
  // []
  // [4, 5]
  // []
  lcw col1_tbl0{{{1, 2, 3}, valids}, {}, {4, 5}, {}};

  // [7, 8, 9]
  // []
  // [10, 11]
  // NULL
  lcw col1_tbl1{{{7, 8, 9}, {}, {10, 11}, {}}, valids2};

  // COL2 (non-nested columns to test proper schema construction)
  size_t num_rows_tbl0 = static_cast<cudf::column_view>(col0_tbl0).size();
  size_t num_rows_tbl1 = static_cast<cudf::column_view>(col0_tbl1).size();
  auto seq_col0        = random_values<int>(num_rows_tbl0);
  auto seq_col1        = random_values<int>(num_rows_tbl1);

  column_wrapper<int> col2_tbl0{seq_col0.begin(), seq_col0.end(), valids};
  column_wrapper<int> col2_tbl1{seq_col1.begin(), seq_col1.end(), valids2};

  auto tbl0 = table_view({col0_tbl0, col1_tbl0, col2_tbl0});
  auto tbl1 = table_view({col0_tbl1, col1_tbl1, col2_tbl1});

  auto expected = cudf::concatenate(std::vector<table_view>({tbl0, tbl1}));

  auto filepath = temp_env->get_temp_filepath("ChunkedLists.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(tbl0).write(tbl1);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetChunkedWriterTest, ListOfStruct)
{
  // Table 1
  auto weight_1   = cudf::test::fixed_width_column_wrapper<float>{{57.5, 51.1, 15.3}};
  auto ages_1     = cudf::test::fixed_width_column_wrapper<int32_t>{{30, 27, 5}};
  auto struct_1_1 = cudf::test::structs_column_wrapper{weight_1, ages_1};
  auto is_human_1 = cudf::test::fixed_width_column_wrapper<bool>{{true, true, false}};
  auto struct_2_1 = cudf::test::structs_column_wrapper{{is_human_1, struct_1_1}};

  auto list_offsets_column_1 =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 3, 3}.release();
  auto num_list_rows_1 = list_offsets_column_1->size() - 1;

  auto list_col_1 = cudf::make_lists_column(
    num_list_rows_1, std::move(list_offsets_column_1), struct_2_1.release(), 0, {});

  auto table_1 = table_view({*list_col_1});

  // Table 2
  auto weight_2   = cudf::test::fixed_width_column_wrapper<float>{{1.1, -1.0, -1.0}};
  auto ages_2     = cudf::test::fixed_width_column_wrapper<int32_t>{{31, 351, 351}, {1, 1, 0}};
  auto struct_1_2 = cudf::test::structs_column_wrapper{{weight_2, ages_2}, {1, 0, 1}};
  auto is_human_2 = cudf::test::fixed_width_column_wrapper<bool>{{false, false, false}, {1, 1, 0}};
  auto struct_2_2 = cudf::test::structs_column_wrapper{{is_human_2, struct_1_2}};

  auto list_offsets_column_2 =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2, 3}.release();
  auto num_list_rows_2 = list_offsets_column_2->size() - 1;

  auto list_col_2 = cudf::make_lists_column(
    num_list_rows_2, std::move(list_offsets_column_2), struct_2_2.release(), 0, {});

  auto table_2 = table_view({*list_col_2});

  auto full_table = cudf::concatenate(std::vector<table_view>({table_1, table_2}));

  cudf::io::table_input_metadata expected_metadata(table_1);
  expected_metadata.column_metadata[0].set_name("family");
  expected_metadata.column_metadata[0].child(1).set_nullability(false);
  expected_metadata.column_metadata[0].child(1).child(0).set_name("human?");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).child(1).set_name("age");

  auto filepath = temp_env->get_temp_filepath("ChunkedListOfStruct.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  args.set_metadata(expected_metadata);
  cudf::io::parquet_chunked_writer(args).write(table_1).write(table_2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result.tbl, *full_table);
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(ParquetChunkedWriterTest, ListOfStructOfStructOfListOfList)
{
  auto valids  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // Table 1 ===========================

  // []
  // [NULL, 2, NULL]
  // [4, 5]
  // NULL
  lcw land_1{{{}, {{1, 2, 3}, valids}, {4, 5}, {}}, valids2};

  // []
  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8], []]
  // [[]]
  lcw flats_1{lcw{}, {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, {{7, 8}, {}}, lcw{lcw{}}};

  auto weight_1   = cudf::test::fixed_width_column_wrapper<float>{{57.5, 51.1, 15.3, 1.1}};
  auto ages_1     = cudf::test::fixed_width_column_wrapper<int32_t>{{30, 27, 5, 31}};
  auto struct_1_1 = cudf::test::structs_column_wrapper{weight_1, ages_1, land_1, flats_1};
  auto is_human_1 = cudf::test::fixed_width_column_wrapper<bool>{{true, true, false, false}};
  auto struct_2_1 = cudf::test::structs_column_wrapper{{is_human_1, struct_1_1}};

  auto list_offsets_column_1 =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 3, 4}.release();
  auto num_list_rows_1 = list_offsets_column_1->size() - 1;

  auto list_col_1 = cudf::make_lists_column(
    num_list_rows_1, std::move(list_offsets_column_1), struct_2_1.release(), 0, {});

  auto table_1 = table_view({*list_col_1});

  // Table 2 ===========================

  // []
  // [7, 8, 9]
  lcw land_2{{}, {7, 8, 9}};

  // [[]]
  // [[], [], []]
  lcw flats_2{lcw{lcw{}}, lcw{lcw{}, lcw{}, lcw{}}};

  auto weight_2   = cudf::test::fixed_width_column_wrapper<float>{{-1.0, -1.0}};
  auto ages_2     = cudf::test::fixed_width_column_wrapper<int32_t>{{351, 351}, {1, 0}};
  auto struct_1_2 = cudf::test::structs_column_wrapper{{weight_2, ages_2, land_2, flats_2}, {0, 1}};
  auto is_human_2 = cudf::test::fixed_width_column_wrapper<bool>{{false, false}, {1, 0}};
  auto struct_2_2 = cudf::test::structs_column_wrapper{{is_human_2, struct_1_2}};

  auto list_offsets_column_2 =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2}.release();
  auto num_list_rows_2 = list_offsets_column_2->size() - 1;

  auto list_col_2 = cudf::make_lists_column(
    num_list_rows_2, std::move(list_offsets_column_2), struct_2_2.release(), 0, {});

  auto table_2 = table_view({*list_col_2});

  auto full_table = cudf::concatenate(std::vector<table_view>({table_1, table_2}));

  cudf::io::table_input_metadata expected_metadata(table_1);
  expected_metadata.column_metadata[0].set_name("family");
  expected_metadata.column_metadata[0].child(1).set_nullability(false);
  expected_metadata.column_metadata[0].child(1).child(0).set_name("human?");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).child(1).set_name("age");
  expected_metadata.column_metadata[0].child(1).child(1).child(2).set_name("land_unit");
  expected_metadata.column_metadata[0].child(1).child(1).child(3).set_name("flats");

  auto filepath = temp_env->get_temp_filepath("ListOfStructOfStructOfListOfList.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  args.set_metadata(expected_metadata);
  cudf::io::parquet_chunked_writer(args).write(table_1).write(table_2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result.tbl, *full_table);
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);

  // We specifically mentioned in input schema that struct_2 is non-nullable across chunked calls.
  auto result_parent_list = result.tbl->get_column(0);
  auto result_struct_2    = result_parent_list.child(cudf::lists_column_view::child_column_index);
  EXPECT_EQ(result_struct_2.nullable(), false);
}

TEST_F(ParquetChunkedWriterTest, MismatchedTypes)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(4, 4, true);
  auto table2 = create_random_fixed_table<float>(4, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedMismatchedTypes.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer writer(args);
  writer.write(*table1);
  EXPECT_THROW(writer.write(*table2), cudf::logic_error);
  writer.close();
}

TEST_F(ParquetChunkedWriterTest, ChunkedWriteAfterClosing)
{
  srand(31337);
  auto table = create_random_fixed_table<int>(4, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedWriteAfterClosing.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer writer(args);
  writer.write(*table).close();
  EXPECT_THROW(writer.write(*table), cudf::logic_error);
}

TEST_F(ParquetChunkedWriterTest, ReadingUnclosedFile)
{
  srand(31337);
  auto table = create_random_fixed_table<int>(4, 4, true);

  auto filepath = temp_env->get_temp_filepath("ReadingUnclosedFile.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer writer(args);
  writer.write(*table);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
}

TEST_F(ParquetChunkedWriterTest, MismatchedStructure)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(4, 4, true);
  auto table2 = create_random_fixed_table<float>(3, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedMismatchedStructure.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer writer(args);
  writer.write(*table1);
  EXPECT_THROW(writer.write(*table2), cudf::logic_error);
  writer.close();
}

TEST_F(ParquetChunkedWriterTest, MismatchedStructureList)
{
  auto valids  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // COL0 (mismatched depth) ====================
  // [NULL, 2, NULL]
  // []
  // [4, 5]
  // NULL
  lcw col00{{{{1, 2, 3}, valids}, {}, {4, 5}, {}}, valids2};

  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8]]
  // []
  // [[]]
  lcw col01{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, {{7, 8}}, lcw{}, lcw{lcw{}}};

  // COL2 (non-nested columns to test proper schema construction)
  size_t num_rows = static_cast<cudf::column_view>(col00).size();
  auto seq_col0   = random_values<int>(num_rows);
  auto seq_col1   = random_values<int>(num_rows);

  column_wrapper<int> col10{seq_col0.begin(), seq_col0.end(), valids};
  column_wrapper<int> col11{seq_col1.begin(), seq_col1.end(), valids2};

  auto tbl0 = table_view({col00, col10});
  auto tbl1 = table_view({col01, col11});

  auto filepath = temp_env->get_temp_filepath("ChunkedLists.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer writer(args);
  writer.write(tbl0);
  EXPECT_THROW(writer.write(tbl1), cudf::logic_error);
}

TEST_F(ParquetChunkedWriterTest, DifferentNullability)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, false);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedNullable.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(*table1).write(*table2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, DifferentNullabilityStruct)
{
  // Struct<is_human:bool (non-nullable),
  //        Struct<weight:float>,
  //               age:int
  //              > (nullable)
  //       > (non-nullable)

  // Table 1: is_human and struct_1 are non-nullable but should be nullable when read back.
  auto weight_1   = cudf::test::fixed_width_column_wrapper<float>{{57.5, 51.1, 15.3}};
  auto ages_1     = cudf::test::fixed_width_column_wrapper<int32_t>{{30, 27, 5}};
  auto struct_1_1 = cudf::test::structs_column_wrapper{weight_1, ages_1};
  auto is_human_1 = cudf::test::fixed_width_column_wrapper<bool>{{true, true, false}};
  auto struct_2_1 = cudf::test::structs_column_wrapper{{is_human_1, struct_1_1}};
  auto table_1    = cudf::table_view({struct_2_1});

  // Table 2: struct_1 and is_human are nullable now so if we hadn't assumed worst case (nullable)
  // when writing table_1, we would have wrong pages for it.
  auto weight_2   = cudf::test::fixed_width_column_wrapper<float>{{1.1, -1.0, -1.0}};
  auto ages_2     = cudf::test::fixed_width_column_wrapper<int32_t>{{31, 351, 351}, {1, 1, 0}};
  auto struct_1_2 = cudf::test::structs_column_wrapper{{weight_2, ages_2}, {1, 0, 1}};
  auto is_human_2 = cudf::test::fixed_width_column_wrapper<bool>{{false, false, false}, {1, 1, 0}};
  auto struct_2_2 = cudf::test::structs_column_wrapper{{is_human_2, struct_1_2}};
  auto table_2    = cudf::table_view({struct_2_2});

  auto full_table = cudf::concatenate(std::vector<table_view>({table_1, table_2}));

  cudf::io::table_input_metadata expected_metadata(table_1);
  expected_metadata.column_metadata[0].set_name("being");
  expected_metadata.column_metadata[0].child(0).set_name("human?");
  expected_metadata.column_metadata[0].child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("age");

  auto filepath = temp_env->get_temp_filepath("ChunkedNullableStruct.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  args.set_metadata(expected_metadata);
  cudf::io::parquet_chunked_writer(args).write(table_1).write(table_2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result.tbl, *full_table);
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(ParquetChunkedWriterTest, ForcedNullability)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, false);
  auto table2 = create_random_fixed_table<int>(5, 5, false);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedNoNullable.parquet");

  cudf::io::table_input_metadata metadata(*table1);

  // In the absence of prescribed per-column nullability in metadata, the writer assumes the worst
  // and considers all columns nullable. However cudf::concatenate will not force nulls in case no
  // columns are nullable. To get the expected result, we tell the writer the nullability of all
  // columns in advance.
  for (auto& col_meta : metadata.column_metadata) {
    col_meta.set_nullability(false);
  }

  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath})
      .metadata(std::move(metadata));
  cudf::io::parquet_chunked_writer(args).write(*table1).write(*table2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, ForcedNullabilityList)
{
  srand(31337);

  auto valids  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // COL0 ====================
  // [1, 2, 3]
  // []
  // [4, 5]
  // NULL
  lcw col00{{{1, 2, 3}, {}, {4, 5}, {}}, valids2};

  // [7]
  // []
  // [8, 9, 10, 11]
  // NULL
  lcw col01{{{7}, {}, {8, 9, 10, 11}, {}}, valids2};

  // COL1 (non-nested columns to test proper schema construction)
  size_t num_rows = static_cast<cudf::column_view>(col00).size();
  auto seq_col0   = random_values<int>(num_rows);
  auto seq_col1   = random_values<int>(num_rows);

  column_wrapper<int> col10{seq_col0.begin(), seq_col0.end(), valids};
  column_wrapper<int> col11{seq_col1.begin(), seq_col1.end(), valids2};

  auto table1 = table_view({col00, col10});
  auto table2 = table_view({col01, col11});

  auto full_table = cudf::concatenate(std::vector<table_view>({table1, table2}));

  cudf::io::table_input_metadata metadata(table1);
  metadata.column_metadata[0].set_nullability(true);  // List is nullable at first (root) level
  metadata.column_metadata[0].child(1).set_nullability(
    false);                                           // non-nullable at second (leaf) level
  metadata.column_metadata[1].set_nullability(true);

  auto filepath = temp_env->get_temp_filepath("ChunkedListNullable.parquet");

  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath})
      .metadata(std::move(metadata));
  cudf::io::parquet_chunked_writer(args).write(table1).write(table2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, ForcedNullabilityStruct)
{
  // Struct<is_human:bool (non-nullable),
  //        Struct<weight:float>,
  //               age:int
  //              > (nullable)
  //       > (non-nullable)

  // Table 1: is_human and struct_2 are non-nullable and should stay that way when read back.
  auto weight_1   = cudf::test::fixed_width_column_wrapper<float>{{57.5, 51.1, 15.3}};
  auto ages_1     = cudf::test::fixed_width_column_wrapper<int32_t>{{30, 27, 5}};
  auto struct_1_1 = cudf::test::structs_column_wrapper{weight_1, ages_1};
  auto is_human_1 = cudf::test::fixed_width_column_wrapper<bool>{{true, true, false}};
  auto struct_2_1 = cudf::test::structs_column_wrapper{{is_human_1, struct_1_1}};
  auto table_1    = cudf::table_view({struct_2_1});

  auto weight_2   = cudf::test::fixed_width_column_wrapper<float>{{1.1, -1.0, -1.0}};
  auto ages_2     = cudf::test::fixed_width_column_wrapper<int32_t>{{31, 351, 351}, {1, 1, 0}};
  auto struct_1_2 = cudf::test::structs_column_wrapper{{weight_2, ages_2}, {1, 0, 1}};
  auto is_human_2 = cudf::test::fixed_width_column_wrapper<bool>{{false, false, false}};
  auto struct_2_2 = cudf::test::structs_column_wrapper{{is_human_2, struct_1_2}};
  auto table_2    = cudf::table_view({struct_2_2});

  auto full_table = cudf::concatenate(std::vector<table_view>({table_1, table_2}));

  cudf::io::table_input_metadata expected_metadata(table_1);
  expected_metadata.column_metadata[0].set_name("being").set_nullability(false);
  expected_metadata.column_metadata[0].child(0).set_name("human?").set_nullability(false);
  expected_metadata.column_metadata[0].child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("age");

  auto filepath = temp_env->get_temp_filepath("ChunkedNullableStruct.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  args.set_metadata(expected_metadata);
  cudf::io::parquet_chunked_writer(args).write(table_1).write(table_2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(ParquetChunkedWriterTest, ReadRowGroups)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, true);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table2, *table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedRowGroups.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  {
    cudf::io::parquet_chunked_writer(args).write(*table1).write(*table2);
  }

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .row_groups({{1, 0, 1}});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, ReadRowGroupsError)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedRowGroupsError.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(*table1);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).row_groups({{0, 1}});
  EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
  read_opts.set_row_groups({{-1}});
  EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
  read_opts.set_row_groups({{0}, {0}});
  EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
}

TEST_F(ParquetWriterTest, DecimalWrite)
{
  constexpr cudf::size_type num_rows = 500;
  auto seq_col0                      = random_values<int32_t>(num_rows);
  auto seq_col1                      = random_values<int64_t>(num_rows);

  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });

  auto col0 = cudf::test::fixed_point_column_wrapper<int32_t>{
    seq_col0.begin(), seq_col0.end(), valids, numeric::scale_type{5}};
  auto col1 = cudf::test::fixed_point_column_wrapper<int64_t>{
    seq_col1.begin(), seq_col1.end(), valids, numeric::scale_type{-9}};

  auto table = table_view({col0, col1});

  auto filepath = temp_env->get_temp_filepath("DecimalWrite.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table);

  cudf::io::table_input_metadata expected_metadata(table);

  // verify failure if too small a precision is given
  expected_metadata.column_metadata[0].set_decimal_precision(7);
  expected_metadata.column_metadata[1].set_decimal_precision(1);
  args.set_metadata(expected_metadata);
  EXPECT_THROW(cudf::io::write_parquet(args), cudf::logic_error);

  // verify success if equal precision is given
  expected_metadata.column_metadata[0].set_decimal_precision(7);
  expected_metadata.column_metadata[1].set_decimal_precision(9);
  args.set_metadata(std::move(expected_metadata));
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, table);
}

TYPED_TEST(ParquetChunkedWriterNumericTypeTest, UnalignedSize)
{
  // write out two 31 row tables and make sure they get
  // read back with all their validity bits in the right place

  using T = TypeParam;

  int num_els = 31;
  std::vector<std::unique_ptr<cudf::column>> cols;

  bool mask[] = {false, true, true, true, true, true, true, true, true, true, true,
                 true,  true, true, true, true, true, true, true, true, true, true,

                 true,  true, true, true, true, true, true, true, true};
  T c1a[num_els];
  std::fill(c1a, c1a + num_els, static_cast<T>(5));
  T c1b[num_els];
  std::fill(c1b, c1b + num_els, static_cast<T>(6));
  column_wrapper<T> c1a_w(c1a, c1a + num_els, mask);
  column_wrapper<T> c1b_w(c1b, c1b + num_els, mask);
  cols.push_back(c1a_w.release());
  cols.push_back(c1b_w.release());
  cudf::table tbl1(std::move(cols));

  T c2a[num_els];
  std::fill(c2a, c2a + num_els, static_cast<T>(8));
  T c2b[num_els];
  std::fill(c2b, c2b + num_els, static_cast<T>(9));
  column_wrapper<T> c2a_w(c2a, c2a + num_els, mask);
  column_wrapper<T> c2b_w(c2b, c2b + num_els, mask);
  cols.push_back(c2a_w.release());
  cols.push_back(c2b_w.release());
  cudf::table tbl2(std::move(cols));

  auto expected = cudf::concatenate(std::vector<table_view>({tbl1, tbl2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedUnalignedSize.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(tbl1).write(tbl2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TYPED_TEST(ParquetChunkedWriterNumericTypeTest, UnalignedSize2)
{
  // write out two 33 row tables and make sure they get
  // read back with all their validity bits in the right place

  using T = TypeParam;

  int num_els = 33;
  std::vector<std::unique_ptr<cudf::column>> cols;

  bool mask[] = {false, true, true, true, true, true, true, true, true, true, true,
                 true,  true, true, true, true, true, true, true, true, true, true,
                 true,  true, true, true, true, true, true, true, true, true, true};

  T c1a[num_els];
  std::fill(c1a, c1a + num_els, static_cast<T>(5));
  T c1b[num_els];
  std::fill(c1b, c1b + num_els, static_cast<T>(6));
  column_wrapper<T> c1a_w(c1a, c1a + num_els, mask);
  column_wrapper<T> c1b_w(c1b, c1b + num_els, mask);
  cols.push_back(c1a_w.release());
  cols.push_back(c1b_w.release());
  cudf::table tbl1(std::move(cols));

  T c2a[num_els];
  std::fill(c2a, c2a + num_els, static_cast<T>(8));
  T c2b[num_els];
  std::fill(c2b, c2b + num_els, static_cast<T>(9));
  column_wrapper<T> c2a_w(c2a, c2a + num_els, mask);
  column_wrapper<T> c2b_w(c2b, c2b + num_els, mask);
  cols.push_back(c2a_w.release());
  cols.push_back(c2b_w.release());
  cudf::table tbl2(std::move(cols));

  auto expected = cudf::concatenate(std::vector<table_view>({tbl1, tbl2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedUnalignedSize2.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(tbl1).write(tbl2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

// custom mem mapped data sink that supports device writes
template <bool supports_device_writes>
class custom_test_memmap_sink : public cudf::io::data_sink {
 public:
  explicit custom_test_memmap_sink(std::vector<char>* mm_writer_buf)
  {
    mm_writer = cudf::io::data_sink::create(mm_writer_buf);
  }

  virtual ~custom_test_memmap_sink() { mm_writer->flush(); }

  void host_write(void const* data, size_t size) override { mm_writer->host_write(data, size); }

  [[nodiscard]] bool supports_device_write() const override { return supports_device_writes; }

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    this->device_write_async(gpu_data, size, stream).get();
  }

  std::future<void> device_write_async(void const* gpu_data,
                                       size_t size,
                                       rmm::cuda_stream_view stream) override
  {
    return std::async(std::launch::deferred, [=] {
      char* ptr = nullptr;
      CUDF_CUDA_TRY(cudaMallocHost(&ptr, size));
      CUDF_CUDA_TRY(cudaMemcpyAsync(ptr, gpu_data, size, cudaMemcpyDefault, stream.value()));
      stream.synchronize();
      mm_writer->host_write(ptr, size);
      CUDF_CUDA_TRY(cudaFreeHost(ptr));
    });
  }

  void flush() override { mm_writer->flush(); }

  size_t bytes_written() override { return mm_writer->bytes_written(); }

 private:
  std::unique_ptr<data_sink> mm_writer;
};

TEST_F(ParquetWriterStressTest, LargeTableWeakCompression)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<false> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_random_fixed_table<int>(16, 4 * 1024 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, LargeTableGoodCompression)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<false> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 128 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, LargeTableWithValids)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<false> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 6, true);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, DeviceWriteLargeTableWeakCompression)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<true> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_random_fixed_table<int>(16, 4 * 1024 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, DeviceWriteLargeTableGoodCompression)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<true> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 128 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, DeviceWriteLargeTableWithValids)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<true> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 6, true);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetReaderTest, UserBounds)
{
  // trying to read more rows than there are should result in
  // receiving the properly capped # of rows
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("TooManyRows.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).num_rows(16);
    auto result = cudf::io::read_parquet(read_opts);

    // we should only get back 4 rows
    EXPECT_EQ(result.tbl->view().column(0).size(), 4);
  }

  // trying to read past the end of the # of actual rows should result
  // in empty columns.
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("PastBounds.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).skip_rows(4);
    auto result = cudf::io::read_parquet(read_opts);

    // we should get empty columns back
    EXPECT_EQ(result.tbl->view().num_columns(), 4);
    EXPECT_EQ(result.tbl->view().column(0).size(), 0);
  }

  // trying to read 0 rows should result in empty columns
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("ZeroRows.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).num_rows(0);
    auto result = cudf::io::read_parquet(read_opts);

    EXPECT_EQ(result.tbl->view().num_columns(), 4);
    EXPECT_EQ(result.tbl->view().column(0).size(), 0);
  }

  // trying to read 0 rows past the end of the # of actual rows should result
  // in empty columns.
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("ZeroRowsPastBounds.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .skip_rows(4)
        .num_rows(0);
    auto result = cudf::io::read_parquet(read_opts);

    // we should get empty columns back
    EXPECT_EQ(result.tbl->view().num_columns(), 4);
    EXPECT_EQ(result.tbl->view().column(0).size(), 0);
  }
}

TEST_F(ParquetReaderTest, UserBoundsWithNulls)
{
  // clang-format off
  cudf::test::fixed_width_column_wrapper<float> col{{1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3, 4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5, 6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7, 8,8,8,8,8,8,8,8}
                                                   ,{1,1,1,0,0,0,1,1, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,0,0,  1,0,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,0}};
  // clang-format on
  cudf::table_view tbl({col});
  auto filepath = temp_env->get_temp_filepath("UserBoundsWithNulls.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl);
  cudf::io::write_parquet(out_args);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{ {-1, -1}, {1, 3}, {3, -1},
                                           {31, -1}, {32, -1}, {33, -1},
                                           {31, 5}, {32, 5}, {33, 5},
                                           {-1, 7}, {-1, 31}, {-1, 32}, {-1, 33},
                                           {62, -1}, {63, -1},
                                           {62, 2}, {63, 1}};
  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? static_cast<cudf::column_view>(col).size() - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(col, slice_indices);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), expected[0]);
  }
}

TEST_F(ParquetReaderTest, UserBoundsWithNullsMixedTypes)
{
  constexpr int num_rows = 32 * 1024;

  std::mt19937 gen(6542);
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto values = thrust::make_counting_iterator(0);

  // int64
  cudf::test::fixed_width_column_wrapper<int64_t> c0(values, values + num_rows, valids);

  // list<float>
  constexpr int floats_per_row = 4;
  auto c1_offset_iter          = cudf::detail::make_counting_transform_iterator(
    0, [floats_per_row](cudf::size_type idx) { return idx * floats_per_row; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> c1_offsets(c1_offset_iter,
                                                                     c1_offset_iter + num_rows + 1);
  cudf::test::fixed_width_column_wrapper<float> c1_floats(
    values, values + (num_rows * floats_per_row), valids);
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids, valids + num_rows);

  auto _c1 = cudf::make_lists_column(
    num_rows, c1_offsets.release(), c1_floats.release(), null_count, std::move(null_mask));
  auto c1 = cudf::purge_nonempty_nulls(*_c1);

  // list<list<int>>
  auto c2 = make_parquet_list_list_col<int>(0, num_rows, 5, 8, true);

  // struct<list<string>, int, float>
  std::vector<std::string> strings{
    "abc", "x", "bananas", "gpu", "minty", "backspace", "", "cayenne", "turbine", "soft"};
  std::uniform_int_distribution<int> uni(0, strings.size() - 1);
  auto string_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](cudf::size_type idx) { return strings[uni(gen)]; });
  constexpr int string_per_row  = 3;
  constexpr int num_string_rows = num_rows * string_per_row;
  cudf::test::strings_column_wrapper string_col{string_iter, string_iter + num_string_rows};
  auto offset_iter = cudf::detail::make_counting_transform_iterator(
    0, [string_per_row](cudf::size_type idx) { return idx * string_per_row; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets(offset_iter,
                                                                  offset_iter + num_rows + 1);

  auto _c3_valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 200; });
  std::vector<bool> c3_valids(num_rows);
  std::copy(_c3_valids, _c3_valids + num_rows, c3_valids.begin());
  std::tie(null_mask, null_count) = cudf::test::detail::make_null_mask(valids, valids + num_rows);
  auto _c3_list                   = cudf::make_lists_column(
    num_rows, offsets.release(), string_col.release(), null_count, std::move(null_mask));
  auto c3_list = cudf::purge_nonempty_nulls(*_c3_list);
  cudf::test::fixed_width_column_wrapper<int> c3_ints(values, values + num_rows, valids);
  cudf::test::fixed_width_column_wrapper<float> c3_floats(values, values + num_rows, valids);
  std::vector<std::unique_ptr<cudf::column>> c3_children;
  c3_children.push_back(std::move(c3_list));
  c3_children.push_back(c3_ints.release());
  c3_children.push_back(c3_floats.release());
  cudf::test::structs_column_wrapper _c3(std::move(c3_children), c3_valids);
  auto c3 = cudf::purge_nonempty_nulls(_c3);

  // write it out
  cudf::table_view tbl({c0, *c1, *c2, *c3});
  auto filepath = temp_env->get_temp_filepath("UserBoundsWithNullsMixedTypes.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl);
  cudf::io::write_parquet(out_args);

  // read it back
  std::vector<std::pair<int, int>> params{
    {-1, -1}, {0, num_rows}, {1, num_rows - 1}, {num_rows - 1, 1}, {517, 22000}};
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? num_rows - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(tbl, slice_indices);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, expected[0]);
  }
}

TEST_F(ParquetReaderTest, UserBoundsWithNullsLarge)
{
  constexpr int num_rows = 30 * 1000000;

  std::mt19937 gen(6747);
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto values = thrust::make_counting_iterator(0);

  cudf::test::fixed_width_column_wrapper<int> col(values, values + num_rows, valids);

  // this file will have row groups of 1,000,000 each
  cudf::table_view tbl({col});
  auto filepath = temp_env->get_temp_filepath("UserBoundsWithNullsLarge.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl);
  cudf::io::write_parquet(out_args);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{ {-1, -1}, {31, -1}, {32, -1}, {33, -1}, {1613470, -1}, {1999999, -1},
                                           {31, 1}, {32, 1}, {33, 1},
                                           // deliberately span some row group boundaries
                                           {999000, 1001}, {999000, 2000}, {2999999, 2}, {13999997, -1},
                                           {16785678, 3}, {22996176, 31},
                                           {24001231, 17}, {29000001, 989999}, {29999999, 1} };
  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? static_cast<cudf::column_view>(col).size() - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(col, slice_indices);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), expected[0]);
  }
}

TEST_F(ParquetReaderTest, ListUserBoundsWithNullsLarge)
{
  constexpr int num_rows = 5 * 1000000;
  auto colp              = make_parquet_list_list_col<int>(0, num_rows, 5, 8, true);
  cudf::column_view col  = *colp;

  // this file will have row groups of 1,000,000 each
  cudf::table_view tbl({col});
  auto filepath = temp_env->get_temp_filepath("ListUserBoundsWithNullsLarge.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl);
  cudf::io::write_parquet(out_args);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{ {-1, -1}, {31, -1}, {32, -1}, {33, -1}, {161470, -1}, {4499997, -1},
                                           {31, 1}, {32, 1}, {33, 1},
                                           // deliberately span some row group boundaries
                                           {999000, 1001}, {999000, 2000}, {2999999, 2},
                                           {1678567, 3}, {4299676, 31},
                                           {4001231, 17}, {1900000, 989999}, {4999999, 1} };
  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? static_cast<cudf::column_view>(col).size() - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(col, slice_indices);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), expected[0]);
  }
}

TEST_F(ParquetReaderTest, ReorderedColumns)
{
  {
    auto a = cudf::test::strings_column_wrapper{{"a", "", "c"}, {true, false, true}};
    auto b = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3};

    cudf::table_view tbl{{a, b}};
    auto filepath = temp_env->get_temp_filepath("ReorderedColumns.parquet");
    cudf::io::table_input_metadata md(tbl);
    md.column_metadata[0].set_name("a");
    md.column_metadata[1].set_name("b");
    cudf::io::parquet_writer_options opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl).metadata(md);
    cudf::io::write_parquet(opts);

    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .columns({"b", "a"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), a);
  }

  {
    auto a = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3};
    auto b = cudf::test::strings_column_wrapper{{"a", "", "c"}, {true, false, true}};

    cudf::table_view tbl{{a, b}};
    auto filepath = temp_env->get_temp_filepath("ReorderedColumns2.parquet");
    cudf::io::table_input_metadata md(tbl);
    md.column_metadata[0].set_name("a");
    md.column_metadata[1].set_name("b");
    cudf::io::parquet_writer_options opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl).metadata(md);
    cudf::io::write_parquet(opts);

    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .columns({"b", "a"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), a);
  }

  auto a = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3, 10, 20, 30};
  auto b = cudf::test::strings_column_wrapper{{"a", "", "c", "cats", "dogs", "owls"},
                                              {true, false, true, true, false, true}};
  auto c = cudf::test::fixed_width_column_wrapper<int>{{15, 16, 17, 25, 26, 32},
                                                       {false, true, true, true, true, false}};
  auto d = cudf::test::strings_column_wrapper{"ducks", "sheep", "cows", "fish", "birds", "ants"};

  cudf::table_view tbl{{a, b, c, d}};
  auto filepath = temp_env->get_temp_filepath("ReorderedColumns3.parquet");
  cudf::io::table_input_metadata md(tbl);
  md.column_metadata[0].set_name("a");
  md.column_metadata[1].set_name("b");
  md.column_metadata[2].set_name("c");
  md.column_metadata[3].set_name("d");
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .metadata(std::move(md));
  cudf::io::write_parquet(opts);

  {
    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .columns({"d", "a", "b", "c"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), d);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), a);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(3), c);
  }

  {
    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .columns({"c", "d", "a", "b"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), c);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), d);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), a);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(3), b);
  }

  {
    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .columns({"d", "c", "b", "a"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), d);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), c);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(3), a);
  }
}

TEST_F(ParquetReaderTest, SelectNestedColumn)
{
  // Struct<is_human:bool,
  //        Struct<weight:float,
  //               ages:int,
  //               land_unit:List<int>>,
  //               flats:List<List<int>>
  //              >
  //       >

  auto weights_col = cudf::test::fixed_width_column_wrapper<float>{1.1, 2.4, 5.3, 8.0, 9.6, 6.9};

  auto ages_col =
    cudf::test::fixed_width_column_wrapper<int32_t>{{48, 27, 25, 31, 351, 351}, {1, 1, 1, 1, 1, 0}};

  auto struct_1 = cudf::test::structs_column_wrapper{{weights_col, ages_col}, {1, 1, 1, 1, 0, 1}};

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {1, 1, 0, 1, 1, 0}};

  auto struct_2 =
    cudf::test::structs_column_wrapper{{is_human_col, struct_1}, {0, 1, 1, 1, 1, 1}}.release();

  auto input = table_view({*struct_2});

  cudf::io::table_input_metadata input_metadata(input);
  input_metadata.column_metadata[0].set_name("being");
  input_metadata.column_metadata[0].child(0).set_name("human?");
  input_metadata.column_metadata[0].child(1).set_name("particulars");
  input_metadata.column_metadata[0].child(1).child(0).set_name("weight");
  input_metadata.column_metadata[0].child(1).child(1).set_name("age");

  auto filepath = temp_env->get_temp_filepath("SelectNestedColumn.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
      .metadata(std::move(input_metadata));
  cudf::io::write_parquet(args);

  {  // Test selecting a single leaf from the table
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath))
        .columns({"being.particulars.age"});
    auto const result = cudf::io::read_parquet(read_args);

    auto expect_ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
      {48, 27, 25, 31, 351, 351}, {1, 1, 1, 1, 1, 0}};
    auto expect_s_1 = cudf::test::structs_column_wrapper{{expect_ages_col}, {1, 1, 1, 1, 0, 1}};
    auto expect_s_2 =
      cudf::test::structs_column_wrapper{{expect_s_1}, {0, 1, 1, 1, 1, 1}}.release();
    auto expected = table_view({*expect_s_2});

    cudf::io::table_input_metadata expected_metadata(expected);
    expected_metadata.column_metadata[0].set_name("being");
    expected_metadata.column_metadata[0].child(0).set_name("particulars");
    expected_metadata.column_metadata[0].child(0).child(0).set_name("age");

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }

  {  // Test selecting a non-leaf and expecting all hierarchy from that node onwards
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath))
        .columns({"being.particulars"});
    auto const result = cudf::io::read_parquet(read_args);

    auto expected_weights_col =
      cudf::test::fixed_width_column_wrapper<float>{1.1, 2.4, 5.3, 8.0, 9.6, 6.9};

    auto expected_ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
      {48, 27, 25, 31, 351, 351}, {1, 1, 1, 1, 1, 0}};

    auto expected_s_1 = cudf::test::structs_column_wrapper{
      {expected_weights_col, expected_ages_col}, {1, 1, 1, 1, 0, 1}};

    auto expect_s_2 =
      cudf::test::structs_column_wrapper{{expected_s_1}, {0, 1, 1, 1, 1, 1}}.release();
    auto expected = table_view({*expect_s_2});

    cudf::io::table_input_metadata expected_metadata(expected);
    expected_metadata.column_metadata[0].set_name("being");
    expected_metadata.column_metadata[0].child(0).set_name("particulars");
    expected_metadata.column_metadata[0].child(0).child(0).set_name("weight");
    expected_metadata.column_metadata[0].child(0).child(1).set_name("age");

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }

  {  // Test selecting struct children out of order
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath))
        .columns({"being.particulars.age", "being.particulars.weight", "being.human?"});
    auto const result = cudf::io::read_parquet(read_args);

    auto expected_weights_col =
      cudf::test::fixed_width_column_wrapper<float>{1.1, 2.4, 5.3, 8.0, 9.6, 6.9};

    auto expected_ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
      {48, 27, 25, 31, 351, 351}, {1, 1, 1, 1, 1, 0}};

    auto expected_is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
      {true, true, false, false, false, false}, {1, 1, 0, 1, 1, 0}};

    auto expect_s_1 = cudf::test::structs_column_wrapper{{expected_ages_col, expected_weights_col},
                                                         {1, 1, 1, 1, 0, 1}};

    auto expect_s_2 =
      cudf::test::structs_column_wrapper{{expect_s_1, expected_is_human_col}, {0, 1, 1, 1, 1, 1}}
        .release();

    auto expected = table_view({*expect_s_2});

    cudf::io::table_input_metadata expected_metadata(expected);
    expected_metadata.column_metadata[0].set_name("being");
    expected_metadata.column_metadata[0].child(0).set_name("particulars");
    expected_metadata.column_metadata[0].child(0).child(0).set_name("age");
    expected_metadata.column_metadata[0].child(0).child(1).set_name("weight");
    expected_metadata.column_metadata[0].child(1).set_name("human?");

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }
}

TEST_F(ParquetReaderTest, DecimalRead)
{
  {
    /* We could add a dataset to include this file, but we don't want tests in cudf to have data.
       This test is a temporary test until python gains the ability to write decimal, so we're
       embedding
       a parquet file directly into the code here to prevent issues with finding the file */
    unsigned char const decimals_parquet[] = {
      0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0xb0, 0x03, 0x15, 0xb8, 0x03, 0x2c, 0x15, 0x6a,
      0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1c, 0x36, 0x02, 0x28, 0x04, 0x7f, 0x96, 0x98, 0x00,
      0x18, 0x04, 0x81, 0x69, 0x67, 0xff, 0x00, 0x00, 0x00, 0xd8, 0x01, 0xf0, 0xd7, 0x04, 0x00,
      0x00, 0x00, 0x64, 0x01, 0x03, 0x06, 0x68, 0x12, 0xdc, 0xff, 0xbd, 0x18, 0xfd, 0xff, 0x64,
      0x13, 0x80, 0x00, 0xb3, 0x5d, 0x62, 0x00, 0x90, 0x35, 0xa9, 0xff, 0xa2, 0xde, 0xe3, 0xff,
      0xe9, 0xbf, 0x96, 0xff, 0x1f, 0x8a, 0x98, 0xff, 0xb1, 0x50, 0x34, 0x00, 0x88, 0x24, 0x59,
      0x00, 0x2a, 0x33, 0xbe, 0xff, 0xd5, 0x16, 0xbc, 0xff, 0x13, 0x50, 0x8d, 0xff, 0xcb, 0x63,
      0x2d, 0x00, 0x80, 0x8f, 0xbe, 0xff, 0x82, 0x40, 0x10, 0x00, 0x84, 0x68, 0x70, 0xff, 0x9b,
      0x69, 0x78, 0x00, 0x14, 0x6c, 0x10, 0x00, 0x50, 0xd9, 0xe1, 0xff, 0xaa, 0xcd, 0x6a, 0x00,
      0xcf, 0xb1, 0x28, 0x00, 0x77, 0x57, 0x8d, 0x00, 0xee, 0x05, 0x79, 0x00, 0xf0, 0x15, 0xeb,
      0xff, 0x02, 0xe2, 0x06, 0x00, 0x87, 0x43, 0x86, 0x00, 0xf8, 0x2d, 0x2e, 0x00, 0xee, 0x2e,
      0x98, 0xff, 0x39, 0xcb, 0x4d, 0x00, 0x1e, 0x6b, 0xea, 0xff, 0x80, 0x8e, 0x6c, 0xff, 0x97,
      0x25, 0x26, 0x00, 0x4d, 0x0d, 0x0a, 0x00, 0xca, 0x64, 0x7f, 0x00, 0xf4, 0xbe, 0xa1, 0xff,
      0xe2, 0x12, 0x6c, 0xff, 0xbd, 0x77, 0xae, 0xff, 0xf9, 0x4b, 0x36, 0x00, 0xb0, 0xe3, 0x79,
      0xff, 0xa2, 0x2a, 0x29, 0x00, 0xcd, 0x06, 0xbc, 0xff, 0x2d, 0xa3, 0x7e, 0x00, 0xa9, 0x08,
      0xa1, 0xff, 0xbf, 0x81, 0xd0, 0xff, 0x4f, 0x03, 0x73, 0x00, 0xb0, 0x99, 0x0c, 0x00, 0xbd,
      0x6f, 0xf8, 0xff, 0x6b, 0x02, 0x05, 0x00, 0xc1, 0xe1, 0xba, 0xff, 0x81, 0x69, 0x67, 0xff,
      0x7f, 0x96, 0x98, 0x00, 0x15, 0x00, 0x15, 0xd0, 0x06, 0x15, 0xda, 0x06, 0x2c, 0x15, 0x6a,
      0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1c, 0x36, 0x02, 0x28, 0x08, 0xff, 0x3f, 0x7a, 0x10,
      0xf3, 0x5a, 0x00, 0x00, 0x18, 0x08, 0x01, 0xc0, 0x85, 0xef, 0x0c, 0xa5, 0xff, 0xff, 0x00,
      0x00, 0x00, 0xa8, 0x03, 0xf4, 0xa7, 0x01, 0x04, 0x00, 0x00, 0x00, 0x64, 0x01, 0x03, 0x06,
      0x55, 0x6f, 0xc5, 0xe4, 0x9f, 0x1a, 0x00, 0x00, 0x47, 0x89, 0x0a, 0xe8, 0x58, 0xf0, 0xff,
      0xff, 0x63, 0xee, 0x21, 0xdd, 0xdd, 0xca, 0xff, 0xff, 0xbe, 0x6f, 0x3b, 0xaa, 0xe9, 0x3d,
      0x00, 0x00, 0xd6, 0x91, 0x2a, 0xb7, 0x08, 0x02, 0x00, 0x00, 0x75, 0x45, 0x2c, 0xd7, 0x76,
      0x0c, 0x00, 0x00, 0x54, 0x49, 0x92, 0x44, 0x9c, 0xbf, 0xff, 0xff, 0x41, 0xa9, 0x6d, 0xec,
      0x7a, 0xd0, 0xff, 0xff, 0x27, 0xa0, 0x23, 0x41, 0x44, 0xc1, 0xff, 0xff, 0x18, 0xd4, 0xe1,
      0x30, 0xd3, 0xe0, 0xff, 0xff, 0x59, 0xac, 0x14, 0xf4, 0xec, 0x58, 0x00, 0x00, 0x2c, 0x17,
      0x29, 0x57, 0x44, 0x13, 0x00, 0x00, 0xa2, 0x0d, 0x4a, 0xcc, 0x63, 0xff, 0xff, 0xff, 0x81,
      0x33, 0xbc, 0xda, 0xd5, 0xda, 0xff, 0xff, 0x4c, 0x05, 0xf4, 0x78, 0x19, 0xea, 0xff, 0xff,
      0x06, 0x71, 0x25, 0xde, 0x5a, 0xaf, 0xff, 0xff, 0x95, 0x32, 0x5f, 0x76, 0x98, 0xb3, 0xff,
      0xff, 0xf1, 0x34, 0x3c, 0xbf, 0xa8, 0xbe, 0xff, 0xff, 0x27, 0x73, 0x40, 0x0c, 0x7d, 0xcd,
      0xff, 0xff, 0x68, 0xa9, 0xc2, 0xe9, 0x2c, 0x03, 0x00, 0x00, 0x3f, 0x79, 0xd9, 0x04, 0x8c,
      0xe5, 0xff, 0xff, 0x91, 0xb4, 0x9b, 0xe3, 0x8f, 0x21, 0x00, 0x00, 0xb8, 0x20, 0xc8, 0xc2,
      0x4d, 0xa6, 0xff, 0xff, 0x47, 0xfa, 0xde, 0x36, 0x4a, 0xf3, 0xff, 0xff, 0x72, 0x80, 0x94,
      0x59, 0xdd, 0x4e, 0x00, 0x00, 0x29, 0xe4, 0xd6, 0x43, 0xb0, 0xf0, 0xff, 0xff, 0x68, 0x36,
      0xbc, 0x2d, 0xd1, 0xa9, 0xff, 0xff, 0xbc, 0xe4, 0xbe, 0xd7, 0xed, 0x1b, 0x00, 0x00, 0x02,
      0x8b, 0xcb, 0xd7, 0xed, 0x47, 0x00, 0x00, 0x3c, 0x06, 0xe4, 0xda, 0xc7, 0x47, 0x00, 0x00,
      0xf3, 0x39, 0x55, 0x28, 0x97, 0xba, 0xff, 0xff, 0x07, 0x79, 0x38, 0x4e, 0xe0, 0x21, 0x00,
      0x00, 0xde, 0xed, 0x1c, 0x23, 0x09, 0x49, 0x00, 0x00, 0x49, 0x46, 0x49, 0x5d, 0x8f, 0x34,
      0x00, 0x00, 0x38, 0x18, 0x50, 0xf6, 0xa1, 0x11, 0x00, 0x00, 0xdf, 0xb8, 0x19, 0x14, 0xd1,
      0xe1, 0xff, 0xff, 0x2c, 0x56, 0x72, 0x93, 0x64, 0x3f, 0x00, 0x00, 0x1c, 0xe0, 0xbe, 0x87,
      0x7d, 0xf9, 0xff, 0xff, 0x73, 0x0e, 0x3c, 0x01, 0x91, 0xf9, 0xff, 0xff, 0xb2, 0x37, 0x85,
      0x81, 0x5f, 0x54, 0x00, 0x00, 0x58, 0x44, 0xb0, 0x1a, 0xac, 0xbb, 0xff, 0xff, 0x36, 0xbf,
      0xbe, 0x5e, 0x22, 0xff, 0xff, 0xff, 0x06, 0x20, 0xa0, 0x23, 0x0d, 0x3b, 0x00, 0x00, 0x19,
      0xc6, 0x49, 0x0a, 0x00, 0xcf, 0xff, 0xff, 0x4f, 0xcd, 0xc6, 0x95, 0x4b, 0xf1, 0xff, 0xff,
      0xa3, 0x59, 0xaf, 0x65, 0xec, 0xe9, 0xff, 0xff, 0x58, 0xef, 0x05, 0x50, 0x63, 0xe4, 0xff,
      0xff, 0xc7, 0x6a, 0x9e, 0xf1, 0x69, 0x20, 0x00, 0x00, 0xd1, 0xb3, 0xc9, 0x14, 0xb2, 0x29,
      0x00, 0x00, 0x1d, 0x48, 0x16, 0x70, 0xf0, 0x40, 0x00, 0x00, 0x01, 0xc0, 0x85, 0xef, 0x0c,
      0xa5, 0xff, 0xff, 0xff, 0x3f, 0x7a, 0x10, 0xf3, 0x5a, 0x00, 0x00, 0x15, 0x00, 0x15, 0x90,
      0x0d, 0x15, 0x9a, 0x0d, 0x2c, 0x15, 0x6a, 0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1c, 0x36,
      0x02, 0x28, 0x10, 0x4b, 0x3b, 0x4c, 0xa8, 0x5a, 0x86, 0xc4, 0x7a, 0x09, 0x8a, 0x22, 0x3f,
      0xff, 0xff, 0xff, 0xff, 0x18, 0x10, 0xb4, 0xc4, 0xb3, 0x57, 0xa5, 0x79, 0x3b, 0x85, 0xf6,
      0x75, 0xdd, 0xc0, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xc8, 0x06, 0xf4, 0x47, 0x03,
      0x04, 0x00, 0x00, 0x00, 0x64, 0x01, 0x03, 0x06, 0x05, 0x49, 0xf7, 0xfc, 0x89, 0x3d, 0x3e,
      0x20, 0x07, 0x72, 0x3e, 0xa1, 0x66, 0x81, 0x67, 0x80, 0x23, 0x78, 0x06, 0x68, 0x0e, 0x78,
      0xf5, 0x08, 0xed, 0x20, 0xcd, 0x0e, 0x7f, 0x9c, 0x70, 0xa0, 0xb9, 0x16, 0x44, 0xb2, 0x41,
      0x62, 0xba, 0x82, 0xad, 0xe1, 0x12, 0x9b, 0xa6, 0x53, 0x8d, 0x20, 0x27, 0xd5, 0x84, 0x63,
      0xb8, 0x07, 0x4b, 0x5b, 0xa4, 0x1c, 0xa4, 0x1c, 0x17, 0xbf, 0x4b, 0x00, 0x24, 0x04, 0x56,
      0xa8, 0x52, 0xaf, 0x33, 0xf7, 0xad, 0x7c, 0xc8, 0x83, 0x25, 0x13, 0xaf, 0x80, 0x25, 0x6f,
      0xbd, 0xd1, 0x15, 0x69, 0x64, 0x20, 0x7b, 0xd7, 0x33, 0xba, 0x66, 0x29, 0x8a, 0x00, 0xda,
      0x42, 0x07, 0x2c, 0x6c, 0x39, 0x76, 0x9f, 0xdc, 0x17, 0xad, 0xb6, 0x58, 0xdf, 0x5f, 0x00,
      0x18, 0x3a, 0xae, 0x1c, 0xd6, 0x5f, 0x9d, 0x78, 0x8d, 0x73, 0xdd, 0x3e, 0xd6, 0x18, 0x33,
      0x40, 0xe4, 0x36, 0xde, 0xb0, 0xb7, 0x33, 0x2a, 0x6b, 0x08, 0x03, 0x6c, 0x6d, 0x8f, 0x13,
      0x93, 0xd0, 0xd7, 0x87, 0x62, 0x63, 0x53, 0xfb, 0xd8, 0xbb, 0xc9, 0x54, 0x90, 0xd6, 0xa9,
      0x8f, 0xc8, 0x60, 0xbd, 0xec, 0x75, 0x23, 0x9a, 0x21, 0xec, 0xe4, 0x86, 0x43, 0xd7, 0xc1,
      0x88, 0xdc, 0x82, 0x00, 0x32, 0x79, 0xc9, 0x2b, 0x70, 0x85, 0xb7, 0x25, 0xa1, 0xcc, 0x7d,
      0x0b, 0x29, 0x03, 0xea, 0x80, 0xff, 0x9b, 0xf3, 0x24, 0x7f, 0xd1, 0xff, 0xf0, 0x22, 0x65,
      0x85, 0x99, 0x17, 0x63, 0xc2, 0xc0, 0xb7, 0x62, 0x05, 0xda, 0x7a, 0xa0, 0xc3, 0x2a, 0x6f,
      0x1f, 0xee, 0x1f, 0x31, 0xa8, 0x42, 0x80, 0xe4, 0xb7, 0x6c, 0xf6, 0xac, 0x47, 0xb0, 0x17,
      0x69, 0xcb, 0xff, 0x66, 0x8a, 0xd6, 0x25, 0x00, 0xf3, 0xcf, 0x0a, 0xaf, 0xf8, 0x92, 0x8a,
      0xa0, 0xdf, 0x71, 0x13, 0x8d, 0x9d, 0xff, 0x7e, 0xe0, 0x0a, 0x52, 0xf1, 0x97, 0x01, 0xa9,
      0x73, 0x27, 0xfd, 0x63, 0x58, 0x00, 0x32, 0xa6, 0xf6, 0x78, 0xb8, 0xe4, 0xfd, 0x20, 0x7c,
      0x90, 0xee, 0xad, 0x8c, 0xc9, 0x71, 0x35, 0x66, 0x71, 0x3c, 0xe0, 0xe4, 0x0b, 0xbb, 0xa0,
      0x50, 0xe9, 0xf2, 0x81, 0x1d, 0x3a, 0x95, 0x94, 0x00, 0xd5, 0x49, 0x00, 0x07, 0xdf, 0x21,
      0x53, 0x36, 0x8d, 0x9e, 0xd9, 0xa5, 0x52, 0x4d, 0x0d, 0x29, 0x74, 0xf0, 0x40, 0xbd, 0xda,
      0x63, 0x4e, 0xdd, 0x91, 0x8e, 0xa6, 0xa7, 0xf6, 0x78, 0x58, 0x3b, 0x0a, 0x5c, 0x60, 0x3c,
      0x15, 0x34, 0xf8, 0x2c, 0x21, 0xe3, 0x56, 0x1b, 0x9e, 0xd9, 0x56, 0xd3, 0x13, 0x2e, 0x80,
      0x2c, 0x36, 0xda, 0x1d, 0xc8, 0xfb, 0x52, 0xee, 0x17, 0xb3, 0x2b, 0xf3, 0xd2, 0xeb, 0x29,
      0xa0, 0x37, 0xa0, 0x12, 0xce, 0x1c, 0x50, 0x6a, 0xf4, 0x11, 0xcd, 0x96, 0x88, 0x3f, 0x43,
      0x78, 0xc0, 0x2c, 0x53, 0x6c, 0xa6, 0xdf, 0xb9, 0x9e, 0x93, 0xd4, 0x1e, 0xa9, 0x7f, 0x67,
      0xa6, 0xc1, 0x80, 0x46, 0x0f, 0x63, 0x7d, 0x15, 0xf2, 0x4c, 0xc5, 0xda, 0x11, 0x9a, 0x20,
      0x67, 0x27, 0xe8, 0x00, 0xec, 0x03, 0x1d, 0x15, 0xa7, 0x92, 0xb3, 0x1f, 0xda, 0x20, 0x92,
      0xd8, 0x00, 0xfb, 0x06, 0x80, 0xeb, 0x4b, 0x0c, 0xc1, 0x1f, 0x49, 0x40, 0x06, 0x8d, 0x8a,
      0xf8, 0x34, 0xb1, 0x0c, 0x1d, 0x20, 0xd0, 0x47, 0xe5, 0xb1, 0x7e, 0xf7, 0xe4, 0xb4, 0x7e,
      0x9c, 0x84, 0x18, 0x61, 0x32, 0x4f, 0xc0, 0xc2, 0xb2, 0xcc, 0x63, 0xf6, 0xe1, 0x16, 0xd6,
      0xd9, 0x4b, 0x74, 0x13, 0x01, 0xa1, 0xe2, 0x00, 0xb7, 0x9e, 0xc1, 0x3a, 0xc5, 0xaf, 0xe8,
      0x54, 0x07, 0x2a, 0x20, 0xfd, 0x2c, 0x6f, 0xb9, 0x80, 0x18, 0x92, 0x87, 0xa0, 0x81, 0x24,
      0x60, 0x47, 0x17, 0x4f, 0xbc, 0xbe, 0xf5, 0x03, 0x69, 0x80, 0xe3, 0x10, 0x54, 0xd6, 0x68,
      0x7d, 0x75, 0xd3, 0x0a, 0x45, 0x38, 0x9e, 0xa9, 0xfd, 0x05, 0x40, 0xd2, 0x1e, 0x6f, 0x5c,
      0x30, 0x10, 0xfe, 0x9b, 0x9f, 0x6d, 0xc0, 0x9d, 0x6c, 0x17, 0x7d, 0x00, 0x09, 0xb6, 0x8a,
      0x31, 0x8e, 0x1b, 0x6b, 0x84, 0x1e, 0x79, 0xce, 0x10, 0x55, 0x59, 0x6a, 0x40, 0x16, 0xdc,
      0x9a, 0xcf, 0x4d, 0xb0, 0x8f, 0xac, 0xe3, 0x8d, 0xee, 0xd2, 0xef, 0x01, 0x8c, 0xe0, 0x2b,
      0x24, 0xe5, 0xb4, 0xe1, 0x86, 0x72, 0x00, 0x30, 0x07, 0xce, 0x02, 0x23, 0x41, 0x33, 0x40,
      0xf0, 0x9b, 0xc2, 0x2d, 0x30, 0xec, 0x3b, 0x17, 0xb2, 0x8f, 0x64, 0x7d, 0xcd, 0x70, 0x9e,
      0x80, 0x22, 0xb5, 0xdf, 0x6d, 0x2a, 0x43, 0xd4, 0x2b, 0x5a, 0xf6, 0x96, 0xa6, 0xea, 0x91,
      0x62, 0x80, 0x39, 0xf2, 0x5a, 0x8e, 0xc0, 0xb9, 0x29, 0x99, 0x17, 0xe7, 0x35, 0x2c, 0xf6,
      0x4d, 0x18, 0x00, 0x48, 0x10, 0x85, 0xb4, 0x3f, 0x89, 0x60, 0x49, 0x6e, 0xf0, 0xcd, 0x9d,
      0x92, 0xeb, 0x96, 0x80, 0xcf, 0xf9, 0xf1, 0x46, 0x1d, 0xc0, 0x49, 0xb3, 0x36, 0x2e, 0x24,
      0xc8, 0xdb, 0x41, 0x72, 0x20, 0xf5, 0xde, 0x5c, 0xf9, 0x4a, 0x6e, 0xa0, 0x0b, 0x13, 0xfc,
      0x2d, 0x17, 0x07, 0x16, 0x5e, 0x00, 0x3c, 0x54, 0x41, 0x0e, 0xa2, 0x0d, 0xf3, 0x48, 0x12,
      0x2e, 0x7c, 0xab, 0x3c, 0x59, 0x1c, 0x40, 0xca, 0xb0, 0x71, 0xc7, 0x29, 0xf0, 0xbb, 0x9f,
      0xf4, 0x3f, 0x25, 0x49, 0xad, 0xc2, 0x8f, 0x80, 0x04, 0x38, 0x6d, 0x35, 0x02, 0xca, 0xe6,
      0x02, 0x83, 0x89, 0x4e, 0x74, 0xdb, 0x08, 0x5a, 0x80, 0x13, 0x99, 0xd4, 0x26, 0xc1, 0x27,
      0xce, 0xb0, 0x98, 0x99, 0xca, 0xf6, 0x3e, 0x50, 0x49, 0xd0, 0xbf, 0xcb, 0x6f, 0xbe, 0x5b,
      0x92, 0x63, 0xde, 0x94, 0xd3, 0x8f, 0x07, 0x06, 0x0f, 0x2b, 0x80, 0x36, 0xf1, 0x77, 0xf6,
      0x29, 0x33, 0x13, 0xa9, 0x4a, 0x55, 0x3d, 0x6c, 0xca, 0xdb, 0x4e, 0x40, 0xc4, 0x95, 0x54,
      0xf4, 0xe2, 0x8c, 0x1b, 0xa0, 0xfe, 0x30, 0x50, 0x9d, 0x62, 0xbc, 0x5c, 0x00, 0xb4, 0xc4,
      0xb3, 0x57, 0xa5, 0x79, 0x3b, 0x85, 0xf6, 0x75, 0xdd, 0xc0, 0x00, 0x00, 0x00, 0x01, 0x4b,
      0x3b, 0x4c, 0xa8, 0x5a, 0x86, 0xc4, 0x7a, 0x09, 0x8a, 0x22, 0x3f, 0xff, 0xff, 0xff, 0xff,
      0x15, 0x02, 0x19, 0x4c, 0x48, 0x0c, 0x73, 0x70, 0x61, 0x72, 0x6b, 0x5f, 0x73, 0x63, 0x68,
      0x65, 0x6d, 0x61, 0x15, 0x06, 0x00, 0x15, 0x02, 0x25, 0x02, 0x18, 0x06, 0x64, 0x65, 0x63,
      0x37, 0x70, 0x34, 0x25, 0x0a, 0x15, 0x08, 0x15, 0x0e, 0x00, 0x15, 0x04, 0x25, 0x02, 0x18,
      0x07, 0x64, 0x65, 0x63, 0x31, 0x34, 0x70, 0x35, 0x25, 0x0a, 0x15, 0x0a, 0x15, 0x1c, 0x00,
      0x15, 0x0e, 0x15, 0x20, 0x15, 0x02, 0x18, 0x08, 0x64, 0x65, 0x63, 0x33, 0x38, 0x70, 0x31,
      0x38, 0x25, 0x0a, 0x15, 0x24, 0x15, 0x4c, 0x00, 0x16, 0x6a, 0x19, 0x1c, 0x19, 0x3c, 0x26,
      0x08, 0x1c, 0x15, 0x02, 0x19, 0x35, 0x06, 0x08, 0x00, 0x19, 0x18, 0x06, 0x64, 0x65, 0x63,
      0x37, 0x70, 0x34, 0x15, 0x02, 0x16, 0x6a, 0x16, 0xf6, 0x03, 0x16, 0xfe, 0x03, 0x26, 0x08,
      0x3c, 0x36, 0x02, 0x28, 0x04, 0x7f, 0x96, 0x98, 0x00, 0x18, 0x04, 0x81, 0x69, 0x67, 0xff,
      0x00, 0x19, 0x1c, 0x15, 0x00, 0x15, 0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x26, 0x86, 0x04,
      0x1c, 0x15, 0x04, 0x19, 0x35, 0x06, 0x08, 0x00, 0x19, 0x18, 0x07, 0x64, 0x65, 0x63, 0x31,
      0x34, 0x70, 0x35, 0x15, 0x02, 0x16, 0x6a, 0x16, 0xa6, 0x07, 0x16, 0xb0, 0x07, 0x26, 0x86,
      0x04, 0x3c, 0x36, 0x02, 0x28, 0x08, 0xff, 0x3f, 0x7a, 0x10, 0xf3, 0x5a, 0x00, 0x00, 0x18,
      0x08, 0x01, 0xc0, 0x85, 0xef, 0x0c, 0xa5, 0xff, 0xff, 0x00, 0x19, 0x1c, 0x15, 0x00, 0x15,
      0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x26, 0xb6, 0x0b, 0x1c, 0x15, 0x0e, 0x19, 0x35, 0x06,
      0x08, 0x00, 0x19, 0x18, 0x08, 0x64, 0x65, 0x63, 0x33, 0x38, 0x70, 0x31, 0x38, 0x15, 0x02,
      0x16, 0x6a, 0x16, 0x86, 0x0e, 0x16, 0x90, 0x0e, 0x26, 0xb6, 0x0b, 0x3c, 0x36, 0x02, 0x28,
      0x10, 0x4b, 0x3b, 0x4c, 0xa8, 0x5a, 0x86, 0xc4, 0x7a, 0x09, 0x8a, 0x22, 0x3f, 0xff, 0xff,
      0xff, 0xff, 0x18, 0x10, 0xb4, 0xc4, 0xb3, 0x57, 0xa5, 0x79, 0x3b, 0x85, 0xf6, 0x75, 0xdd,
      0xc0, 0x00, 0x00, 0x00, 0x01, 0x00, 0x19, 0x1c, 0x15, 0x00, 0x15, 0x00, 0x15, 0x02, 0x00,
      0x00, 0x00, 0x16, 0xa2, 0x19, 0x16, 0x6a, 0x00, 0x19, 0x2c, 0x18, 0x18, 0x6f, 0x72, 0x67,
      0x2e, 0x61, 0x70, 0x61, 0x63, 0x68, 0x65, 0x2e, 0x73, 0x70, 0x61, 0x72, 0x6b, 0x2e, 0x76,
      0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x18, 0x05, 0x33, 0x2e, 0x30, 0x2e, 0x31, 0x00, 0x18,
      0x29, 0x6f, 0x72, 0x67, 0x2e, 0x61, 0x70, 0x61, 0x63, 0x68, 0x65, 0x2e, 0x73, 0x70, 0x61,
      0x72, 0x6b, 0x2e, 0x73, 0x71, 0x6c, 0x2e, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2e,
      0x72, 0x6f, 0x77, 0x2e, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x18, 0xf4, 0x01,
      0x7b, 0x22, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3a, 0x22, 0x73, 0x74, 0x72, 0x75, 0x63, 0x74,
      0x22, 0x2c, 0x22, 0x66, 0x69, 0x65, 0x6c, 0x64, 0x73, 0x22, 0x3a, 0x5b, 0x7b, 0x22, 0x6e,
      0x61, 0x6d, 0x65, 0x22, 0x3a, 0x22, 0x64, 0x65, 0x63, 0x37, 0x70, 0x34, 0x22, 0x2c, 0x22,
      0x74, 0x79, 0x70, 0x65, 0x22, 0x3a, 0x22, 0x64, 0x65, 0x63, 0x69, 0x6d, 0x61, 0x6c, 0x28,
      0x37, 0x2c, 0x34, 0x29, 0x22, 0x2c, 0x22, 0x6e, 0x75, 0x6c, 0x6c, 0x61, 0x62, 0x6c, 0x65,
      0x22, 0x3a, 0x74, 0x72, 0x75, 0x65, 0x2c, 0x22, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74,
      0x61, 0x22, 0x3a, 0x7b, 0x7d, 0x7d, 0x2c, 0x7b, 0x22, 0x6e, 0x61, 0x6d, 0x65, 0x22, 0x3a,
      0x22, 0x64, 0x65, 0x63, 0x31, 0x34, 0x70, 0x35, 0x22, 0x2c, 0x22, 0x74, 0x79, 0x70, 0x65,
      0x22, 0x3a, 0x22, 0x64, 0x65, 0x63, 0x69, 0x6d, 0x61, 0x6c, 0x28, 0x31, 0x34, 0x2c, 0x35,
      0x29, 0x22, 0x2c, 0x22, 0x6e, 0x75, 0x6c, 0x6c, 0x61, 0x62, 0x6c, 0x65, 0x22, 0x3a, 0x74,
      0x72, 0x75, 0x65, 0x2c, 0x22, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x22, 0x3a,
      0x7b, 0x7d, 0x7d, 0x2c, 0x7b, 0x22, 0x6e, 0x61, 0x6d, 0x65, 0x22, 0x3a, 0x22, 0x64, 0x65,
      0x63, 0x33, 0x38, 0x70, 0x31, 0x38, 0x22, 0x2c, 0x22, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3a,
      0x22, 0x64, 0x65, 0x63, 0x69, 0x6d, 0x61, 0x6c, 0x28, 0x33, 0x38, 0x2c, 0x31, 0x38, 0x29,
      0x22, 0x2c, 0x22, 0x6e, 0x75, 0x6c, 0x6c, 0x61, 0x62, 0x6c, 0x65, 0x22, 0x3a, 0x74, 0x72,
      0x75, 0x65, 0x2c, 0x22, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x22, 0x3a, 0x7b,
      0x7d, 0x7d, 0x5d, 0x7d, 0x00, 0x18, 0x4a, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2d,
      0x6d, 0x72, 0x20, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x20, 0x31, 0x2e, 0x31, 0x30,
      0x2e, 0x31, 0x20, 0x28, 0x62, 0x75, 0x69, 0x6c, 0x64, 0x20, 0x61, 0x38, 0x39, 0x64, 0x66,
      0x38, 0x66, 0x39, 0x39, 0x33, 0x32, 0x62, 0x36, 0x65, 0x66, 0x36, 0x36, 0x33, 0x33, 0x64,
      0x30, 0x36, 0x30, 0x36, 0x39, 0x65, 0x35, 0x30, 0x63, 0x39, 0x62, 0x37, 0x39, 0x37, 0x30,
      0x62, 0x65, 0x62, 0x64, 0x31, 0x29, 0x19, 0x3c, 0x1c, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x1c,
      0x00, 0x00, 0x00, 0xd3, 0x02, 0x00, 0x00, 0x50, 0x41, 0x52, 0x31};
    unsigned int decimals_parquet_len = 2366;

    cudf::io::parquet_reader_options read_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{reinterpret_cast<char const*>(decimals_parquet), decimals_parquet_len});
    auto result = cudf::io::read_parquet(read_opts);

    auto validity =
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 50; });

    EXPECT_EQ(result.tbl->view().num_columns(), 3);

    int32_t col0_data[] = {
      -2354584, -190275,  8393572,  6446515,  -5687920, -1843550, -6897687, -6780385, 3428529,
      5842056,  -4312278, -4450603, -7516141, 2974667,  -4288640, 1065090,  -9410428, 7891355,
      1076244,  -1975984, 6999466,  2666959,  9262967,  7931374,  -1370640, 451074,   8799111,
      3026424,  -6803730, 5098297,  -1414370, -9662848, 2499991,  658765,   8348874,  -6177036,
      -9694494, -5343299, 3558393,  -8789072, 2697890,  -4454707, 8299309,  -6223703, -3112513,
      7537487,  825776,   -495683,  328299,   -4529727, 0,        -9999999, 9999999};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(0).size()),
              sizeof(col0_data) / sizeof(col0_data[0]));
    cudf::test::fixed_point_column_wrapper<int32_t> col0(
      std::begin(col0_data), std::end(col0_data), validity, numeric::scale_type{-4});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), col0);

    int64_t col1_data[] = {29274040266581,  -17210335917753, -58420730139037,
                           68073792696254,  2236456014294,   13704555677045,
                           -70797090469548, -52248605513407, -68976081919961,
                           -34277313883112, 97774730521689,  21184241014572,
                           -670882460254,   -40862944054399, -24079852370612,
                           -88670167797498, -84007574359403, -71843004533519,
                           -55538016554201, 3491435293032,   -29085437167297,
                           36901882672273,  -98622066122568, -13974902998457,
                           86712597643378,  -16835133643735, -94759096142232,
                           30708340810940,  79086853262082,  78923696440892,
                           -76316597208589, 37247268714759,  80303592631774,
                           57790350050889,  19387319851064,  -33186875066145,
                           69701203023404,  -7157433049060,  -7073790423437,
                           92769171617714,  -75127120182184, -951893180618,
                           64927618310150,  -53875897154023, -16168039035569,
                           -24273449166429, -30359781249192, 35639397345991,
                           45844829680593,  71401416837149,  0,
                           -99999999999999, 99999999999999};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(1).size()),
              sizeof(col1_data) / sizeof(col1_data[0]));
    cudf::test::fixed_point_column_wrapper<int64_t> col1(
      std::begin(col1_data), std::end(col1_data), validity, numeric::scale_type{-5});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), col1);

    cudf::io::parquet_reader_options read_strict_opts = read_opts;
    read_strict_opts.set_columns({"dec7p4", "dec14p5"});
    EXPECT_NO_THROW(cudf::io::read_parquet(read_strict_opts));
  }
  {
    // dec7p3: Decimal(precision=7, scale=3) backed by FIXED_LENGTH_BYTE_ARRAY(length = 4)
    // dec12p11: Decimal(precision=12, scale=11) backed by FIXED_LENGTH_BYTE_ARRAY(length = 6)
    // dec20p1: Decimal(precision=20, scale=1) backed by FIXED_LENGTH_BYTE_ARRAY(length = 9)
    unsigned char const fixed_len_bytes_decimal_parquet[] = {
      0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0xA8, 0x01, 0x15, 0xAE, 0x01, 0x2C, 0x15, 0x28,
      0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1C, 0x36, 0x02, 0x28, 0x04, 0x00, 0x97, 0x45, 0x72,
      0x18, 0x04, 0x00, 0x01, 0x81, 0x3B, 0x00, 0x00, 0x00, 0x54, 0xF0, 0x53, 0x04, 0x00, 0x00,
      0x00, 0x26, 0x01, 0x03, 0x00, 0x00, 0x61, 0x10, 0xCF, 0x00, 0x0A, 0xA9, 0x08, 0x00, 0x77,
      0x58, 0x6F, 0x00, 0x6B, 0xEE, 0xA4, 0x00, 0x92, 0xF8, 0x94, 0x00, 0x2E, 0x18, 0xD4, 0x00,
      0x4F, 0x45, 0x33, 0x00, 0x97, 0x45, 0x72, 0x00, 0x0D, 0xC2, 0x75, 0x00, 0x76, 0xAA, 0xAA,
      0x00, 0x30, 0x9F, 0x86, 0x00, 0x4B, 0x9D, 0xB1, 0x00, 0x4E, 0x4B, 0x3B, 0x00, 0x01, 0x81,
      0x3B, 0x00, 0x22, 0xD4, 0x53, 0x00, 0x72, 0xC4, 0xAF, 0x00, 0x43, 0x9B, 0x72, 0x00, 0x1D,
      0x91, 0xC3, 0x00, 0x45, 0x27, 0x48, 0x15, 0x00, 0x15, 0xF4, 0x01, 0x15, 0xFA, 0x01, 0x2C,
      0x15, 0x28, 0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1C, 0x36, 0x02, 0x28, 0x06, 0x00, 0xD5,
      0xD7, 0x31, 0x99, 0xA6, 0x18, 0x06, 0xFF, 0x17, 0x2B, 0x5A, 0xF0, 0x01, 0x00, 0x00, 0x00,
      0x7A, 0xF0, 0x79, 0x04, 0x00, 0x00, 0x00, 0x24, 0x01, 0x03, 0x02, 0x00, 0x54, 0x23, 0xCF,
      0x13, 0x0A, 0x00, 0x07, 0x22, 0xB1, 0x21, 0x7E, 0x00, 0x64, 0x19, 0xD6, 0xD2, 0xA5, 0x00,
      0x61, 0x7F, 0xF6, 0xB9, 0xB0, 0x00, 0xD0, 0x7F, 0x9C, 0xA9, 0xE9, 0x00, 0x65, 0x58, 0xF0,
      0xAD, 0xFB, 0x00, 0xBC, 0x61, 0xE2, 0x03, 0xDA, 0xFF, 0x17, 0x2B, 0x5A, 0xF0, 0x01, 0x00,
      0x63, 0x4B, 0x4C, 0xFE, 0x45, 0x00, 0x7A, 0xA0, 0xD8, 0xD1, 0xC0, 0x00, 0xC0, 0x63, 0xF7,
      0x9D, 0x0A, 0x00, 0x88, 0x22, 0x0F, 0x1B, 0x25, 0x00, 0x1A, 0x80, 0x56, 0x34, 0xC7, 0x00,
      0x5F, 0x48, 0x61, 0x09, 0x7C, 0x00, 0x61, 0xEF, 0x92, 0x42, 0x2F, 0x00, 0xD5, 0xD7, 0x31,
      0x99, 0xA6, 0xFF, 0x17, 0x2B, 0x5A, 0xF0, 0x01, 0x00, 0x71, 0xDD, 0xE2, 0x22, 0x7B, 0x00,
      0x54, 0xBF, 0xAE, 0xE9, 0x3C, 0x15, 0x00, 0x15, 0xD4, 0x02, 0x15, 0xDC, 0x02, 0x2C, 0x15,
      0x28, 0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1C, 0x36, 0x04, 0x28, 0x09, 0x00, 0x7D, 0xFE,
      0x02, 0xDA, 0xB2, 0x62, 0xA3, 0xFB, 0x18, 0x09, 0x00, 0x03, 0x9C, 0xCD, 0x5A, 0xAC, 0xBB,
      0xF1, 0xE3, 0x00, 0x00, 0x00, 0xAA, 0x01, 0xF0, 0xA9, 0x04, 0x00, 0x00, 0x00, 0x07, 0xBF,
      0xBF, 0x0F, 0x00, 0x7D, 0xFE, 0x02, 0xDA, 0xB2, 0x62, 0xA3, 0xFB, 0x00, 0x7D, 0x9A, 0xCB,
      0xDA, 0x4B, 0x10, 0x8B, 0xAC, 0x00, 0x20, 0xBA, 0x97, 0x87, 0x2E, 0x3B, 0x4E, 0x04, 0x00,
      0x15, 0xBB, 0xC2, 0xDF, 0x2D, 0x25, 0x08, 0xB6, 0x00, 0x5C, 0x67, 0x0E, 0x36, 0x30, 0xF1,
      0xAC, 0xA4, 0x00, 0x44, 0xF1, 0x8E, 0xFB, 0x17, 0x5E, 0xE1, 0x96, 0x00, 0x64, 0x69, 0xF9,
      0x66, 0x3F, 0x11, 0xED, 0xB9, 0x00, 0x45, 0xB5, 0xDA, 0x14, 0x9C, 0xA3, 0xFA, 0x64, 0x00,
      0x26, 0x5F, 0xDE, 0xD7, 0x67, 0x95, 0xEF, 0xB1, 0x00, 0x35, 0xDB, 0x9B, 0x88, 0x46, 0xD0,
      0xA1, 0x0E, 0x00, 0x45, 0xA9, 0x92, 0x8E, 0x89, 0xD1, 0xAC, 0x4C, 0x00, 0x4C, 0xF1, 0xCB,
      0x27, 0x82, 0x3A, 0x7D, 0xB7, 0x00, 0x64, 0xD3, 0xD2, 0x2F, 0x9C, 0x83, 0x16, 0x75, 0x00,
      0x15, 0xDF, 0xC2, 0xA9, 0x63, 0xB8, 0x33, 0x65, 0x00, 0x27, 0x40, 0x28, 0x97, 0x05, 0x8E,
      0xE3, 0x46, 0x00, 0x03, 0x9C, 0xCD, 0x5A, 0xAC, 0xBB, 0xF1, 0xE3, 0x00, 0x22, 0x23, 0xF5,
      0xE8, 0x9D, 0x55, 0xD4, 0x9C, 0x00, 0x25, 0xB9, 0xD8, 0x87, 0x2D, 0xF1, 0xF2, 0x17, 0x15,
      0x02, 0x19, 0x4C, 0x48, 0x0C, 0x73, 0x70, 0x61, 0x72, 0x6B, 0x5F, 0x73, 0x63, 0x68, 0x65,
      0x6D, 0x61, 0x15, 0x06, 0x00, 0x15, 0x0E, 0x15, 0x08, 0x15, 0x02, 0x18, 0x06, 0x64, 0x65,
      0x63, 0x37, 0x70, 0x33, 0x25, 0x0A, 0x15, 0x06, 0x15, 0x0E, 0x00, 0x15, 0x0E, 0x15, 0x0C,
      0x15, 0x02, 0x18, 0x08, 0x64, 0x65, 0x63, 0x31, 0x32, 0x70, 0x31, 0x31, 0x25, 0x0A, 0x15,
      0x16, 0x15, 0x18, 0x00, 0x15, 0x0E, 0x15, 0x12, 0x15, 0x02, 0x18, 0x07, 0x64, 0x65, 0x63,
      0x32, 0x30, 0x70, 0x31, 0x25, 0x0A, 0x15, 0x02, 0x15, 0x28, 0x00, 0x16, 0x28, 0x19, 0x1C,
      0x19, 0x3C, 0x26, 0x08, 0x1C, 0x15, 0x0E, 0x19, 0x35, 0x06, 0x08, 0x00, 0x19, 0x18, 0x06,
      0x64, 0x65, 0x63, 0x37, 0x70, 0x33, 0x15, 0x02, 0x16, 0x28, 0x16, 0xEE, 0x01, 0x16, 0xF4,
      0x01, 0x26, 0x08, 0x3C, 0x36, 0x02, 0x28, 0x04, 0x00, 0x97, 0x45, 0x72, 0x18, 0x04, 0x00,
      0x01, 0x81, 0x3B, 0x00, 0x19, 0x1C, 0x15, 0x00, 0x15, 0x00, 0x15, 0x02, 0x00, 0x00, 0x00,
      0x26, 0xFC, 0x01, 0x1C, 0x15, 0x0E, 0x19, 0x35, 0x06, 0x08, 0x00, 0x19, 0x18, 0x08, 0x64,
      0x65, 0x63, 0x31, 0x32, 0x70, 0x31, 0x31, 0x15, 0x02, 0x16, 0x28, 0x16, 0xC2, 0x02, 0x16,
      0xC8, 0x02, 0x26, 0xFC, 0x01, 0x3C, 0x36, 0x02, 0x28, 0x06, 0x00, 0xD5, 0xD7, 0x31, 0x99,
      0xA6, 0x18, 0x06, 0xFF, 0x17, 0x2B, 0x5A, 0xF0, 0x01, 0x00, 0x19, 0x1C, 0x15, 0x00, 0x15,
      0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x26, 0xC4, 0x04, 0x1C, 0x15, 0x0E, 0x19, 0x35, 0x06,
      0x08, 0x00, 0x19, 0x18, 0x07, 0x64, 0x65, 0x63, 0x32, 0x30, 0x70, 0x31, 0x15, 0x02, 0x16,
      0x28, 0x16, 0xAE, 0x03, 0x16, 0xB6, 0x03, 0x26, 0xC4, 0x04, 0x3C, 0x36, 0x04, 0x28, 0x09,
      0x00, 0x7D, 0xFE, 0x02, 0xDA, 0xB2, 0x62, 0xA3, 0xFB, 0x18, 0x09, 0x00, 0x03, 0x9C, 0xCD,
      0x5A, 0xAC, 0xBB, 0xF1, 0xE3, 0x00, 0x19, 0x1C, 0x15, 0x00, 0x15, 0x00, 0x15, 0x02, 0x00,
      0x00, 0x00, 0x16, 0xDE, 0x07, 0x16, 0x28, 0x00, 0x19, 0x2C, 0x18, 0x18, 0x6F, 0x72, 0x67,
      0x2E, 0x61, 0x70, 0x61, 0x63, 0x68, 0x65, 0x2E, 0x73, 0x70, 0x61, 0x72, 0x6B, 0x2E, 0x76,
      0x65, 0x72, 0x73, 0x69, 0x6F, 0x6E, 0x18, 0x05, 0x33, 0x2E, 0x30, 0x2E, 0x31, 0x00, 0x18,
      0x29, 0x6F, 0x72, 0x67, 0x2E, 0x61, 0x70, 0x61, 0x63, 0x68, 0x65, 0x2E, 0x73, 0x70, 0x61,
      0x72, 0x6B, 0x2E, 0x73, 0x71, 0x6C, 0x2E, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2E,
      0x72, 0x6F, 0x77, 0x2E, 0x6D, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x18, 0xF4, 0x01,
      0x7B, 0x22, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3A, 0x22, 0x73, 0x74, 0x72, 0x75, 0x63, 0x74,
      0x22, 0x2C, 0x22, 0x66, 0x69, 0x65, 0x6C, 0x64, 0x73, 0x22, 0x3A, 0x5B, 0x7B, 0x22, 0x6E,
      0x61, 0x6D, 0x65, 0x22, 0x3A, 0x22, 0x64, 0x65, 0x63, 0x37, 0x70, 0x33, 0x22, 0x2C, 0x22,
      0x74, 0x79, 0x70, 0x65, 0x22, 0x3A, 0x22, 0x64, 0x65, 0x63, 0x69, 0x6D, 0x61, 0x6C, 0x28,
      0x37, 0x2C, 0x33, 0x29, 0x22, 0x2C, 0x22, 0x6E, 0x75, 0x6C, 0x6C, 0x61, 0x62, 0x6C, 0x65,
      0x22, 0x3A, 0x74, 0x72, 0x75, 0x65, 0x2C, 0x22, 0x6D, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74,
      0x61, 0x22, 0x3A, 0x7B, 0x7D, 0x7D, 0x2C, 0x7B, 0x22, 0x6E, 0x61, 0x6D, 0x65, 0x22, 0x3A,
      0x22, 0x64, 0x65, 0x63, 0x31, 0x32, 0x70, 0x31, 0x31, 0x22, 0x2C, 0x22, 0x74, 0x79, 0x70,
      0x65, 0x22, 0x3A, 0x22, 0x64, 0x65, 0x63, 0x69, 0x6D, 0x61, 0x6C, 0x28, 0x31, 0x32, 0x2C,
      0x31, 0x31, 0x29, 0x22, 0x2C, 0x22, 0x6E, 0x75, 0x6C, 0x6C, 0x61, 0x62, 0x6C, 0x65, 0x22,
      0x3A, 0x74, 0x72, 0x75, 0x65, 0x2C, 0x22, 0x6D, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61,
      0x22, 0x3A, 0x7B, 0x7D, 0x7D, 0x2C, 0x7B, 0x22, 0x6E, 0x61, 0x6D, 0x65, 0x22, 0x3A, 0x22,
      0x64, 0x65, 0x63, 0x32, 0x30, 0x70, 0x31, 0x22, 0x2C, 0x22, 0x74, 0x79, 0x70, 0x65, 0x22,
      0x3A, 0x22, 0x64, 0x65, 0x63, 0x69, 0x6D, 0x61, 0x6C, 0x28, 0x32, 0x30, 0x2C, 0x31, 0x29,
      0x22, 0x2C, 0x22, 0x6E, 0x75, 0x6C, 0x6C, 0x61, 0x62, 0x6C, 0x65, 0x22, 0x3A, 0x74, 0x72,
      0x75, 0x65, 0x2C, 0x22, 0x6D, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x22, 0x3A, 0x7B,
      0x7D, 0x7D, 0x5D, 0x7D, 0x00, 0x18, 0x4A, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2D,
      0x6D, 0x72, 0x20, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6F, 0x6E, 0x20, 0x31, 0x2E, 0x31, 0x30,
      0x2E, 0x31, 0x20, 0x28, 0x62, 0x75, 0x69, 0x6C, 0x64, 0x20, 0x61, 0x38, 0x39, 0x64, 0x66,
      0x38, 0x66, 0x39, 0x39, 0x33, 0x32, 0x62, 0x36, 0x65, 0x66, 0x36, 0x36, 0x33, 0x33, 0x64,
      0x30, 0x36, 0x30, 0x36, 0x39, 0x65, 0x35, 0x30, 0x63, 0x39, 0x62, 0x37, 0x39, 0x37, 0x30,
      0x62, 0x65, 0x62, 0x64, 0x31, 0x29, 0x19, 0x3C, 0x1C, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x1C,
      0x00, 0x00, 0x00, 0xC5, 0x02, 0x00, 0x00, 0x50, 0x41, 0x52, 0x31,
    };

    unsigned int parquet_len = 1226;

    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{
        reinterpret_cast<char const*>(fixed_len_bytes_decimal_parquet), parquet_len});
    auto result = cudf::io::read_parquet(read_opts);
    EXPECT_EQ(result.tbl->view().num_columns(), 3);

    auto validity_c0    = cudf::test::iterators::nulls_at({19});
    int32_t col0_data[] = {6361295, 698632,  7821423, 7073444, 9631892, 3021012, 5195059,
                           9913714, 901749,  7776938, 3186566, 4955569, 5131067, 98619,
                           2282579, 7521455, 4430706, 1937859, 4532040, 0};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(0).size()),
              sizeof(col0_data) / sizeof(col0_data[0]));
    cudf::test::fixed_point_column_wrapper<int32_t> col0(
      std::begin(col0_data), std::end(col0_data), validity_c0, numeric::scale_type{-3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), col0);

    auto validity_c1    = cudf::test::iterators::nulls_at({18});
    int64_t col1_data[] = {361378026250,
                           30646804862,
                           429930238629,
                           418758703536,
                           895494171113,
                           435283865083,
                           809096053722,
                           -999999999999,
                           426465099333,
                           526684574144,
                           826310892810,
                           584686967589,
                           113822282951,
                           409236212092,
                           420631167535,
                           918438386086,
                           -999999999999,
                           489053889147,
                           0,
                           363993164092};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(1).size()),
              sizeof(col1_data) / sizeof(col1_data[0]));
    cudf::test::fixed_point_column_wrapper<int64_t> col1(
      std::begin(col1_data), std::end(col1_data), validity_c1, numeric::scale_type{-11});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), col1);

    auto validity_c2       = cudf::test::iterators::nulls_at({6, 14});
    __int128_t col2_data[] = {9078697037144433659,
                              9050770539577117612,
                              2358363961733893636,
                              1566059559232276662,
                              6658306200002735268,
                              4967909073046397334,
                              0,
                              7235588493887532473,
                              5023160741463849572,
                              2765173712965988273,
                              3880866513515749646,
                              5019704400576359500,
                              5544435986818825655,
                              7265381725809874549,
                              0,
                              1576192427381240677,
                              2828305195087094598,
                              260308667809395171,
                              2460080200895288476,
                              2718441925197820439};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(2).size()),
              sizeof(col2_data) / sizeof(col2_data[0]));
    cudf::test::fixed_point_column_wrapper<__int128_t> col2(
      std::begin(col2_data), std::end(col2_data), validity_c2, numeric::scale_type{-1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), col2);
  }
}

TEST_F(ParquetReaderTest, EmptyOutput)
{
  cudf::test::fixed_width_column_wrapper<int> c0;
  cudf::test::strings_column_wrapper c1;
  cudf::test::fixed_point_column_wrapper<int> c2({}, numeric::scale_type{2});
  cudf::test::lists_column_wrapper<float> _c3{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
  auto c3 = cudf::empty_like(_c3);

  cudf::test::fixed_width_column_wrapper<int> sc0;
  cudf::test::strings_column_wrapper sc1;
  cudf::test::lists_column_wrapper<int> _sc2{{1, 2}};
  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(sc0.release());
  struct_children.push_back(sc1.release());
  struct_children.push_back(cudf::empty_like(_sc2));
  cudf::test::structs_column_wrapper c4(std::move(struct_children));

  table_view expected({c0, c1, c2, *c3, c4});

  // set precision on the decimal column
  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[2].set_decimal_precision(1);

  auto filepath = temp_env->get_temp_filepath("EmptyOutput.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  out_args.set_metadata(std::move(expected_metadata));
  cudf::io::write_parquet(out_args);

  cudf::io::parquet_reader_options read_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_args);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetWriterTest, RowGroupSizeInvalid)
{
  auto const unused_table = std::make_unique<table>();
  std::vector<char> out_buffer;

  EXPECT_THROW(cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer),
                                                         unused_table->view())
                 .row_group_size_rows(0),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer),
                                                         unused_table->view())
                 .max_page_size_rows(0),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer),
                                                         unused_table->view())
                 .row_group_size_bytes(3 << 8),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer),
                                                         unused_table->view())
                 .max_page_size_bytes(3 << 8),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer),
                                                         unused_table->view())
                 .max_page_size_bytes(0xFFFF'FFFFUL),
               cudf::logic_error);

  EXPECT_THROW(cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(&out_buffer))
                 .row_group_size_rows(0),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(&out_buffer))
                 .max_page_size_rows(0),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(&out_buffer))
                 .row_group_size_bytes(3 << 8),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(&out_buffer))
                 .max_page_size_bytes(3 << 8),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(&out_buffer))
                 .max_page_size_bytes(0xFFFF'FFFFUL),
               cudf::logic_error);
}

TEST_F(ParquetWriterTest, RowGroupPageSizeMatch)
{
  auto const unused_table = std::make_unique<table>();
  std::vector<char> out_buffer;

  auto options = cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer),
                                                           unused_table->view())
                   .row_group_size_bytes(128 * 1024)
                   .max_page_size_bytes(512 * 1024)
                   .row_group_size_rows(10000)
                   .max_page_size_rows(20000)
                   .build();
  EXPECT_EQ(options.get_row_group_size_bytes(), options.get_max_page_size_bytes());
  EXPECT_EQ(options.get_row_group_size_rows(), options.get_max_page_size_rows());
}

TEST_F(ParquetChunkedWriterTest, RowGroupPageSizeMatch)
{
  std::vector<char> out_buffer;

  auto options = cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(&out_buffer))
                   .row_group_size_bytes(128 * 1024)
                   .max_page_size_bytes(512 * 1024)
                   .row_group_size_rows(10000)
                   .max_page_size_rows(20000)
                   .build();
  EXPECT_EQ(options.get_row_group_size_bytes(), options.get_max_page_size_bytes());
  EXPECT_EQ(options.get_row_group_size_rows(), options.get_max_page_size_rows());
}

TEST_F(ParquetWriterTest, EmptyList)
{
  auto L1 = cudf::make_lists_column(0,
                                    cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32)),
                                    cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}),
                                    0,
                                    {});
  auto L0 = cudf::make_lists_column(
    3, cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0}.release(), std::move(L1), 0, {});

  auto filepath = temp_env->get_temp_filepath("EmptyList.parquet");
  cudf::io::write_parquet(cudf::io::parquet_writer_options_builder(cudf::io::sink_info(filepath),
                                                                   cudf::table_view({*L0})));

  auto result = cudf::io::read_parquet(
    cudf::io::parquet_reader_options_builder(cudf::io::source_info(filepath)));

  using lcw     = cudf::test::lists_column_wrapper<int64_t>;
  auto expected = lcw{lcw{}, lcw{}, lcw{}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), expected);
}

TEST_F(ParquetWriterTest, DeepEmptyList)
{
  // Make a list column LLLi st only L is valid and LLi are all null. This tests whether we can
  // handle multiple nullptr offsets

  auto L2 = cudf::make_lists_column(0,
                                    cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32)),
                                    cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}),
                                    0,
                                    {});
  auto L1 = cudf::make_lists_column(
    0, cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32)), std::move(L2), 0, {});
  auto L0 = cudf::make_lists_column(
    3, cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0}.release(), std::move(L1), 0, {});

  auto filepath = temp_env->get_temp_filepath("DeepEmptyList.parquet");
  cudf::io::write_parquet(cudf::io::parquet_writer_options_builder(cudf::io::sink_info(filepath),
                                                                   cudf::table_view({*L0})));

  auto result = cudf::io::read_parquet(
    cudf::io::parquet_reader_options_builder(cudf::io::source_info(filepath)));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), *L0);
}

TEST_F(ParquetWriterTest, EmptyListWithStruct)
{
  auto L2 = cudf::make_lists_column(0,
                                    cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32)),
                                    cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}),
                                    0,
                                    {});

  auto children = std::vector<std::unique_ptr<cudf::column>>{};
  children.push_back(std::move(L2));
  auto S2 = cudf::make_structs_column(0, std::move(children), 0, {});
  auto L1 = cudf::make_lists_column(
    0, cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32)), std::move(S2), 0, {});
  auto L0 = cudf::make_lists_column(
    3, cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0}.release(), std::move(L1), 0, {});

  auto filepath = temp_env->get_temp_filepath("EmptyListWithStruct.parquet");
  cudf::io::write_parquet(cudf::io::parquet_writer_options_builder(cudf::io::sink_info(filepath),
                                                                   cudf::table_view({*L0})));
  auto result = cudf::io::read_parquet(
    cudf::io::parquet_reader_options_builder(cudf::io::source_info(filepath)));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), *L0);
}

TEST_F(ParquetWriterTest, CheckPageRows)
{
  auto sequence = thrust::make_counting_iterator(0);
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  constexpr auto page_rows = 5000;
  constexpr auto num_rows  = 2 * page_rows;
  column_wrapper<int> col(sequence, sequence + num_rows, validity);

  auto expected = table_view{{col}};

  auto const filepath = temp_env->get_temp_filepath("CheckPageRows.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .max_page_size_rows(page_rows);
  cudf::io::write_parquet(out_opts);

  // check first page header and make sure it has only page_rows values
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  ASSERT_GT(fmd.row_groups.size(), 0);
  ASSERT_EQ(fmd.row_groups[0].columns.size(), 1);
  auto const& first_chunk = fmd.row_groups[0].columns[0].meta_data;
  ASSERT_GT(first_chunk.data_page_offset, 0);

  // read first data page header.  sizeof(PageHeader) is not exact, but the thrift encoded
  // version should be smaller than size of the struct.
  auto const ph = read_page_header(
    source, {first_chunk.data_page_offset, sizeof(cudf::io::parquet::PageHeader), 0});

  EXPECT_EQ(ph.data_page_header.num_values, page_rows);
}

TEST_F(ParquetWriterTest, CheckPageRowsAdjusted)
{
  // enough for a few pages with the default 20'000 rows/page
  constexpr auto rows_per_page = 20'000;
  constexpr auto num_rows      = 3 * rows_per_page;
  const std::string s1(32, 'a');
  auto col0_elements =
    cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return s1; });
  auto col0 = cudf::test::strings_column_wrapper(col0_elements, col0_elements + num_rows);

  auto const expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("CheckPageRowsAdjusted.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .max_page_size_rows(rows_per_page);
  cudf::io::write_parquet(out_opts);

  // check first page header and make sure it has only page_rows values
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  ASSERT_GT(fmd.row_groups.size(), 0);
  ASSERT_EQ(fmd.row_groups[0].columns.size(), 1);
  auto const& first_chunk = fmd.row_groups[0].columns[0].meta_data;
  ASSERT_GT(first_chunk.data_page_offset, 0);

  // read first data page header.  sizeof(PageHeader) is not exact, but the thrift encoded
  // version should be smaller than size of the struct.
  auto const ph = read_page_header(
    source, {first_chunk.data_page_offset, sizeof(cudf::io::parquet::PageHeader), 0});

  EXPECT_LE(ph.data_page_header.num_values, rows_per_page);
}

TEST_F(ParquetWriterTest, CheckPageRowsTooSmall)
{
  constexpr auto rows_per_page = 1'000;
  constexpr auto fragment_size = 5'000;
  constexpr auto num_rows      = 3 * rows_per_page;
  const std::string s1(32, 'a');
  auto col0_elements =
    cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return s1; });
  auto col0 = cudf::test::strings_column_wrapper(col0_elements, col0_elements + num_rows);

  auto const expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("CheckPageRowsTooSmall.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .max_page_fragment_size(fragment_size)
      .max_page_size_rows(rows_per_page);
  cudf::io::write_parquet(out_opts);

  // check that file is written correctly when rows/page < fragment size
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  ASSERT_TRUE(fmd.row_groups.size() > 0);
  ASSERT_TRUE(fmd.row_groups[0].columns.size() == 1);
  auto const& first_chunk = fmd.row_groups[0].columns[0].meta_data;
  ASSERT_TRUE(first_chunk.data_page_offset > 0);

  // read first data page header.  sizeof(PageHeader) is not exact, but the thrift encoded
  // version should be smaller than size of the struct.
  auto const ph = read_page_header(
    source, {first_chunk.data_page_offset, sizeof(cudf::io::parquet::PageHeader), 0});

  // there should be only one page since the fragment size is larger than rows_per_page
  EXPECT_EQ(ph.data_page_header.num_values, num_rows);
}

TEST_F(ParquetWriterTest, Decimal128Stats)
{
  // check that decimal128 min and max statistics are written in network byte order
  // this is negative, so should be the min
  std::vector<uint8_t> expected_min{
    0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint8_t> expected_max{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6};

  __int128_t val0 = 0xa1b2'c3d4'e5f6ULL;
  __int128_t val1 = val0 << 80;
  column_wrapper<numeric::decimal128> col0{{numeric::decimal128(val0, numeric::scale_type{0}),
                                            numeric::decimal128(val1, numeric::scale_type{0})}};

  auto expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("Decimal128Stats.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  auto const stats = get_statistics(fmd.row_groups[0].columns[0]);

  EXPECT_EQ(expected_min, stats.min_value);
  EXPECT_EQ(expected_max, stats.max_value);
}

// =============================================================================
// ---- test data for stats sort order tests
// need at least 3 pages, and min page count is 5000, so need at least 15000 values.
// use 20000 to be safe.
static constexpr int num_ordered_rows            = 20000;
static constexpr int page_size_for_ordered_tests = 5000;

namespace {
namespace testdata {
// ----- most numerics. scale by 100 so all values fit in a single byte

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>,
                 cudf::test::fixed_width_column_wrapper<T>>
ascending()
{
  int start = std::is_signed_v<T> ? -num_ordered_rows / 2 : 0;
  auto elements =
    cudf::detail::make_counting_transform_iterator(start, [](auto i) { return i / 100; });
  return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>,
                 cudf::test::fixed_width_column_wrapper<T>>
descending()
{
  if (std::is_signed_v<T>) {
    auto elements = cudf::detail::make_counting_transform_iterator(-num_ordered_rows / 2,
                                                                   [](auto i) { return -i / 100; });
    return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
  } else {
    auto elements = cudf::detail::make_counting_transform_iterator(
      0, [](auto i) { return (num_ordered_rows - i) / 100; });
    return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
  }
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>,
                 cudf::test::fixed_width_column_wrapper<T>>
unordered()
{
  if (std::is_signed_v<T>) {
    auto elements = cudf::detail::make_counting_transform_iterator(
      -num_ordered_rows / 2, [](auto i) { return (i % 2 ? i : -i) / 100; });
    return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
  } else {
    auto elements = cudf::detail::make_counting_transform_iterator(
      0, [](auto i) { return (i % 2 ? i : num_ordered_rows - i) / 100; });
    return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
  }
}

// ----- bool

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, cudf::test::fixed_width_column_wrapper<bool>> ascending()
{
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i >= num_ordered_rows / 2; });
  return cudf::test::fixed_width_column_wrapper<bool>(elements, elements + num_ordered_rows);
}

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, cudf::test::fixed_width_column_wrapper<bool>> descending()
{
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i < num_ordered_rows / 2; });
  return cudf::test::fixed_width_column_wrapper<bool>(elements, elements + num_ordered_rows);
}

template <typename T>
std::enable_if_t<std::is_same_v<T, bool>, cudf::test::fixed_width_column_wrapper<bool>> unordered()
{
  auto elements = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    switch (i / page_size_for_ordered_tests) {
      case 0: return true;
      case 1: return false;
      case 2: return true;
      default: return false;
    }
  });
  return cudf::test::fixed_width_column_wrapper<bool>(elements, elements + num_ordered_rows);
}

// ----- fixed point types

template <typename T>
std::enable_if_t<cudf::is_fixed_point<T>(), cudf::test::fixed_width_column_wrapper<T>> ascending()
{
  auto elements = cudf::detail::make_counting_transform_iterator(
    -num_ordered_rows / 2, [](auto i) { return T(i, numeric::scale_type{0}); });
  return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
}

template <typename T>
std::enable_if_t<cudf::is_fixed_point<T>(), cudf::test::fixed_width_column_wrapper<T>> descending()
{
  auto elements = cudf::detail::make_counting_transform_iterator(
    -num_ordered_rows / 2, [](auto i) { return T(-i, numeric::scale_type{0}); });
  return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
}

template <typename T>
std::enable_if_t<cudf::is_fixed_point<T>(), cudf::test::fixed_width_column_wrapper<T>> unordered()
{
  auto elements = cudf::detail::make_counting_transform_iterator(
    -num_ordered_rows / 2, [](auto i) { return T(i % 2 ? i : -i, numeric::scale_type{0}); });
  return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
}

// ----- chrono types
// ----- timstamp

template <typename T>
std::enable_if_t<cudf::is_timestamp<T>(), cudf::test::fixed_width_column_wrapper<T>> ascending()
{
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return T(typename T::duration(i)); });
  return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
}

template <typename T>
std::enable_if_t<cudf::is_timestamp<T>(), cudf::test::fixed_width_column_wrapper<T>> descending()
{
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return T(typename T::duration(num_ordered_rows - i)); });
  return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
}

template <typename T>
std::enable_if_t<cudf::is_timestamp<T>(), cudf::test::fixed_width_column_wrapper<T>> unordered()
{
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return T(typename T::duration(i % 2 ? i : num_ordered_rows - i)); });
  return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
}

// ----- duration

template <typename T>
std::enable_if_t<cudf::is_duration<T>(), cudf::test::fixed_width_column_wrapper<T>> ascending()
{
  auto elements = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return T(i); });
  return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
}

template <typename T>
std::enable_if_t<cudf::is_duration<T>(), cudf::test::fixed_width_column_wrapper<T>> descending()
{
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return T(num_ordered_rows - i); });
  return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
}

template <typename T>
std::enable_if_t<cudf::is_duration<T>(), cudf::test::fixed_width_column_wrapper<T>> unordered()
{
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return T(i % 2 ? i : num_ordered_rows - i); });
  return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
}

// ----- string_view

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, cudf::test::strings_column_wrapper>
ascending()
{
  char buf[10];
  auto elements = cudf::detail::make_counting_transform_iterator(0, [&buf](auto i) {
    sprintf(buf, "%09d", i);
    return std::string(buf);
  });
  return cudf::test::strings_column_wrapper(elements, elements + num_ordered_rows);
}

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, cudf::test::strings_column_wrapper>
descending()
{
  char buf[10];
  auto elements = cudf::detail::make_counting_transform_iterator(0, [&buf](auto i) {
    sprintf(buf, "%09d", num_ordered_rows - i);
    return std::string(buf);
  });
  return cudf::test::strings_column_wrapper(elements, elements + num_ordered_rows);
}

template <typename T>
std::enable_if_t<std::is_same_v<T, cudf::string_view>, cudf::test::strings_column_wrapper>
unordered()
{
  char buf[10];
  auto elements = cudf::detail::make_counting_transform_iterator(0, [&buf](auto i) {
    sprintf(buf, "%09d", (i % 2 == 0) ? i : (num_ordered_rows - i));
    return std::string(buf);
  });
  return cudf::test::strings_column_wrapper(elements, elements + num_ordered_rows);
}

}  // namespace testdata
}  // anonymous namespace

TYPED_TEST(ParquetWriterComparableTypeTest, ThreeColumnSorted)
{
  using T = TypeParam;

  auto col0 = testdata::ascending<T>();
  auto col1 = testdata::descending<T>();
  auto col2 = testdata::unordered<T>();

  auto const expected = table_view{{col0, col1, col2}};

  auto const filepath = temp_env->get_temp_filepath("ThreeColumnSorted.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .max_page_size_rows(page_size_for_ordered_tests)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  ASSERT_GT(fmd.row_groups.size(), 0);

  auto const& columns = fmd.row_groups[0].columns;
  ASSERT_EQ(columns.size(), static_cast<size_t>(expected.num_columns()));

  // now check that the boundary order for chunk 1 is ascending,
  // chunk 2 is descending, and chunk 3 is unordered
  cudf::io::parquet::BoundaryOrder expected_orders[] = {
    cudf::io::parquet::BoundaryOrder::ASCENDING,
    cudf::io::parquet::BoundaryOrder::DESCENDING,
    cudf::io::parquet::BoundaryOrder::UNORDERED};

  for (std::size_t i = 0; i < columns.size(); i++) {
    auto const ci = read_column_index(source, columns[i]);
    EXPECT_EQ(ci.boundary_order, expected_orders[i]);
  }
}

// utility functions for column index tests

// compare two values.  return -1 if v1 < v2,
// 0 if v1 == v2, and 1 if v1 > v2.
template <typename T>
int32_t compare(T& v1, T& v2)
{
  return (v1 > v2) - (v1 < v2);
}

// compare two binary statistics blobs based on their physical
// and converted types. returns -1 if v1 < v2, 0 if v1 == v2, and
// 1 if v1 > v2.
int32_t compare_binary(std::vector<uint8_t> const& v1,
                       std::vector<uint8_t> const& v2,
                       cudf::io::parquet::Type ptype,
                       cudf::io::parquet::ConvertedType ctype)
{
  switch (ptype) {
    case cudf::io::parquet::INT32:
      switch (ctype) {
        case cudf::io::parquet::UINT_8:
        case cudf::io::parquet::UINT_16:
        case cudf::io::parquet::UINT_32:
          return compare(*(reinterpret_cast<uint32_t const*>(v1.data())),
                         *(reinterpret_cast<uint32_t const*>(v2.data())));
        default:
          return compare(*(reinterpret_cast<int32_t const*>(v1.data())),
                         *(reinterpret_cast<int32_t const*>(v2.data())));
      }

    case cudf::io::parquet::INT64:
      if (ctype == cudf::io::parquet::UINT_64) {
        return compare(*(reinterpret_cast<uint64_t const*>(v1.data())),
                       *(reinterpret_cast<uint64_t const*>(v2.data())));
      }
      return compare(*(reinterpret_cast<int64_t const*>(v1.data())),
                     *(reinterpret_cast<int64_t const*>(v2.data())));

    case cudf::io::parquet::FLOAT:
      return compare(*(reinterpret_cast<float const*>(v1.data())),
                     *(reinterpret_cast<float const*>(v2.data())));

    case cudf::io::parquet::DOUBLE:
      return compare(*(reinterpret_cast<double const*>(v1.data())),
                     *(reinterpret_cast<double const*>(v2.data())));

    case cudf::io::parquet::BYTE_ARRAY: {
      int32_t v1sz = v1.size();
      int32_t v2sz = v2.size();
      int32_t ret  = memcmp(v1.data(), v2.data(), std::min(v1sz, v2sz));
      if (ret != 0 or v1sz == v2sz) { return ret; }
      return v1sz - v2sz;
    }

    default: CUDF_FAIL("Invalid type in compare_binary");
  }

  return 0;
}

TEST_P(ParquetV2Test, LargeColumnIndex)
{
  // create a file large enough to be written in 2 batches (currently 1GB per batch)
  // pick fragment size that num_rows is divisible by, so we'll get equal sized row groups
  const std::string s1(1000, 'a');
  const std::string s2(1000, 'b');
  constexpr auto num_rows  = 512 * 1024;
  constexpr auto frag_size = num_rows / 128;
  auto const is_v2         = GetParam();

  auto col0_elements = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) { return (i < num_rows) ? s1 : s2; });
  auto col0 = cudf::test::strings_column_wrapper(col0_elements, col0_elements + 2 * num_rows);

  auto const expected = table_view{{col0, col0}};

  auto const filepath = temp_env->get_temp_filepath("LargeColumnIndex.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .compression(cudf::io::compression_type::NONE)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER)
      .write_v2_headers(is_v2)
      .max_page_fragment_size(frag_size)
      .row_group_size_bytes(1024 * 1024 * 1024)
      .row_group_size_rows(num_rows);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  for (auto const& rg : fmd.row_groups) {
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      // check trunc(page.min) <= stats.min && trun(page.max) >= stats.max
      auto const ptype = fmd.schema[c + 1].type;
      auto const ctype = fmd.schema[c + 1].converted_type;
      EXPECT_TRUE(compare_binary(ci.min_values[0], stats.min_value, ptype, ctype) <= 0);
      EXPECT_TRUE(compare_binary(ci.max_values[0], stats.max_value, ptype, ctype) >= 0);
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnOffsetIndex)
{
  constexpr auto num_rows = 100000;
  auto const is_v2        = GetParam();
  auto const expected_hdr_type =
    is_v2 ? cudf::io::parquet::PageType::DATA_PAGE_V2 : cudf::io::parquet::PageType::DATA_PAGE;

  // fixed length strings
  auto str1_elements = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    char buf[30];
    sprintf(buf, "%012d", i);
    return std::string(buf);
  });
  auto col0          = cudf::test::strings_column_wrapper(str1_elements, str1_elements + num_rows);

  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<uint64_t>(num_rows);
  auto col5_data = random_values<float>(num_rows);
  auto col6_data = random_values<double>(num_rows);

  auto col1 = cudf::test::fixed_width_column_wrapper<int8_t>(col1_data.begin(), col1_data.end());
  auto col2 = cudf::test::fixed_width_column_wrapper<int16_t>(col2_data.begin(), col2_data.end());
  auto col3 = cudf::test::fixed_width_column_wrapper<int32_t>(col3_data.begin(), col3_data.end());
  auto col4 = cudf::test::fixed_width_column_wrapper<uint64_t>(col4_data.begin(), col4_data.end());
  auto col5 = cudf::test::fixed_width_column_wrapper<float>(col5_data.begin(), col5_data.end());
  auto col6 = cudf::test::fixed_width_column_wrapper<double>(col6_data.begin(), col6_data.end());

  // mixed length strings
  auto str2_elements = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    char buf[30];
    sprintf(buf, "%d", i);
    return std::string(buf);
  });
  auto col7          = cudf::test::strings_column_wrapper(str2_elements, str2_elements + num_rows);

  auto const expected = table_view{{col0, col1, col2, col3, col4, col5, col6, col7}};

  auto const filepath = temp_env->get_temp_filepath("CheckColumnOffsetIndex.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .write_v2_headers(is_v2)
      .max_page_size_rows(20000);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);

      int64_t num_vals = 0;
      for (size_t o = 0; o < oi.page_locations.size(); o++) {
        auto const& page_loc = oi.page_locations[o];
        auto const ph        = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        EXPECT_EQ(page_loc.first_row_index, num_vals);
        num_vals += is_v2 ? ph.data_page_header_v2.num_rows : ph.data_page_header.num_values;
      }

      // loop over page stats from the column index. check that stats.min <= page.min
      // and stats.max >= page.max for each page.
      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      // schema indexing starts at 1
      auto const ptype = fmd.schema[c + 1].type;
      auto const ctype = fmd.schema[c + 1].converted_type;
      for (size_t p = 0; p < ci.min_values.size(); p++) {
        // null_pages should always be false
        EXPECT_FALSE(ci.null_pages[p]);
        // null_counts should always be 0
        EXPECT_EQ(ci.null_counts[p], 0);
        EXPECT_TRUE(compare_binary(stats.min_value, ci.min_values[p], ptype, ctype) <= 0);
      }
      for (size_t p = 0; p < ci.max_values.size(); p++)
        EXPECT_TRUE(compare_binary(stats.max_value, ci.max_values[p], ptype, ctype) >= 0);
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnOffsetIndexNulls)
{
  constexpr auto num_rows = 100000;
  auto const is_v2        = GetParam();
  auto const expected_hdr_type =
    is_v2 ? cudf::io::parquet::PageType::DATA_PAGE_V2 : cudf::io::parquet::PageType::DATA_PAGE;

  // fixed length strings
  auto str1_elements = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    char buf[30];
    sprintf(buf, "%012d", i);
    return std::string(buf);
  });
  auto col0          = cudf::test::strings_column_wrapper(str1_elements, str1_elements + num_rows);

  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<uint64_t>(num_rows);
  auto col5_data = random_values<float>(num_rows);
  auto col6_data = random_values<double>(num_rows);

  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });

  // add null values for all but first column
  auto col1 =
    cudf::test::fixed_width_column_wrapper<int8_t>(col1_data.begin(), col1_data.end(), valids);
  auto col2 =
    cudf::test::fixed_width_column_wrapper<int16_t>(col2_data.begin(), col2_data.end(), valids);
  auto col3 =
    cudf::test::fixed_width_column_wrapper<int32_t>(col3_data.begin(), col3_data.end(), valids);
  auto col4 =
    cudf::test::fixed_width_column_wrapper<uint64_t>(col4_data.begin(), col4_data.end(), valids);
  auto col5 =
    cudf::test::fixed_width_column_wrapper<float>(col5_data.begin(), col5_data.end(), valids);
  auto col6 =
    cudf::test::fixed_width_column_wrapper<double>(col6_data.begin(), col6_data.end(), valids);

  // mixed length strings
  auto str2_elements = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    char buf[30];
    sprintf(buf, "%d", i);
    return std::string(buf);
  });
  auto col7 = cudf::test::strings_column_wrapper(str2_elements, str2_elements + num_rows, valids);

  auto expected = table_view{{col0, col1, col2, col3, col4, col5, col6, col7}};

  auto const filepath = temp_env->get_temp_filepath("CheckColumnOffsetIndexNulls.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .write_v2_headers(is_v2)
      .max_page_size_rows(20000);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);

      int64_t num_vals = 0;
      for (size_t o = 0; o < oi.page_locations.size(); o++) {
        auto const& page_loc = oi.page_locations[o];
        auto const ph        = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        EXPECT_EQ(page_loc.first_row_index, num_vals);
        num_vals += is_v2 ? ph.data_page_header_v2.num_rows : ph.data_page_header.num_values;
      }

      // loop over page stats from the column index. check that stats.min <= page.min
      // and stats.max >= page.max for each page.
      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      // should be half nulls, except no nulls in column 0
      EXPECT_EQ(stats.null_count, c == 0 ? 0 : num_rows / 2);

      // schema indexing starts at 1
      auto const ptype = fmd.schema[c + 1].type;
      auto const ctype = fmd.schema[c + 1].converted_type;
      for (size_t p = 0; p < ci.min_values.size(); p++) {
        EXPECT_FALSE(ci.null_pages[p]);
        if (c > 0) {  // first column has no nulls
          EXPECT_GT(ci.null_counts[p], 0);
        } else {
          EXPECT_EQ(ci.null_counts[p], 0);
        }
        EXPECT_TRUE(compare_binary(stats.min_value, ci.min_values[p], ptype, ctype) <= 0);
      }
      for (size_t p = 0; p < ci.max_values.size(); p++) {
        EXPECT_TRUE(compare_binary(stats.max_value, ci.max_values[p], ptype, ctype) >= 0);
      }
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnOffsetIndexNullColumn)
{
  constexpr auto num_rows = 100000;
  auto const is_v2        = GetParam();
  auto const expected_hdr_type =
    is_v2 ? cudf::io::parquet::PageType::DATA_PAGE_V2 : cudf::io::parquet::PageType::DATA_PAGE;

  // fixed length strings
  auto str1_elements = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    char buf[30];
    sprintf(buf, "%012d", i);
    return std::string(buf);
  });
  auto col0          = cudf::test::strings_column_wrapper(str1_elements, str1_elements + num_rows);

  auto col1_data = random_values<int32_t>(num_rows);
  auto col2_data = random_values<int32_t>(num_rows);

  // col1 is all nulls
  auto valids = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return false; });
  auto col1 =
    cudf::test::fixed_width_column_wrapper<int32_t>(col1_data.begin(), col1_data.end(), valids);
  auto col2 = cudf::test::fixed_width_column_wrapper<int32_t>(col2_data.begin(), col2_data.end());

  // mixed length strings
  auto str2_elements = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    char buf[30];
    sprintf(buf, "%d", i);
    return std::string(buf);
  });
  auto col3          = cudf::test::strings_column_wrapper(str2_elements, str2_elements + num_rows);

  auto expected = table_view{{col0, col1, col2, col3}};

  auto const filepath = temp_env->get_temp_filepath("CheckColumnOffsetIndexNullColumn.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .write_v2_headers(is_v2)
      .max_page_size_rows(20000);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);

      int64_t num_vals = 0;
      for (size_t o = 0; o < oi.page_locations.size(); o++) {
        auto const& page_loc = oi.page_locations[o];
        auto const ph        = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        EXPECT_EQ(page_loc.first_row_index, num_vals);
        num_vals += is_v2 ? ph.data_page_header_v2.num_rows : ph.data_page_header.num_values;
      }

      // loop over page stats from the column index. check that stats.min <= page.min
      // and stats.max >= page.max for each non-empty page.
      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      // there should be no nulls except column 1 which is all nulls
      EXPECT_EQ(stats.null_count, c == 1 ? num_rows : 0);

      // schema indexing starts at 1
      auto const ptype = fmd.schema[c + 1].type;
      auto const ctype = fmd.schema[c + 1].converted_type;
      for (size_t p = 0; p < ci.min_values.size(); p++) {
        // check tnat null_pages is true for column 1
        if (c == 1) {
          EXPECT_TRUE(ci.null_pages[p]);
          EXPECT_GT(ci.null_counts[p], 0);
        }
        if (not ci.null_pages[p]) {
          EXPECT_EQ(ci.null_counts[p], 0);
          EXPECT_TRUE(compare_binary(stats.min_value, ci.min_values[p], ptype, ctype) <= 0);
        }
      }
      for (size_t p = 0; p < ci.max_values.size(); p++) {
        if (not ci.null_pages[p]) {
          EXPECT_TRUE(compare_binary(stats.max_value, ci.max_values[p], ptype, ctype) >= 0);
        }
      }
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnOffsetIndexStruct)
{
  auto const is_v2 = GetParam();
  auto const expected_hdr_type =
    is_v2 ? cudf::io::parquet::PageType::DATA_PAGE_V2 : cudf::io::parquet::PageType::DATA_PAGE;

  auto c0 = testdata::ascending<uint32_t>();

  auto sc0 = testdata::ascending<cudf::string_view>();
  auto sc1 = testdata::descending<int32_t>();
  auto sc2 = testdata::unordered<int64_t>();

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(sc0.release());
  struct_children.push_back(sc1.release());
  struct_children.push_back(sc2.release());
  cudf::test::structs_column_wrapper c1(std::move(struct_children));

  auto listgen = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? i / 2 : num_ordered_rows - (i / 2); });
  auto list =
    cudf::test::fixed_width_column_wrapper<int32_t>(listgen, listgen + 2 * num_ordered_rows);
  auto offgen = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  auto offsets =
    cudf::test::fixed_width_column_wrapper<int32_t>(offgen, offgen + num_ordered_rows + 1);

  auto c2 = cudf::make_lists_column(num_ordered_rows, offsets.release(), list.release(), 0, {});

  table_view expected({c0, c1, *c2});

  auto const filepath = temp_env->get_temp_filepath("CheckColumnOffsetIndexStruct.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .write_v2_headers(is_v2)
      .max_page_size_rows(page_size_for_ordered_tests);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  // hard coded schema indices.
  // TODO find a way to do this without magic
  size_t const colidxs[] = {1, 3, 4, 5, 8};
  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      size_t colidx     = colidxs[c];
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);

      int64_t num_vals = 0;
      for (size_t o = 0; o < oi.page_locations.size(); o++) {
        auto const& page_loc = oi.page_locations[o];
        auto const ph        = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        EXPECT_EQ(page_loc.first_row_index, num_vals);
        // last column has 2 values per row
        num_vals += is_v2 ? ph.data_page_header_v2.num_rows
                          : ph.data_page_header.num_values / (c == rg.columns.size() - 1 ? 2 : 1);
      }

      // loop over page stats from the column index. check that stats.min <= page.min
      // and stats.max >= page.max for each page.
      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      auto const ptype = fmd.schema[colidx].type;
      auto const ctype = fmd.schema[colidx].converted_type;
      for (size_t p = 0; p < ci.min_values.size(); p++) {
        EXPECT_TRUE(compare_binary(stats.min_value, ci.min_values[p], ptype, ctype) <= 0);
      }
      for (size_t p = 0; p < ci.max_values.size(); p++) {
        EXPECT_TRUE(compare_binary(stats.max_value, ci.max_values[p], ptype, ctype) >= 0);
      }
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnOffsetIndexStructNulls)
{
  auto const is_v2 = GetParam();
  auto const expected_hdr_type =
    is_v2 ? cudf::io::parquet::PageType::DATA_PAGE_V2 : cudf::io::parquet::PageType::DATA_PAGE;

  auto validity2 =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 2; });
  auto validity3 = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type i) { return (i % 3) != 0; });
  auto validity4 = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type i) { return (i % 4) != 0; });
  auto validity5 = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type i) { return (i % 5) != 0; });

  auto c0 = testdata::ascending<uint32_t>();

  auto col1_data = random_values<int32_t>(num_ordered_rows);
  auto col2_data = random_values<int32_t>(num_ordered_rows);
  auto col3_data = random_values<int32_t>(num_ordered_rows);

  // col1 is all nulls
  auto col1 =
    cudf::test::fixed_width_column_wrapper<int32_t>(col1_data.begin(), col1_data.end(), validity2);
  auto col2 =
    cudf::test::fixed_width_column_wrapper<int32_t>(col2_data.begin(), col2_data.end(), validity3);
  auto col3 =
    cudf::test::fixed_width_column_wrapper<int32_t>(col2_data.begin(), col2_data.end(), validity4);

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(col1.release());
  struct_children.push_back(col2.release());
  struct_children.push_back(col3.release());
  auto struct_validity = std::vector<bool>(validity5, validity5 + num_ordered_rows);
  cudf::test::structs_column_wrapper c1(std::move(struct_children), struct_validity);
  table_view expected({c0, c1});

  auto const filepath = temp_env->get_temp_filepath("CheckColumnOffsetIndexStructNulls.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .write_v2_headers(is_v2)
      .max_page_size_rows(page_size_for_ordered_tests);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);
      auto const ci = read_column_index(source, chunk);

      int64_t num_vals = 0;
      for (size_t o = 0; o < oi.page_locations.size(); o++) {
        auto const& page_loc = oi.page_locations[o];
        auto const ph        = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        EXPECT_EQ(page_loc.first_row_index, num_vals);
        num_vals += is_v2 ? ph.data_page_header_v2.num_rows : ph.data_page_header.num_values;
        // check that null counts match
        if (is_v2) { EXPECT_EQ(ci.null_counts[o], ph.data_page_header_v2.num_nulls); }
      }
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnIndexListWithNulls)
{
  auto const is_v2 = GetParam();
  auto const expected_hdr_type =
    is_v2 ? cudf::io::parquet::PageType::DATA_PAGE_V2 : cudf::io::parquet::PageType::DATA_PAGE;

  using cudf::test::iterators::null_at;
  using cudf::test::iterators::nulls_at;
  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // 4 nulls
  // [NULL, 2, NULL]
  // []
  // [4, 5]
  // NULL
  lcw col0{{{{1, 2, 3}, nulls_at({0, 2})}, {}, {4, 5}, {}}, null_at(3)};

  // 4 nulls
  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8]]
  // []
  // [[]]
  lcw col1{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, {{7, 8}}, lcw{}, lcw{lcw{}}};

  // 4 nulls
  // [[1, 2, 3], [], [4, 5], NULL, [0, 6, 0]]
  // [[7, 8]]
  // []
  // [[]]
  lcw col2{{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, null_at(3)}, {{7, 8}}, lcw{}, lcw{lcw{}}};

  // 6 nulls
  // [[1, 2, 3], [], [4, 5], NULL, [NULL, 6, NULL]]
  // [[7, 8]]
  // []
  // [[]]
  using dlcw = cudf::test::lists_column_wrapper<double>;
  dlcw col3{{{{1., 2., 3.}, {}, {4., 5.}, {}, {{0., 6., 0.}, nulls_at({0, 2})}}, null_at(3)},
            {{7., 8.}},
            dlcw{},
            dlcw{dlcw{}}};

  // 4 nulls
  // [[1, 2, 3], [], [4, 5], NULL, [0, 6, 0]]
  // [[7, 8]]
  // []
  // NULL
  using ui16lcw = cudf::test::lists_column_wrapper<uint16_t>;
  cudf::test::lists_column_wrapper<uint16_t> col4{
    {{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, null_at(3)}, {{7, 8}}, ui16lcw{}, ui16lcw{ui16lcw{}}},
    null_at(3)};

  // 6 nulls
  // [[1, 2, 3], [], [4, 5], NULL, [NULL, 6, NULL]]
  // [[7, 8]]
  // []
  // NULL
  lcw col5{{{{{1, 2, 3}, {}, {4, 5}, {}, {{0, 6, 0}, nulls_at({0, 2})}}, null_at(3)},
            {{7, 8}},
            lcw{},
            lcw{lcw{}}},
           null_at(3)};

  // 4 nulls
  using strlcw = cudf::test::lists_column_wrapper<cudf::string_view>;
  cudf::test::lists_column_wrapper<cudf::string_view> col6{
    {{"Monday", "Monday", "Friday"}, {}, {"Monday", "Friday"}, {}, {"Sunday", "Funday"}},
    {{"bee", "sting"}},
    strlcw{},
    strlcw{strlcw{}}};

  // 11 nulls
  // [[[NULL,2,NULL,4]], [[NULL,6,NULL], [8,9]]]
  // [NULL, [[13],[14,15,16]],  NULL]
  // [NULL, [], NULL, [[]]]
  // NULL
  lcw col7{{
             {{{{1, 2, 3, 4}, nulls_at({0, 2})}}, {{{5, 6, 7}, nulls_at({0, 2})}, {8, 9}}},
             {{{{10, 11}, {12}}, {{13}, {14, 15, 16}}, {{17, 18}}}, nulls_at({0, 2})},
             {{lcw{lcw{}}, lcw{}, lcw{}, lcw{lcw{}}}, nulls_at({0, 2})},
             lcw{lcw{lcw{}}},
           },
           null_at(3)};

  table_view expected({col0, col1, col2, col3, col4, col5, col6, col7});

  int64_t const expected_null_counts[] = {4, 4, 4, 6, 4, 6, 4, 11};

  auto const filepath = temp_env->get_temp_filepath("ColumnIndexListWithNulls.parquet");
  auto out_opts = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
                    .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
                    .write_v2_headers(is_v2)
                    .compression(cudf::io::compression_type::NONE);

  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);

      for (size_t o = 0; o < oi.page_locations.size(); o++) {
        auto const& page_loc = oi.page_locations[o];
        auto const ph        = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        // check null counts in V2 header
        if (is_v2) { EXPECT_EQ(ph.data_page_header_v2.num_nulls, expected_null_counts[c]); }
      }

      // check null counts in column chunk stats and page indexes
      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);
      EXPECT_EQ(stats.null_count, expected_null_counts[c]);

      // should only be one page
      EXPECT_FALSE(ci.null_pages[0]);
      EXPECT_EQ(ci.null_counts[0], expected_null_counts[c]);
    }
  }
}

TEST_F(ParquetWriterTest, CheckColumnIndexTruncation)
{
  char const* coldata[] = {
    // in-range 7 bit.  should truncate to "yyyyyyyz"
    "yyyyyyyyy",
    // max 7 bit. should truncate to "x7fx7fx7fx7fx7fx7fx7fx80", since it's
    // considered binary, not UTF-8.  If UTF-8 it should not truncate.
    "\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f",
    // max binary.  this should not truncate
    "\xff\xff\xff\xff\xff\xff\xff\xff\xff",
    // in-range 2-byte UTF8 (U+00E9). should truncate to "éééê"
    "ééééé",
    // max 2-byte UTF8 (U+07FF). should not truncate
    "߿߿߿߿߿",
    // in-range 3-byte UTF8 (U+0800). should truncate to "ࠀࠁ"
    "ࠀࠀࠀ",
    // max 3-byte UTF8 (U+FFFF). should not truncate
    "\xef\xbf\xbf\xef\xbf\xbf\xef\xbf\xbf",
    // in-range 4-byte UTF8 (U+10000). should truncate to "𐀀𐀁"
    "𐀀𐀀𐀀",
    // max unicode (U+10FFFF). should truncate to \xf4\x8f\xbf\xbf\xf4\x90\x80\x80,
    // which is no longer valid unicode, but is still ok UTF-8???
    "\xf4\x8f\xbf\xbf\xf4\x8f\xbf\xbf\xf4\x8f\xbf\xbf",
    // max 4-byte UTF8 (U+1FFFFF). should not truncate
    "\xf7\xbf\xbf\xbf\xf7\xbf\xbf\xbf\xf7\xbf\xbf\xbf"};

  // NOTE: UTF8 min is initialized with 0xf7bfbfbf. Binary values larger
  // than that will not become minimum value (when written as UTF-8).
  char const* truncated_min[] = {"yyyyyyyy",
                                 "\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f",
                                 "\xf7\xbf\xbf\xbf",
                                 "éééé",
                                 "߿߿߿߿",
                                 "ࠀࠀ",
                                 "\xef\xbf\xbf\xef\xbf\xbf",
                                 "𐀀𐀀",
                                 "\xf4\x8f\xbf\xbf\xf4\x8f\xbf\xbf",
                                 "\xf7\xbf\xbf\xbf"};

  char const* truncated_max[] = {"yyyyyyyz",
                                 "\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x80",
                                 "\xff\xff\xff\xff\xff\xff\xff\xff\xff",
                                 "éééê",
                                 "߿߿߿߿߿",
                                 "ࠀࠁ",
                                 "\xef\xbf\xbf\xef\xbf\xbf\xef\xbf\xbf",
                                 "𐀀𐀁",
                                 "\xf4\x8f\xbf\xbf\xf4\x90\x80\x80",
                                 "\xf7\xbf\xbf\xbf\xf7\xbf\xbf\xbf\xf7\xbf\xbf\xbf"};

  auto cols = [&]() {
    using string_wrapper = column_wrapper<cudf::string_view>;
    std::vector<std::unique_ptr<column>> cols;
    for (auto const str : coldata) {
      cols.push_back(string_wrapper{str}.release());
    }
    return cols;
  }();
  auto expected = std::make_unique<table>(std::move(cols));

  auto const filepath = temp_env->get_temp_filepath("CheckColumnIndexTruncation.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected->view())
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .column_index_truncate_length(8);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      // check trunc(page.min) <= stats.min && trun(page.max) >= stats.max
      auto const ptype = fmd.schema[c + 1].type;
      auto const ctype = fmd.schema[c + 1].converted_type;
      EXPECT_TRUE(compare_binary(ci.min_values[0], stats.min_value, ptype, ctype) <= 0);
      EXPECT_TRUE(compare_binary(ci.max_values[0], stats.max_value, ptype, ctype) >= 0);

      // check that truncated values == expected
      EXPECT_EQ(memcmp(ci.min_values[0].data(), truncated_min[c], ci.min_values[0].size()), 0);
      EXPECT_EQ(memcmp(ci.max_values[0].data(), truncated_max[c], ci.max_values[0].size()), 0);
    }
  }
}

TEST_F(ParquetWriterTest, BinaryColumnIndexTruncation)
{
  std::vector<uint8_t> truncated_min[] = {{0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe},
                                          {0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
                                          {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}};

  std::vector<uint8_t> truncated_max[] = {{0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xff},
                                          {0xff},
                                          {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}};

  cudf::test::lists_column_wrapper<uint8_t> col0{
    {0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe}};
  cudf::test::lists_column_wrapper<uint8_t> col1{
    {0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}};
  cudf::test::lists_column_wrapper<uint8_t> col2{
    {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}};

  auto expected = table_view{{col0, col1, col2}};

  cudf::io::table_input_metadata output_metadata(expected);
  output_metadata.column_metadata[0].set_name("col_binary0").set_output_as_binary(true);
  output_metadata.column_metadata[1].set_name("col_binary1").set_output_as_binary(true);
  output_metadata.column_metadata[2].set_name("col_binary2").set_output_as_binary(true);

  auto const filepath = temp_env->get_temp_filepath("BinaryColumnIndexTruncation.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(output_metadata))
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .column_index_truncate_length(8);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      // check trunc(page.min) <= stats.min && trun(page.max) >= stats.max
      auto const ptype = fmd.schema[c + 1].type;
      auto const ctype = fmd.schema[c + 1].converted_type;
      EXPECT_TRUE(compare_binary(ci.min_values[0], stats.min_value, ptype, ctype) <= 0);
      EXPECT_TRUE(compare_binary(ci.max_values[0], stats.max_value, ptype, ctype) >= 0);

      // check that truncated values == expected
      EXPECT_EQ(ci.min_values[0], truncated_min[c]);
      EXPECT_EQ(ci.max_values[0], truncated_max[c]);
    }
  }
}

TEST_F(ParquetReaderTest, EmptyColumnsParam)
{
  srand(31337);
  auto const expected = create_random_fixed_table<int>(2, 4, false);

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&out_buffer}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{out_buffer.data(), out_buffer.size()})
      .columns({});
  auto const result = cudf::io::read_parquet(read_opts);

  EXPECT_EQ(result.tbl->num_columns(), 0);
  EXPECT_EQ(result.tbl->num_rows(), 0);
}

TEST_F(ParquetReaderTest, BinaryAsStrings)
{
  std::vector<char const*> strings{
    "Monday", "Wednesday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  auto const num_rows = strings.size();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);
  auto seq_col3 = random_values<uint8_t>(num_rows);
  auto validity = cudf::test::iterators::no_nulls();

  column_wrapper<int> int_col{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<cudf::string_view> string_col{strings.begin(), strings.end()};
  column_wrapper<float> float_col{seq_col2.begin(), seq_col2.end(), validity};
  cudf::test::lists_column_wrapper<uint8_t> list_int_col{
    {'M', 'o', 'n', 'd', 'a', 'y'},
    {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'M', 'o', 'n', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'u', 'n', 'd', 'a', 'y'}};

  auto output = table_view{{int_col, string_col, float_col, string_col, list_int_col}};
  cudf::io::table_input_metadata output_metadata(output);
  output_metadata.column_metadata[0].set_name("col_other");
  output_metadata.column_metadata[1].set_name("col_string");
  output_metadata.column_metadata[2].set_name("col_float");
  output_metadata.column_metadata[3].set_name("col_string2").set_output_as_binary(true);
  output_metadata.column_metadata[4].set_name("col_binary").set_output_as_binary(true);

  auto filepath = temp_env->get_temp_filepath("BinaryReadStrings.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, output)
      .metadata(std::move(output_metadata));
  cudf::io::write_parquet(out_opts);

  auto expected_string = table_view{{int_col, string_col, float_col, string_col, string_col}};
  auto expected_mixed  = table_view{{int_col, string_col, float_col, list_int_col, list_int_col}};

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema({{}, {}, {}, {}, {}});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_string, result.tbl->view());

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_string, result.tbl->view());

  std::vector<cudf::io::reader_column_schema> md{
    {},
    {},
    {},
    cudf::io::reader_column_schema().set_convert_binary_to_strings(false),
    cudf::io::reader_column_schema().set_convert_binary_to_strings(false)};

  cudf::io::parquet_reader_options mixed_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema(md);
  result = cudf::io::read_parquet(mixed_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_mixed, result.tbl->view());
}

TEST_F(ParquetReaderTest, NestedByteArray)
{
  constexpr auto num_rows = 8;

  auto seq_col0       = random_values<int>(num_rows);
  auto seq_col2       = random_values<float>(num_rows);
  auto seq_col3       = random_values<uint8_t>(num_rows);
  auto const validity = cudf::test::iterators::no_nulls();

  column_wrapper<int> int_col{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<float> float_col{seq_col2.begin(), seq_col2.end(), validity};
  cudf::test::lists_column_wrapper<uint8_t> list_list_int_col{
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'}, {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'u', 'n', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'u', 'n', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'}, {'F', 'r', 'i', 'd', 'a', 'y'}}};

  auto const expected = table_view{{int_col, float_col, list_list_int_col}};
  cudf::io::table_input_metadata output_metadata(expected);
  output_metadata.column_metadata[0].set_name("col_other");
  output_metadata.column_metadata[1].set_name("col_float");
  output_metadata.column_metadata[2].set_name("col_binary").child(1).set_output_as_binary(true);

  auto filepath = temp_env->get_temp_filepath("NestedByteArray.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(output_metadata));
  cudf::io::write_parquet(out_opts);

  auto source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  EXPECT_EQ(fmd.schema[5].type, cudf::io::parquet::Type::BYTE_ARRAY);

  std::vector<cudf::io::reader_column_schema> md{
    {},
    {},
    cudf::io::reader_column_schema().add_child(
      cudf::io::reader_column_schema().set_convert_binary_to_strings(false))};

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema(md);
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetWriterTest, ByteArrayStats)
{
  // check that byte array min and max statistics are written as expected. If a byte array is
  // written as a string, max utf8 is 0xf7bfbfbf and so the minimum value will be set to that value
  // instead of a potential minimum higher than that.
  std::vector<uint8_t> expected_col0_min{0xf0};
  std::vector<uint8_t> expected_col0_max{0xf0, 0xf5, 0xf5};
  std::vector<uint8_t> expected_col1_min{0xfe, 0xfe, 0xfe};
  std::vector<uint8_t> expected_col1_max{0xfe, 0xfe, 0xfe};

  cudf::test::lists_column_wrapper<uint8_t> list_int_col0{
    {0xf0}, {0xf0, 0xf5, 0xf3}, {0xf0, 0xf5, 0xf5}};
  cudf::test::lists_column_wrapper<uint8_t> list_int_col1{
    {0xfe, 0xfe, 0xfe}, {0xfe, 0xfe, 0xfe}, {0xfe, 0xfe, 0xfe}};

  auto expected = table_view{{list_int_col0, list_int_col1}};
  cudf::io::table_input_metadata output_metadata(expected);
  output_metadata.column_metadata[0].set_name("col_binary0").set_output_as_binary(true);
  output_metadata.column_metadata[1].set_name("col_binary1").set_output_as_binary(true);

  auto filepath = temp_env->get_temp_filepath("ByteArrayStats.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(output_metadata));
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema({{}, {}});
  auto result = cudf::io::read_parquet(in_opts);

  auto source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  EXPECT_EQ(fmd.schema[1].type, cudf::io::parquet::Type::BYTE_ARRAY);
  EXPECT_EQ(fmd.schema[2].type, cudf::io::parquet::Type::BYTE_ARRAY);

  auto const stats0 = get_statistics(fmd.row_groups[0].columns[0]);
  auto const stats1 = get_statistics(fmd.row_groups[0].columns[1]);

  EXPECT_EQ(expected_col0_min, stats0.min_value);
  EXPECT_EQ(expected_col0_max, stats0.max_value);
  EXPECT_EQ(expected_col1_min, stats1.min_value);
  EXPECT_EQ(expected_col1_max, stats1.max_value);
}

TEST_F(ParquetReaderTest, StructByteArray)
{
  constexpr auto num_rows = 100;

  auto seq_col0       = random_values<uint8_t>(num_rows);
  auto const validity = cudf::test::iterators::no_nulls();

  column_wrapper<uint8_t> int_col{seq_col0.begin(), seq_col0.end(), validity};
  cudf::test::lists_column_wrapper<uint8_t> list_of_int{{seq_col0.begin(), seq_col0.begin() + 50},
                                                        {seq_col0.begin() + 50, seq_col0.end()}};
  auto struct_col = cudf::test::structs_column_wrapper{{list_of_int}, validity};

  auto const expected = table_view{{struct_col}};
  EXPECT_EQ(1, expected.num_columns());
  cudf::io::table_input_metadata output_metadata(expected);
  output_metadata.column_metadata[0]
    .set_name("struct_binary")
    .child(0)
    .set_name("a")
    .set_output_as_binary(true);

  auto filepath = temp_env->get_temp_filepath("StructByteArray.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(output_metadata));
  cudf::io::write_parquet(out_opts);

  std::vector<cudf::io::reader_column_schema> md{cudf::io::reader_column_schema().add_child(
    cudf::io::reader_column_schema().set_convert_binary_to_strings(false))};

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema(md);
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetReaderTest, NestingOptimizationTest)
{
  // test nesting levels > cudf::io::parquet::gpu::max_cacheable_nesting_decode_info deep.
  constexpr cudf::size_type num_nesting_levels = 16;
  static_assert(num_nesting_levels > cudf::io::parquet::gpu::max_cacheable_nesting_decode_info);
  constexpr cudf::size_type rows_per_level = 2;

  constexpr cudf::size_type num_values = (1 << num_nesting_levels) * rows_per_level;
  auto value_iter                      = thrust::make_counting_iterator(0);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 2; });
  cudf::test::fixed_width_column_wrapper<int> values(value_iter, value_iter + num_values, validity);

  // ~256k values with num_nesting_levels = 16
  int total_values_produced = num_values;
  auto prev_col             = values.release();
  for (int idx = 0; idx < num_nesting_levels; idx++) {
    auto const depth    = num_nesting_levels - idx;
    auto const num_rows = (1 << (num_nesting_levels - idx));

    auto offsets_iter = cudf::detail::make_counting_transform_iterator(
      0, [depth, rows_per_level](cudf::size_type i) { return i * rows_per_level; });
    total_values_produced += (num_rows + 1);

    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets(offsets_iter,
                                                                    offsets_iter + num_rows + 1);
    auto c   = cudf::make_lists_column(num_rows, offsets.release(), std::move(prev_col), 0, {});
    prev_col = std::move(c);
  }
  auto const& expect = prev_col;

  auto filepath = temp_env->get_temp_filepath("NestingDecodeCache.parquet");
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table_view{{*expect}});
  cudf::io::write_parquet(opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expect, result.tbl->get_column(0));
}

TEST_F(ParquetWriterTest, SingleValueDictionaryTest)
{
  constexpr unsigned int expected_bits = 1;
  constexpr unsigned int nrows         = 1'000'000U;

  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "a unique string value suffixed with 1"; });
  auto const col0     = cudf::test::strings_column_wrapper(elements, elements + nrows);
  auto const expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("SingleValueDictionaryTest.parquet");
  // set row group size so that there will be only one row group
  // no compression so we can easily read page data
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::NONE)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .row_group_size_rows(nrows);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());

  // make sure dictionary was used
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  auto used_dict = [&fmd]() {
    for (auto enc : fmd.row_groups[0].columns[0].meta_data.encodings) {
      if (enc == cudf::io::parquet::Encoding::PLAIN_DICTIONARY or
          enc == cudf::io::parquet::Encoding::RLE_DICTIONARY) {
        return true;
      }
    }
    return false;
  };
  EXPECT_TRUE(used_dict());

  // and check that the correct number of bits was used
  auto const oi    = read_offset_index(source, fmd.row_groups[0].columns[0]);
  auto const nbits = read_dict_bits(source, oi.page_locations[0]);
  EXPECT_EQ(nbits, expected_bits);
}

TEST_F(ParquetWriterTest, DictionaryNeverTest)
{
  constexpr unsigned int nrows = 1'000U;

  // only one value, so would normally use dictionary
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "a unique string value suffixed with 1"; });
  auto const col0     = cudf::test::strings_column_wrapper(elements, elements + nrows);
  auto const expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("DictionaryNeverTest.parquet");
  // no compression so we can easily read page data
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::NONE)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());

  // make sure dictionary was not used
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  auto used_dict = [&fmd]() {
    for (auto enc : fmd.row_groups[0].columns[0].meta_data.encodings) {
      if (enc == cudf::io::parquet::Encoding::PLAIN_DICTIONARY or
          enc == cudf::io::parquet::Encoding::RLE_DICTIONARY) {
        return true;
      }
    }
    return false;
  };
  EXPECT_FALSE(used_dict());
}

TEST_F(ParquetWriterTest, DictionaryAdaptiveTest)
{
  constexpr unsigned int nrows = 65'536U;
  // cardinality is chosen to result in a dictionary > 1MB in size
  constexpr unsigned int cardinality = 32'768U;

  // single value will have a small dictionary
  auto elements0 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "a unique string value suffixed with 1"; });
  auto const col0 = cudf::test::strings_column_wrapper(elements0, elements0 + nrows);

  // high cardinality will have a large dictionary
  auto elements1  = cudf::detail::make_counting_transform_iterator(0, [cardinality](auto i) {
    return "a unique string value suffixed with " + std::to_string(i % cardinality);
  });
  auto const col1 = cudf::test::strings_column_wrapper(elements1, elements1 + nrows);

  auto const expected = table_view{{col0, col1}};

  auto const filepath = temp_env->get_temp_filepath("DictionaryAdaptiveTest.parquet");
  // no compression so we can easily read page data
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::ZSTD)
      .dictionary_policy(cudf::io::dictionary_policy::ADAPTIVE);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());

  // make sure dictionary was used as expected. col0 should use one,
  // col1 should not.
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  auto used_dict = [&fmd](int col) {
    for (auto enc : fmd.row_groups[0].columns[col].meta_data.encodings) {
      if (enc == cudf::io::parquet::Encoding::PLAIN_DICTIONARY or
          enc == cudf::io::parquet::Encoding::RLE_DICTIONARY) {
        return true;
      }
    }
    return false;
  };
  EXPECT_TRUE(used_dict(0));
  EXPECT_FALSE(used_dict(1));
}

TEST_F(ParquetWriterTest, DictionaryAlwaysTest)
{
  constexpr unsigned int nrows = 65'536U;
  // cardinality is chosen to result in a dictionary > 1MB in size
  constexpr unsigned int cardinality = 32'768U;

  // single value will have a small dictionary
  auto elements0 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "a unique string value suffixed with 1"; });
  auto const col0 = cudf::test::strings_column_wrapper(elements0, elements0 + nrows);

  // high cardinality will have a large dictionary
  auto elements1  = cudf::detail::make_counting_transform_iterator(0, [cardinality](auto i) {
    return "a unique string value suffixed with " + std::to_string(i % cardinality);
  });
  auto const col1 = cudf::test::strings_column_wrapper(elements1, elements1 + nrows);

  auto const expected = table_view{{col0, col1}};

  auto const filepath = temp_env->get_temp_filepath("DictionaryAlwaysTest.parquet");
  // no compression so we can easily read page data
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::ZSTD)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());

  // make sure dictionary was used for both columns
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  auto used_dict = [&fmd](int col) {
    for (auto enc : fmd.row_groups[0].columns[col].meta_data.encodings) {
      if (enc == cudf::io::parquet::Encoding::PLAIN_DICTIONARY or
          enc == cudf::io::parquet::Encoding::RLE_DICTIONARY) {
        return true;
      }
    }
    return false;
  };
  EXPECT_TRUE(used_dict(0));
  EXPECT_TRUE(used_dict(1));
}

TEST_F(ParquetWriterTest, DictionaryPageSizeEst)
{
  // one page
  constexpr unsigned int nrows = 20'000U;

  // this test is creating a pattern of repeating then non-repeating values to trigger
  // a "worst-case" for page size estimation in the presence of a dictionary. have confirmed
  // that this fails for values over 16 in the final term of `max_RLE_page_size()`.
  // The output of the iterator will be 'CCCCCRRRRRCCCCCRRRRR...` where 'C' is a changing
  // value, and 'R' repeats. The encoder will turn this into a literal run of 8 values
  // (`CCCCCRRR`) followed by a repeated run of 2 (`RR`). This pattern then repeats, getting
  // as close as possible to a condition of repeated 8 value literal runs.
  auto elements0  = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    if ((i / 5) % 2 == 1) {
      return std::string("non-unique string");
    } else {
      return "a unique string value suffixed with " + std::to_string(i);
    }
  });
  auto const col0 = cudf::test::strings_column_wrapper(elements0, elements0 + nrows);

  auto const expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("DictionaryPageSizeEst.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::ZSTD)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_P(ParquetSizedTest, DictionaryTest)
{
  unsigned int const cardinality = (1 << (GetParam() - 1)) + 1;
  unsigned int const nrows       = std::max(cardinality * 3 / 2, 3'000'000U);

  auto elements       = cudf::detail::make_counting_transform_iterator(0, [cardinality](auto i) {
    return "a unique string value suffixed with " + std::to_string(i % cardinality);
  });
  auto const col0     = cudf::test::strings_column_wrapper(elements, elements + nrows);
  auto const expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("DictionaryTest.parquet");
  // set row group size so that there will be only one row group
  // no compression so we can easily read page data
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::NONE)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
      .row_group_size_rows(nrows)
      .row_group_size_bytes(512 * 1024 * 1024);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());

  // make sure dictionary was used
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  auto used_dict = [&fmd]() {
    for (auto enc : fmd.row_groups[0].columns[0].meta_data.encodings) {
      if (enc == cudf::io::parquet::Encoding::PLAIN_DICTIONARY or
          enc == cudf::io::parquet::Encoding::RLE_DICTIONARY) {
        return true;
      }
    }
    return false;
  };
  EXPECT_TRUE(used_dict());

  // and check that the correct number of bits was used
  auto const oi    = read_offset_index(source, fmd.row_groups[0].columns[0]);
  auto const nbits = read_dict_bits(source, oi.page_locations[0]);
  EXPECT_EQ(nbits, GetParam());
}

TYPED_TEST(ParquetReaderSourceTest, BufferSourceTypes)
{
  using T = TypeParam;

  srand(31337);
  auto table = create_random_fixed_table<int>(5, 5, true);

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer), *table);
  cudf::io::write_parquet(out_opts);

  {
    cudf::io::parquet_reader_options in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(
        cudf::host_span<T>(reinterpret_cast<T*>(out_buffer.data()), out_buffer.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*table, result.tbl->view());
  }

  {
    cudf::io::parquet_reader_options in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(cudf::host_span<T const>(
        reinterpret_cast<T const*>(out_buffer.data()), out_buffer.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*table, result.tbl->view());
  }
}

TYPED_TEST(ParquetReaderSourceTest, BufferSourceArrayTypes)
{
  using T = TypeParam;

  srand(31337);
  auto table = create_random_fixed_table<int>(5, 5, true);

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer), *table);
  cudf::io::write_parquet(out_opts);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table, *table}));

  {
    auto spans = std::vector<cudf::host_span<T>>{
      cudf::host_span<T>(reinterpret_cast<T*>(out_buffer.data()), out_buffer.size()),
      cudf::host_span<T>(reinterpret_cast<T*>(out_buffer.data()), out_buffer.size())};
    cudf::io::parquet_reader_options in_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info(cudf::host_span<cudf::host_span<T>>(spans.data(), spans.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*full_table, result.tbl->view());
  }

  {
    auto spans = std::vector<cudf::host_span<T const>>{
      cudf::host_span<T const>(reinterpret_cast<T const*>(out_buffer.data()), out_buffer.size()),
      cudf::host_span<T const>(reinterpret_cast<T const*>(out_buffer.data()), out_buffer.size())};
    cudf::io::parquet_reader_options in_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info(cudf::host_span<cudf::host_span<T const>>(spans.data(), spans.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*full_table, result.tbl->view());
  }
}

TEST_F(ParquetWriterTest, UserNullability)
{
  auto weight_col = cudf::test::fixed_width_column_wrapper<float>{{57.5, 51.1, 15.3}};
  auto ages_col   = cudf::test::fixed_width_column_wrapper<int32_t>{{30, 27, 5}};
  auto struct_col = cudf::test::structs_column_wrapper{weight_col, ages_col};

  auto expected = table_view({struct_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_nullability(false);
  expected_metadata.column_metadata[0].child(0).set_nullability(true);

  auto filepath = temp_env->get_temp_filepath("SingleWriteNullable.parquet");
  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_parquet(write_opts);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  EXPECT_FALSE(result.tbl->view().column(0).nullable());
  EXPECT_TRUE(result.tbl->view().column(0).child(0).nullable());
  EXPECT_FALSE(result.tbl->view().column(0).child(1).nullable());
}

TEST_F(ParquetWriterTest, UserNullabilityInvalid)
{
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 2; });
  auto col      = cudf::test::fixed_width_column_wrapper<double>{{57.5, 51.1, 15.3}, valids};
  auto expected = table_view({col});

  auto filepath = temp_env->get_temp_filepath("SingleWriteNullableInvalid.parquet");
  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  // Should work without the nullability option
  EXPECT_NO_THROW(cudf::io::write_parquet(write_opts));

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_nullability(false);
  write_opts.set_metadata(std::move(expected_metadata));
  // Can't write a column with nulls as not nullable
  EXPECT_THROW(cudf::io::write_parquet(write_opts), cudf::logic_error);
}

TEST_F(ParquetReaderTest, SingleLevelLists)
{
  unsigned char list_bytes[] = {
    0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0x28, 0x15, 0x28, 0x15, 0xa7, 0xce, 0x91, 0x8c, 0x06,
    0x1c, 0x15, 0x04, 0x15, 0x00, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
    0x02, 0x02, 0x00, 0x00, 0x00, 0x03, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x15,
    0x02, 0x19, 0x3c, 0x48, 0x0c, 0x73, 0x70, 0x61, 0x72, 0x6b, 0x5f, 0x73, 0x63, 0x68, 0x65, 0x6d,
    0x61, 0x15, 0x02, 0x00, 0x35, 0x00, 0x18, 0x01, 0x66, 0x15, 0x02, 0x15, 0x06, 0x4c, 0x3c, 0x00,
    0x00, 0x00, 0x15, 0x02, 0x25, 0x04, 0x18, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x00, 0x16, 0x02,
    0x19, 0x1c, 0x19, 0x1c, 0x26, 0x08, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x00, 0x06, 0x19, 0x28, 0x01,
    0x66, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x15, 0x00, 0x16, 0x04, 0x16, 0x56, 0x16, 0x56, 0x26,
    0x08, 0x3c, 0x18, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x16, 0x00,
    0x28, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0x1c, 0x15,
    0x00, 0x15, 0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x16, 0x56, 0x16, 0x02, 0x26, 0x08, 0x16, 0x56,
    0x14, 0x00, 0x00, 0x28, 0x13, 0x52, 0x41, 0x50, 0x49, 0x44, 0x53, 0x20, 0x53, 0x70, 0x61, 0x72,
    0x6b, 0x20, 0x50, 0x6c, 0x75, 0x67, 0x69, 0x6e, 0x19, 0x1c, 0x1c, 0x00, 0x00, 0x00, 0x9f, 0x00,
    0x00, 0x00, 0x50, 0x41, 0x52, 0x31};

  // read single level list reproducing parquet file
  cudf::io::parquet_reader_options read_opts = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{reinterpret_cast<char const*>(list_bytes), sizeof(list_bytes)});
  auto table = cudf::io::read_parquet(read_opts);

  auto const c0 = table.tbl->get_column(0);
  EXPECT_TRUE(c0.type().id() == cudf::type_id::LIST);

  auto const lc    = cudf::lists_column_view(c0);
  auto const child = lc.child();
  EXPECT_TRUE(child.type().id() == cudf::type_id::INT32);
}

TEST_F(ParquetReaderTest, ChunkedSingleLevelLists)
{
  unsigned char list_bytes[] = {
    0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0x28, 0x15, 0x28, 0x15, 0xa7, 0xce, 0x91, 0x8c, 0x06,
    0x1c, 0x15, 0x04, 0x15, 0x00, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
    0x02, 0x02, 0x00, 0x00, 0x00, 0x03, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x15,
    0x02, 0x19, 0x3c, 0x48, 0x0c, 0x73, 0x70, 0x61, 0x72, 0x6b, 0x5f, 0x73, 0x63, 0x68, 0x65, 0x6d,
    0x61, 0x15, 0x02, 0x00, 0x35, 0x00, 0x18, 0x01, 0x66, 0x15, 0x02, 0x15, 0x06, 0x4c, 0x3c, 0x00,
    0x00, 0x00, 0x15, 0x02, 0x25, 0x04, 0x18, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x00, 0x16, 0x02,
    0x19, 0x1c, 0x19, 0x1c, 0x26, 0x08, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x00, 0x06, 0x19, 0x28, 0x01,
    0x66, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x15, 0x00, 0x16, 0x04, 0x16, 0x56, 0x16, 0x56, 0x26,
    0x08, 0x3c, 0x18, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x16, 0x00,
    0x28, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0x1c, 0x15,
    0x00, 0x15, 0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x16, 0x56, 0x16, 0x02, 0x26, 0x08, 0x16, 0x56,
    0x14, 0x00, 0x00, 0x28, 0x13, 0x52, 0x41, 0x50, 0x49, 0x44, 0x53, 0x20, 0x53, 0x70, 0x61, 0x72,
    0x6b, 0x20, 0x50, 0x6c, 0x75, 0x67, 0x69, 0x6e, 0x19, 0x1c, 0x1c, 0x00, 0x00, 0x00, 0x9f, 0x00,
    0x00, 0x00, 0x50, 0x41, 0x52, 0x31};

  auto reader = cudf::io::chunked_parquet_reader(
    1L << 31,
    cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{reinterpret_cast<char const*>(list_bytes), sizeof(list_bytes)}));
  int iterations = 0;
  while (reader.has_next() && iterations < 10) {
    auto chunk = reader.read_chunk();
  }
  EXPECT_TRUE(iterations < 10);
}

TEST_F(ParquetWriterTest, CompStats)
{
  auto table = create_random_fixed_table<int>(1, 100000, true);

  auto const stats = std::make_shared<cudf::io::writer_compression_statistics>();

  std::vector<char> unused_buffer;
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&unused_buffer}, table->view())
      .compression_statistics(stats);
  cudf::io::write_parquet(opts);

  EXPECT_NE(stats->num_compressed_bytes(), 0);
  EXPECT_EQ(stats->num_failed_bytes(), 0);
  EXPECT_EQ(stats->num_skipped_bytes(), 0);
  EXPECT_FALSE(std::isnan(stats->compression_ratio()));
}

TEST_F(ParquetChunkedWriterTest, CompStats)
{
  auto table = create_random_fixed_table<int>(1, 100000, true);

  auto const stats = std::make_shared<cudf::io::writer_compression_statistics>();

  std::vector<char> unused_buffer;
  cudf::io::chunked_parquet_writer_options opts =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{&unused_buffer})
      .compression_statistics(stats);
  cudf::io::parquet_chunked_writer(opts).write(*table);

  EXPECT_NE(stats->num_compressed_bytes(), 0);
  EXPECT_EQ(stats->num_failed_bytes(), 0);
  EXPECT_EQ(stats->num_skipped_bytes(), 0);
  EXPECT_FALSE(std::isnan(stats->compression_ratio()));

  auto const single_table_comp_stats = *stats;
  cudf::io::parquet_chunked_writer(opts).write(*table);

  EXPECT_EQ(stats->compression_ratio(), single_table_comp_stats.compression_ratio());
  EXPECT_EQ(stats->num_compressed_bytes(), 2 * single_table_comp_stats.num_compressed_bytes());

  EXPECT_EQ(stats->num_failed_bytes(), 0);
  EXPECT_EQ(stats->num_skipped_bytes(), 0);
}

void expect_compression_stats_empty(std::shared_ptr<cudf::io::writer_compression_statistics> stats)
{
  EXPECT_EQ(stats->num_compressed_bytes(), 0);
  EXPECT_EQ(stats->num_failed_bytes(), 0);
  EXPECT_EQ(stats->num_skipped_bytes(), 0);
  EXPECT_TRUE(std::isnan(stats->compression_ratio()));
}

TEST_F(ParquetWriterTest, CompStatsEmptyTable)
{
  auto table_no_rows = create_random_fixed_table<int>(20, 0, false);

  auto const stats = std::make_shared<cudf::io::writer_compression_statistics>();

  std::vector<char> unused_buffer;
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&unused_buffer},
                                              table_no_rows->view())
      .compression_statistics(stats);
  cudf::io::write_parquet(opts);

  expect_compression_stats_empty(stats);
}

TEST_F(ParquetChunkedWriterTest, CompStatsEmptyTable)
{
  auto table_no_rows = create_random_fixed_table<int>(20, 0, false);

  auto const stats = std::make_shared<cudf::io::writer_compression_statistics>();

  std::vector<char> unused_buffer;
  cudf::io::chunked_parquet_writer_options opts =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{&unused_buffer})
      .compression_statistics(stats);
  cudf::io::parquet_chunked_writer(opts).write(*table_no_rows);

  expect_compression_stats_empty(stats);
}

TEST_F(ParquetReaderTest, ReorderedReadMultipleFiles)
{
  constexpr auto num_rows    = 50'000;
  constexpr auto cardinality = 20'000;

  // table 1
  auto str1 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "cat " + std::to_string(i % cardinality); });
  auto cols1 = cudf::test::strings_column_wrapper(str1, str1 + num_rows);

  auto int1 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % cardinality; });
  auto coli1 = cudf::test::fixed_width_column_wrapper<int>(int1, int1 + num_rows);

  auto const expected1 = table_view{{cols1, coli1}};
  auto const swapped1  = table_view{{coli1, cols1}};

  auto const filepath1 = temp_env->get_temp_filepath("LargeReorderedRead1.parquet");
  auto out_opts1 =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath1}, expected1)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(out_opts1);

  // table 2
  auto str2 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "dog " + std::to_string(i % cardinality); });
  auto cols2 = cudf::test::strings_column_wrapper(str2, str2 + num_rows);

  auto int2 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return (i % cardinality) + cardinality; });
  auto coli2 = cudf::test::fixed_width_column_wrapper<int>(int2, int2 + num_rows);

  auto const expected2 = table_view{{cols2, coli2}};
  auto const swapped2  = table_view{{coli2, cols2}};

  auto const filepath2 = temp_env->get_temp_filepath("LargeReorderedRead2.parquet");
  auto out_opts2 =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath2}, expected2)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(out_opts2);

  // read in both files swapping the columns
  auto read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{{filepath1, filepath2}})
      .columns({"_col1", "_col0"});
  auto result = cudf::io::read_parquet(read_opts);
  auto sliced = cudf::slice(result.tbl->view(), {0, num_rows, num_rows, 2 * num_rows});
  CUDF_TEST_EXPECT_TABLES_EQUAL(sliced[0], swapped1);
  CUDF_TEST_EXPECT_TABLES_EQUAL(sliced[1], swapped2);
}

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

TEST_F(ParquetMetadataReaderTest, TestBasic)
{
  auto const num_rows = 1200;

  auto ints   = random_values<int>(num_rows);
  auto floats = random_values<float>(num_rows);
  column_wrapper<int> int_col(ints.begin(), ints.end());
  column_wrapper<float> float_col(floats.begin(), floats.end());

  table_view expected({int_col, float_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("int_col");
  expected_metadata.column_metadata[1].set_name("float_col");

  auto filepath = temp_env->get_temp_filepath("MetadataTest.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_parquet(out_opts);

  auto meta = read_parquet_metadata(cudf::io::source_info{filepath});
  EXPECT_EQ(meta.num_rows(), num_rows);

  std::string expected_schema = R"(schema
 int_col
 float_col
)";
  EXPECT_EQ(expected_schema, print(meta.schema().root()));

  EXPECT_EQ(meta.schema().root().name(), "schema");
  EXPECT_EQ(meta.schema().root().type_kind(), cudf::io::parquet::TypeKind::UNDEFINED_TYPE);
  ASSERT_EQ(meta.schema().root().num_children(), 2);

  EXPECT_EQ(meta.schema().root().child(0).name(), "int_col");
  EXPECT_EQ(meta.schema().root().child(1).name(), "float_col");
}

TEST_F(ParquetMetadataReaderTest, TestNested)
{
  auto const num_rows       = 1200;
  auto const lists_per_row  = 4;
  auto const num_child_rows = num_rows * lists_per_row;

  auto keys = random_values<int>(num_child_rows);
  auto vals = random_values<float>(num_child_rows);
  column_wrapper<int> keys_col(keys.begin(), keys.end());
  column_wrapper<float> vals_col(vals.begin(), vals.end());
  auto s_col = cudf::test::structs_column_wrapper({keys_col, vals_col}).release();

  std::vector<int> row_offsets(num_rows + 1);
  for (int idx = 0; idx < num_rows + 1; ++idx) {
    row_offsets[idx] = idx * lists_per_row;
  }
  column_wrapper<int> offsets(row_offsets.begin(), row_offsets.end());

  auto list_col =
    cudf::make_lists_column(num_rows, offsets.release(), std::move(s_col), 0, rmm::device_buffer{});

  table_view expected({*list_col, *list_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("maps");
  expected_metadata.column_metadata[0].set_list_column_as_map();
  expected_metadata.column_metadata[1].set_name("lists");
  expected_metadata.column_metadata[1].child(1).child(0).set_name("int_field");
  expected_metadata.column_metadata[1].child(1).child(1).set_name("float_field");

  auto filepath = temp_env->get_temp_filepath("MetadataTest.orc");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_parquet(out_opts);

  auto meta = read_parquet_metadata(cudf::io::source_info{filepath});
  EXPECT_EQ(meta.num_rows(), num_rows);

  std::string expected_schema = R"(schema
 maps
  key_value
   key
   value
 lists
  list
   element
    int_field
    float_field
)";
  EXPECT_EQ(expected_schema, print(meta.schema().root()));

  EXPECT_EQ(meta.schema().root().name(), "schema");
  EXPECT_EQ(meta.schema().root().type_kind(),
            cudf::io::parquet::TypeKind::UNDEFINED_TYPE);  // struct
  ASSERT_EQ(meta.schema().root().num_children(), 2);

  auto const& out_map_col = meta.schema().root().child(0);
  EXPECT_EQ(out_map_col.name(), "maps");
  EXPECT_EQ(out_map_col.type_kind(), cudf::io::parquet::TypeKind::UNDEFINED_TYPE);  // map

  ASSERT_EQ(out_map_col.num_children(), 1);
  EXPECT_EQ(out_map_col.child(0).name(), "key_value");       // key_value (named in parquet writer)
  ASSERT_EQ(out_map_col.child(0).num_children(), 2);
  EXPECT_EQ(out_map_col.child(0).child(0).name(), "key");    // key (named in parquet writer)
  EXPECT_EQ(out_map_col.child(0).child(1).name(), "value");  // value (named in parquet writer)
  EXPECT_EQ(out_map_col.child(0).child(0).type_kind(), cudf::io::parquet::TypeKind::INT32);  // int
  EXPECT_EQ(out_map_col.child(0).child(1).type_kind(),
            cudf::io::parquet::TypeKind::FLOAT);  // float

  auto const& out_list_col = meta.schema().root().child(1);
  EXPECT_EQ(out_list_col.name(), "lists");
  EXPECT_EQ(out_list_col.type_kind(), cudf::io::parquet::TypeKind::UNDEFINED_TYPE);  // list
  // TODO repetition type?
  ASSERT_EQ(out_list_col.num_children(), 1);
  EXPECT_EQ(out_list_col.child(0).name(), "list");  // list (named in parquet writer)
  ASSERT_EQ(out_list_col.child(0).num_children(), 1);

  auto const& out_list_struct_col = out_list_col.child(0).child(0);
  EXPECT_EQ(out_list_struct_col.name(), "element");        // elements (named in parquet writer)
  EXPECT_EQ(out_list_struct_col.type_kind(),
            cudf::io::parquet::TypeKind::UNDEFINED_TYPE);  // struct
  ASSERT_EQ(out_list_struct_col.num_children(), 2);

  auto const& out_int_col = out_list_struct_col.child(0);
  EXPECT_EQ(out_int_col.name(), "int_field");
  EXPECT_EQ(out_int_col.type_kind(), cudf::io::parquet::TypeKind::INT32);

  auto const& out_float_col = out_list_struct_col.child(1);
  EXPECT_EQ(out_float_col.name(), "float_field");
  EXPECT_EQ(out_float_col.type_kind(), cudf::io::parquet::TypeKind::FLOAT);
}

TEST_F(ParquetWriterTest, NoNullsAsNonNullable)
{
  auto valids = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });
  column_wrapper<int32_t> col{{1, 2, 3}, valids};
  table_view expected({col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_nullability(false);

  auto filepath = temp_env->get_temp_filepath("NonNullable.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  // Writer should be able to write a column without nulls as non-nullable
  EXPECT_NO_THROW(cudf::io::write_parquet(out_opts));
}

TEST_F(ParquetReaderTest, FilterSimple)
{
  srand(31337);
  auto written_table = create_random_fixed_table<int>(9, 9, false);

  auto filepath = temp_env->get_temp_filepath("FilterSimple.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *written_table);
  cudf::io::write_parquet(args);

  // Filtering AST - table[0] < RAND_MAX/2
  auto literal_value     = cudf::numeric_scalar<decltype(RAND_MAX)>(RAND_MAX / 2);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_reference(0);
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto predicate = cudf::compute_column(*written_table, filter_expression);
  EXPECT_EQ(predicate->view().type().id(), cudf::type_id::BOOL8)
    << "Predicate filter should return a boolean";
  auto expected = cudf::apply_boolean_mask(*written_table, *predicate);
  // To make sure AST filters out some elements
  EXPECT_LT(expected->num_rows(), written_table->num_rows());

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(filter_expression);
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

auto create_parquet_with_stats(std::string const& filename)
{
  auto col0 = testdata::ascending<uint32_t>();
  auto col1 = testdata::descending<int64_t>();
  auto col2 = testdata::unordered<double>();

  auto const expected = table_view{{col0, col1, col2}};

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_uint32");
  expected_metadata.column_metadata[1].set_name("col_int64");
  expected_metadata.column_metadata[2].set_name("col_double");

  auto const filepath = temp_env->get_temp_filepath(filename);
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata))
      .row_group_size_rows(8000)
      .stats_level(cudf::io::statistics_freq::STATISTICS_ROWGROUP);
  cudf::io::write_parquet(out_opts);

  std::vector<std::unique_ptr<column>> columns;
  columns.push_back(col0.release());
  columns.push_back(col1.release());
  columns.push_back(col2.release());

  return std::pair{cudf::table{std::move(columns)}, filepath};
}

TEST_F(ParquetReaderTest, FilterIdentity)
{
  auto [src, filepath] = create_parquet_with_stats("FilterIdentity.parquet");

  // Filtering AST - identity function, always true.
  auto literal_value     = cudf::numeric_scalar<bool>(true);
  auto literal           = cudf::ast::literal(literal_value);
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, literal);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(filter_expression);
  auto result = cudf::io::read_parquet(read_opts);

  cudf::io::parquet_reader_options read_opts2 =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result2 = cudf::io::read_parquet(read_opts2);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *result2.tbl);
}

TEST_F(ParquetReaderTest, FilterReferenceExpression)
{
  auto [src, filepath] = create_parquet_with_stats("FilterReferenceExpression.parquet");
  // Filtering AST - table[0] < 150
  auto literal_value     = cudf::numeric_scalar<uint32_t>(150);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_reference(0);
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Expected result
  auto predicate = cudf::compute_column(src, filter_expression);
  auto expected  = cudf::apply_boolean_mask(src, *predicate);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(filter_expression);
  auto result = cudf::io::read_parquet(read_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetReaderTest, FilterNamedExpression)
{
  auto [src, filepath] = create_parquet_with_stats("NamedExpression.parquet");
  // Filtering AST - table["col_uint32"] < 150
  auto literal_value  = cudf::numeric_scalar<uint32_t>(150);
  auto literal        = cudf::ast::literal(literal_value);
  auto col_name_0     = cudf::ast::column_name_reference("col_uint32");
  auto parquet_filter = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_name_0, literal);
  auto col_ref_0      = cudf::ast::column_reference(0);
  auto table_filter   = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Expected result
  auto predicate = cudf::compute_column(src, table_filter);
  auto expected  = cudf::apply_boolean_mask(src, *predicate);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(parquet_filter);
  auto result = cudf::io::read_parquet(read_opts);

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

// Test for Types - numeric, chrono, string.
template <typename T>
struct ParquetReaderPredicatePushdownTest : public ParquetReaderTest {};

// These chrono types are not supported because parquet writer does not have a type to represent
// them.
using UnsupportedChronoTypes =
  cudf::test::Types<cudf::timestamp_s, cudf::duration_D, cudf::duration_s>;
// Also fixed point types unsupported, because AST does not support them yet.
using SupportedTestTypes = cudf::test::RemoveIf<cudf::test::ContainedIn<UnsupportedChronoTypes>,
                                                cudf::test::ComparableTypes>;

TYPED_TEST_SUITE(ParquetReaderPredicatePushdownTest, SupportedTestTypes);

template <typename T>
auto create_parquet_typed_with_stats(std::string const& filename)
{
  auto col0 = testdata::ascending<T>();
  auto col1 = testdata::descending<T>();
  auto col2 = testdata::unordered<T>();

  auto const written_table = table_view{{col0, col1, col2}};
  auto const filepath      = temp_env->get_temp_filepath("FilterTyped.parquet");
  {
    cudf::io::table_input_metadata expected_metadata(written_table);
    expected_metadata.column_metadata[0].set_name("col0");
    expected_metadata.column_metadata[1].set_name("col1");
    expected_metadata.column_metadata[2].set_name("col2");

    const cudf::io::parquet_writer_options out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .metadata(std::move(expected_metadata))
        .row_group_size_rows(8000);
    cudf::io::write_parquet(out_opts);
  }

  std::vector<std::unique_ptr<column>> columns;
  columns.push_back(col0.release());
  columns.push_back(col1.release());
  columns.push_back(col2.release());

  return std::pair{cudf::table{std::move(columns)}, filepath};
}

TYPED_TEST(ParquetReaderPredicatePushdownTest, FilterTyped)
{
  using T = TypeParam;

  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterTyped.parquet");
  auto const written_table   = src.view();

  // Filtering AST
  auto literal_value = []() {
    if constexpr (cudf::is_timestamp<T>()) {
      // table[0] < 10000 timestamp days/seconds/milliseconds/microseconds/nanoseconds
      return cudf::timestamp_scalar<T>(T(typename T::duration(10000)));  // i (0-20,000)
    } else if constexpr (cudf::is_duration<T>()) {
      // table[0] < 10000 day/seconds/milliseconds/microseconds/nanoseconds
      return cudf::duration_scalar<T>(T(10000));  // i (0-20,000)
    } else if constexpr (std::is_same_v<T, cudf::string_view>) {
      // table[0] < "000010000"
      return cudf::string_scalar("000010000");  // i (0-20,000)
    } else {
      // table[0] < 0 or 100u
      return cudf::numeric_scalar<T>((100 - 100 * std::is_signed_v<T>));  // i/100 (-100-100/ 0-200)
    }
  }();
  auto literal           = cudf::ast::literal(literal_value);
  auto col_name_0        = cudf::ast::column_name_reference("col0");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_name_0, literal);
  auto col_ref_0         = cudf::ast::column_reference(0);
  auto ref_filter        = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Expected result
  auto predicate = cudf::compute_column(written_table, ref_filter);
  EXPECT_EQ(predicate->view().type().id(), cudf::type_id::BOOL8)
    << "Predicate filter should return a boolean";
  auto expected = cudf::apply_boolean_mask(written_table, *predicate);

  // Reading with Predicate Pushdown
  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(filter_expression);
  auto result       = cudf::io::read_parquet(read_opts);
  auto result_table = result.tbl->view();

  // tests
  EXPECT_EQ(int(written_table.column(0).type().id()), int(result_table.column(0).type().id()))
    << "col0 type mismatch";
  // To make sure AST filters out some elements
  EXPECT_LT(expected->num_rows(), written_table.num_rows());
  EXPECT_EQ(result_table.num_rows(), expected->num_rows());
  EXPECT_EQ(result_table.num_columns(), expected->num_columns());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result_table);
}

TEST_F(ParquetReaderTest, FilterMultiple1)
{
  using T = cudf::string_view;

  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterMultiple1.parquet");
  auto const written_table   = src.view();

  // Filtering AST - 10000 < table[0] < 12000
  std::string const low  = "000010000";
  std::string const high = "000012000";
  auto lov               = cudf::string_scalar(low, true);
  auto hiv               = cudf::string_scalar(high, true);
  auto filter_col        = cudf::ast::column_reference(0);
  auto lo_lit            = cudf::ast::literal(lov);
  auto hi_lit            = cudf::ast::literal(hiv);
  auto expr_1 = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col, lo_lit);
  auto expr_2 = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col, hi_lit);
  auto expr_3 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr_3);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_3);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterMultiple2)
{
  // multiple conditions on same column.
  using T = cudf::string_view;

  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterMultiple2.parquet");
  auto const written_table   = src.view();
  // 0-8000, 8001-16000, 16001-20000

  // Filtering AST
  // (table[0] >= "000010000" AND table[0] < "000012000") OR
  // (table[0] >= "000017000" AND table[0] < "000019000")
  std::string const low1  = "000010000";
  std::string const high1 = "000012000";
  auto lov                = cudf::string_scalar(low1, true);
  auto hiv                = cudf::string_scalar(high1, true);
  auto filter_col         = cudf::ast::column_reference(0);
  auto lo_lit             = cudf::ast::literal(lov);
  auto hi_lit             = cudf::ast::literal(hiv);
  auto expr_1 = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col, lo_lit);
  auto expr_2 = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col, hi_lit);
  auto expr_3 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);
  std::string const low2  = "000017000";
  std::string const high2 = "000019000";
  auto lov2               = cudf::string_scalar(low2, true);
  auto hiv2               = cudf::string_scalar(high2, true);
  auto lo_lit2            = cudf::ast::literal(lov2);
  auto hi_lit2            = cudf::ast::literal(hiv2);
  auto expr_4 = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col, lo_lit2);
  auto expr_5 = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col, hi_lit2);
  auto expr_6 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4, expr_5);
  auto expr_7 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr_7);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_7);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterMultiple3)
{
  // multiple conditions with reference to multiple columns.
  // index and name references mixed.
  using T                    = uint32_t;
  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterMultiple3.parquet");
  auto const written_table   = src.view();

  // Filtering AST - (table[0] >= 70 AND table[0] < 90) OR (table[1] >= 100 AND table[1] < 120)
  // row groups min, max:
  // table[0] 0-80, 81-160, 161-200.
  // table[1] 200-121, 120-41, 40-0.
  auto filter_col1  = cudf::ast::column_reference(0);
  auto filter_col2  = cudf::ast::column_name_reference("col1");
  T constexpr low1  = 70;
  T constexpr high1 = 90;
  T constexpr low2  = 100;
  T constexpr high2 = 120;
  auto lov          = cudf::numeric_scalar(low1, true);
  auto hiv          = cudf::numeric_scalar(high1, true);
  auto lo_lit1      = cudf::ast::literal(lov);
  auto hi_lit1      = cudf::ast::literal(hiv);
  auto expr_1  = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col1, lo_lit1);
  auto expr_2  = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, hi_lit1);
  auto expr_3  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);
  auto lov2    = cudf::numeric_scalar(low2, true);
  auto hiv2    = cudf::numeric_scalar(high2, true);
  auto lo_lit2 = cudf::ast::literal(lov2);
  auto hi_lit2 = cudf::ast::literal(hiv2);
  auto expr_4  = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col2, lo_lit2);
  auto expr_5  = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2, hi_lit2);
  auto expr_6  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4, expr_5);
  // expression to test
  auto expr_7 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6);

  // Expected result
  auto filter_col2_ref = cudf::ast::column_reference(1);
  auto expr_4_ref =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col2_ref, lo_lit2);
  auto expr_5_ref = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2_ref, hi_lit2);
  auto expr_6_ref =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4_ref, expr_5_ref);
  auto expr_7_ref = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6_ref);
  auto predicate  = cudf::compute_column(written_table, expr_7_ref);
  auto expected   = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_7);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterSupported)
{
  using T                    = uint32_t;
  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterSupported.parquet");
  auto const written_table   = src.view();

  // Filtering AST - ((table[0] > 70 AND table[0] <= 90) OR (table[1] >= 100 AND table[1] < 120))
  //              AND (table[1] != 110)
  // row groups min, max:
  // table[0] 0-80, 81-160, 161-200.
  // table[1] 200-121, 120-41, 40-0.
  auto filter_col1       = cudf::ast::column_reference(0);
  auto filter_col2       = cudf::ast::column_reference(1);
  T constexpr low1       = 70;
  T constexpr high1      = 90;
  T constexpr low2       = 100;
  T constexpr high2      = 120;
  T constexpr skip_value = 110;
  auto lov               = cudf::numeric_scalar(low1, true);
  auto hiv               = cudf::numeric_scalar(high1, true);
  auto lo_lit1           = cudf::ast::literal(lov);
  auto hi_lit1           = cudf::ast::literal(hiv);
  auto expr_1  = cudf::ast::operation(cudf::ast::ast_operator::GREATER, filter_col1, lo_lit1);
  auto expr_2  = cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, filter_col1, hi_lit1);
  auto expr_3  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);
  auto lov2    = cudf::numeric_scalar(low2, true);
  auto hiv2    = cudf::numeric_scalar(high2, true);
  auto lo_lit2 = cudf::ast::literal(lov2);
  auto hi_lit2 = cudf::ast::literal(hiv2);
  auto expr_4  = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col2, lo_lit2);
  auto expr_5  = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2, hi_lit2);
  auto expr_6  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4, expr_5);
  auto expr_7  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6);
  auto skip_ov = cudf::numeric_scalar(skip_value, true);
  auto skip_lit = cudf::ast::literal(skip_ov);
  auto expr_8   = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col2, skip_lit);
  auto expr_9   = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_7, expr_8);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr_9);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_9);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterSupported2)
{
  using T                 = uint32_t;
  constexpr auto num_rows = 4000;
  auto elements0 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 2000; });
  auto elements1 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 1000; });
  auto elements2 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 500; });
  auto col0 = cudf::test::fixed_width_column_wrapper<T>(elements0, elements0 + num_rows);
  auto col1 = cudf::test::fixed_width_column_wrapper<T>(elements1, elements1 + num_rows);
  auto col2 = cudf::test::fixed_width_column_wrapper<T>(elements2, elements2 + num_rows);
  auto const written_table = table_view{{col0, col1, col2}};
  auto const filepath      = temp_env->get_temp_filepath("FilterSupported2.parquet");
  {
    const cudf::io::parquet_writer_options out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .row_group_size_rows(1000);
    cudf::io::write_parquet(out_opts);
  }
  auto si          = cudf::io::source_info(filepath);
  auto filter_col0 = cudf::ast::column_reference(0);
  auto filter_col1 = cudf::ast::column_reference(1);
  auto filter_col2 = cudf::ast::column_reference(2);
  auto s_value     = cudf::numeric_scalar<T>(1, true);
  auto lit_value   = cudf::ast::literal(s_value);

  auto test_expr = [&](auto& expr) {
    // Expected result
    auto predicate = cudf::compute_column(written_table, expr);
    auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

    // tests
    auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr);
    auto table_with_metadata = cudf::io::read_parquet(builder);
    auto result              = table_with_metadata.tbl->view();

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
  };

  // row groups min, max:
  // table[0] 0-0, 0-0, 1-1, 1-1
  // table[1] 0-0, 1-1, 2-2, 3-3
  // table[2] 0-1, 2-3, 4-5, 6-7

  // Filtering AST -   table[i] == 1
  {
    auto expr0 = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, filter_col0, lit_value);
    test_expr(expr0);

    auto expr1 = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, filter_col1, lit_value);
    test_expr(expr1);

    auto expr2 = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, filter_col2, lit_value);
    test_expr(expr2);
  }
  // Filtering AST -   table[i] != 1
  {
    auto expr0 = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col0, lit_value);
    test_expr(expr0);

    auto expr1 = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col1, lit_value);
    test_expr(expr1);

    auto expr2 = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col2, lit_value);
    test_expr(expr2);
  }
}

// Error types - type mismatch, invalid column name, invalid literal type, invalid operator,
// non-bool filter output type.
TEST_F(ParquetReaderTest, FilterErrors)
{
  using T                    = uint32_t;
  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterErrors.parquet");
  auto const written_table   = src.view();
  auto si                    = cudf::io::source_info(filepath);

  // Filtering AST - invalid column index
  {
    auto filter_col1 = cudf::ast::column_reference(3);
    T constexpr low  = 100;
    auto lov         = cudf::numeric_scalar(low, true);
    auto low_lot     = cudf::ast::literal(lov);
    auto expr        = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);

    auto builder = cudf::io::parquet_reader_options::builder(si).filter(expr);
    EXPECT_THROW(cudf::io::read_parquet(builder), cudf::logic_error);
  }

  // Filtering AST - invalid column name
  {
    auto filter_col1 = cudf::ast::column_name_reference("col3");
    T constexpr low  = 100;
    auto lov         = cudf::numeric_scalar(low, true);
    auto low_lot     = cudf::ast::literal(lov);
    auto expr        = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);
    auto builder     = cudf::io::parquet_reader_options::builder(si).filter(expr);
    EXPECT_THROW(cudf::io::read_parquet(builder), cudf::logic_error);
  }

  // Filtering AST - incompatible literal type
  {
    auto filter_col1      = cudf::ast::column_name_reference("col0");
    auto filter_col2      = cudf::ast::column_reference(1);
    int64_t constexpr low = 100;
    auto lov              = cudf::numeric_scalar(low, true);
    auto low_lot          = cudf::ast::literal(lov);
    auto expr1    = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);
    auto expr2    = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2, low_lot);
    auto builder1 = cudf::io::parquet_reader_options::builder(si).filter(expr1);
    EXPECT_THROW(cudf::io::read_parquet(builder1), cudf::logic_error);

    auto builder2 = cudf::io::parquet_reader_options::builder(si).filter(expr2);
    EXPECT_THROW(cudf::io::read_parquet(builder2), cudf::logic_error);
  }

  // Filtering AST - "table[0] + 110" is invalid filter expression
  {
    auto filter_col1      = cudf::ast::column_reference(0);
    T constexpr add_value = 110;
    auto add_v            = cudf::numeric_scalar(add_value, true);
    auto add_lit          = cudf::ast::literal(add_v);
    auto expr_8 = cudf::ast::operation(cudf::ast::ast_operator::ADD, filter_col1, add_lit);

    auto si      = cudf::io::source_info(filepath);
    auto builder = cudf::io::parquet_reader_options::builder(si).filter(expr_8);
    EXPECT_THROW(cudf::io::read_parquet(builder), cudf::logic_error);

    // Expected result throw to show that the filter expression is invalid,
    // not a limitation of the parquet predicate pushdown.
    auto predicate = cudf::compute_column(written_table, expr_8);
    EXPECT_THROW(cudf::apply_boolean_mask(written_table, *predicate), cudf::logic_error);
  }

  // Filtering AST - INT64(table[0] < 100) non-bool expression
  {
    auto filter_col1 = cudf::ast::column_reference(0);
    T constexpr low  = 100;
    auto lov         = cudf::numeric_scalar(low, true);
    auto low_lot     = cudf::ast::literal(lov);
    auto bool_expr   = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);
    auto cast        = cudf::ast::operation(cudf::ast::ast_operator::CAST_TO_INT64, bool_expr);

    auto builder = cudf::io::parquet_reader_options::builder(si).filter(cast);
    EXPECT_THROW(cudf::io::read_parquet(builder), cudf::logic_error);
    EXPECT_NO_THROW(cudf::compute_column(written_table, cast));
    auto predicate = cudf::compute_column(written_table, cast);
    EXPECT_NE(predicate->view().type().id(), cudf::type_id::BOOL8);
  }
}

// Filter without stats information in file.
TEST_F(ParquetReaderTest, FilterNoStats)
{
  using T                 = uint32_t;
  constexpr auto num_rows = 16000;
  auto elements =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 1000; });
  auto col0 = cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_rows);
  auto const written_table = table_view{{col0}};
  auto const filepath      = temp_env->get_temp_filepath("FilterNoStats.parquet");
  {
    const cudf::io::parquet_writer_options out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .row_group_size_rows(8000)
        .stats_level(cudf::io::statistics_freq::STATISTICS_NONE);
    cudf::io::write_parquet(out_opts);
  }
  auto si          = cudf::io::source_info(filepath);
  auto filter_col0 = cudf::ast::column_reference(0);
  auto s_value     = cudf::numeric_scalar<T>(1, true);
  auto lit_value   = cudf::ast::literal(s_value);

  // row groups min, max:
  // table[0] 0-0, 1-1, 2-2, 3-3
  // Filtering AST - table[0] > 1
  auto expr = cudf::ast::operation(cudf::ast::ast_operator::GREATER, filter_col0, lit_value);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  // tests
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

// Filter for float column with NaN values
TEST_F(ParquetReaderTest, FilterFloatNAN)
{
  constexpr auto num_rows = 24000;
  auto elements           = cudf::detail::make_counting_transform_iterator(
    0, [num_rows](auto i) { return i > num_rows / 2 ? NAN : i; });
  auto col0 = cudf::test::fixed_width_column_wrapper<float>(elements, elements + num_rows);
  auto col1 = cudf::test::fixed_width_column_wrapper<double>(elements, elements + num_rows);

  auto const written_table = table_view{{col0, col1}};
  auto const filepath      = temp_env->get_temp_filepath("FilterFloatNAN.parquet");
  {
    const cudf::io::parquet_writer_options out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .row_group_size_rows(8000);
    cudf::io::write_parquet(out_opts);
  }
  auto si          = cudf::io::source_info(filepath);
  auto filter_col0 = cudf::ast::column_reference(0);
  auto filter_col1 = cudf::ast::column_reference(1);
  auto s0_value    = cudf::numeric_scalar<float>(NAN, true);
  auto lit0_value  = cudf::ast::literal(s0_value);
  auto s1_value    = cudf::numeric_scalar<double>(NAN, true);
  auto lit1_value  = cudf::ast::literal(s1_value);

  // row groups min, max:
  // table[0] 0-0, 1-1, 2-2, 3-3
  // Filtering AST - table[0] == NAN, table[1] != NAN
  auto expr_eq  = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, filter_col0, lit0_value);
  auto expr_neq = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col1, lit1_value);

  // Expected result
  auto predicate0 = cudf::compute_column(written_table, expr_eq);
  auto expected0  = cudf::apply_boolean_mask(written_table, *predicate0);
  auto predicate1 = cudf::compute_column(written_table, expr_neq);
  auto expected1  = cudf::apply_boolean_mask(written_table, *predicate1);

  // tests
  auto builder0             = cudf::io::parquet_reader_options::builder(si).filter(expr_eq);
  auto table_with_metadata0 = cudf::io::read_parquet(builder0);
  auto result0              = table_with_metadata0.tbl->view();
  auto builder1             = cudf::io::parquet_reader_options::builder(si).filter(expr_neq);
  auto table_with_metadata1 = cudf::io::read_parquet(builder1);
  auto result1              = table_with_metadata1.tbl->view();

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected0->view(), result0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected1->view(), result1);
}

TEST_F(ParquetWriterTest, TimestampMicrosINT96NoOverflow)
{
  using namespace cuda::std::chrono;
  using namespace cudf::io;

  column_wrapper<cudf::timestamp_us> big_ts_col{
    sys_days{year{3023} / month{7} / day{14}} + 7h + 38min + 45s + 418688us,
    sys_days{year{723} / month{3} / day{21}} + 14h + 20min + 13s + microseconds{781ms}};

  table_view expected({big_ts_col});
  auto filepath = temp_env->get_temp_filepath("BigINT96Timestamp.parquet");

  auto const out_opts =
    parquet_writer_options::builder(sink_info{filepath}, expected).int96_timestamps(true).build();
  write_parquet(out_opts);

  auto const in_opts = parquet_reader_options::builder(source_info(filepath))
                         .timestamp_type(cudf::data_type(cudf::type_id::TIMESTAMP_MICROSECONDS))
                         .build();
  auto const result = read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetWriterTest, PreserveNullability)
{
  constexpr auto num_rows = 100;

  auto const col0_data = random_values<int32_t>(num_rows);
  auto const col1_data = random_values<int32_t>(num_rows);

  auto const col0_validity = cudf::test::iterators::no_nulls();
  auto const col1_validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });

  column_wrapper<int32_t> col0{col0_data.begin(), col0_data.end(), col0_validity};
  column_wrapper<int32_t> col1{col1_data.begin(), col1_data.end(), col1_validity};
  auto const col2 = make_parquet_list_list_col<int>(0, num_rows, 5, 8, true);

  auto const expected = table_view{{col0, col1, *col2}};

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("mandatory");
  expected_metadata.column_metadata[0].set_nullability(false);
  expected_metadata.column_metadata[1].set_name("optional");
  expected_metadata.column_metadata[1].set_nullability(true);
  expected_metadata.column_metadata[2].set_name("lists");
  expected_metadata.column_metadata[2].set_nullability(true);
  // offsets is a cudf thing that's not part of the parquet schema so it won't have nullability set
  expected_metadata.column_metadata[2].child(0).set_name("offsets");
  expected_metadata.column_metadata[2].child(1).set_name("element");
  expected_metadata.column_metadata[2].child(1).set_nullability(false);
  expected_metadata.column_metadata[2].child(1).child(0).set_name("offsets");
  expected_metadata.column_metadata[2].child(1).child(1).set_name("element");
  expected_metadata.column_metadata[2].child(1).child(1).set_nullability(true);

  auto const filepath = temp_env->get_temp_filepath("PreserveNullability.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(expected_metadata);

  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options const in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result        = cudf::io::read_parquet(in_opts);
  auto const read_metadata = cudf::io::table_input_metadata{result.metadata};

  // test that expected_metadata matches read_metadata
  std::function<void(cudf::io::column_in_metadata, cudf::io::column_in_metadata)>
    compare_names_and_nullability = [&](auto lhs, auto rhs) {
      EXPECT_EQ(lhs.get_name(), rhs.get_name());
      ASSERT_EQ(lhs.is_nullability_defined(), rhs.is_nullability_defined());
      if (lhs.is_nullability_defined()) { EXPECT_EQ(lhs.nullable(), rhs.nullable()); }
      ASSERT_EQ(lhs.num_children(), rhs.num_children());
      for (int i = 0; i < lhs.num_children(); ++i) {
        compare_names_and_nullability(lhs.child(i), rhs.child(i));
      }
    };

  ASSERT_EQ(expected_metadata.column_metadata.size(), read_metadata.column_metadata.size());

  for (size_t i = 0; i < expected_metadata.column_metadata.size(); ++i) {
    compare_names_and_nullability(expected_metadata.column_metadata[i],
                                  read_metadata.column_metadata[i]);
  }
}

TEST_P(ParquetV2Test, CheckEncodings)
{
  using cudf::io::parquet::Encoding;
  constexpr auto num_rows = 100'000;
  auto const is_v2        = GetParam();

  auto const validity = cudf::test::iterators::no_nulls();
  // data should be PLAIN for v1, RLE for V2
  auto col0_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> bool { return i % 2 == 0; });
  // data should be PLAIN for both
  auto col1_data = random_values<int32_t>(num_rows);
  // data should be PLAIN_DICTIONARY for v1, PLAIN and RLE_DICTIONARY for v2
  auto col2_data = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return 1; });

  cudf::test::fixed_width_column_wrapper<bool> col0{col0_data, col0_data + num_rows, validity};
  column_wrapper<int32_t> col1{col1_data.begin(), col1_data.end(), validity};
  column_wrapper<int32_t> col2{col2_data, col2_data + num_rows, validity};

  auto expected = table_view{{col0, col1, col2}};

  auto const filename = is_v2 ? "CheckEncodingsV2.parquet" : "CheckEncodingsV1.parquet";
  auto filepath       = temp_env->get_temp_filepath(filename);
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .max_page_size_rows(num_rows)
      .write_v2_headers(is_v2);
  cudf::io::write_parquet(out_opts);

  // make sure the expected encodings are present
  auto contains = [](auto const& vec, auto const& enc) {
    return std::find(vec.begin(), vec.end(), enc) != vec.end();
  };

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  auto const& chunk0_enc = fmd.row_groups[0].columns[0].meta_data.encodings;
  auto const& chunk1_enc = fmd.row_groups[0].columns[1].meta_data.encodings;
  auto const& chunk2_enc = fmd.row_groups[0].columns[2].meta_data.encodings;
  if (is_v2) {
    // col0 should have RLE for rep/def and data
    EXPECT_TRUE(chunk0_enc.size() == 1);
    EXPECT_TRUE(contains(chunk0_enc, Encoding::RLE));
    // col1 should have RLE for rep/def and PLAIN for data
    EXPECT_TRUE(chunk1_enc.size() == 2);
    EXPECT_TRUE(contains(chunk1_enc, Encoding::RLE));
    EXPECT_TRUE(contains(chunk1_enc, Encoding::PLAIN));
    // col2 should have RLE for rep/def, PLAIN for dict, and RLE_DICTIONARY for data
    EXPECT_TRUE(chunk2_enc.size() == 3);
    EXPECT_TRUE(contains(chunk2_enc, Encoding::RLE));
    EXPECT_TRUE(contains(chunk2_enc, Encoding::PLAIN));
    EXPECT_TRUE(contains(chunk2_enc, Encoding::RLE_DICTIONARY));
  } else {
    // col0 should have RLE for rep/def and PLAIN for data
    EXPECT_TRUE(chunk0_enc.size() == 2);
    EXPECT_TRUE(contains(chunk0_enc, Encoding::RLE));
    EXPECT_TRUE(contains(chunk0_enc, Encoding::PLAIN));
    // col1 should have RLE for rep/def and PLAIN for data
    EXPECT_TRUE(chunk1_enc.size() == 2);
    EXPECT_TRUE(contains(chunk1_enc, Encoding::RLE));
    EXPECT_TRUE(contains(chunk1_enc, Encoding::PLAIN));
    // col2 should have RLE for rep/def and PLAIN_DICTIONARY for data and dict
    EXPECT_TRUE(chunk2_enc.size() == 2);
    EXPECT_TRUE(contains(chunk2_enc, Encoding::RLE));
    EXPECT_TRUE(contains(chunk2_enc, Encoding::PLAIN_DICTIONARY));
  }
}

CUDF_TEST_PROGRAM_MAIN()
