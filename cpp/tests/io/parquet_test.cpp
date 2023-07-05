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
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/span.hpp>

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
void read_footer(const std::unique_ptr<cudf::io::datasource>& source,
                 cudf::io::parquet::FileMetaData* file_meta_data)
{
  constexpr auto header_len = sizeof(cudf::io::parquet::file_header_s);
  constexpr auto ender_len  = sizeof(cudf::io::parquet::file_ender_s);

  const auto len           = source->size();
  const auto header_buffer = source->host_read(0, header_len);
  const auto header =
    reinterpret_cast<const cudf::io::parquet::file_header_s*>(header_buffer->data());
  const auto ender_buffer = source->host_read(len - ender_len, ender_len);
  const auto ender = reinterpret_cast<const cudf::io::parquet::file_ender_s*>(ender_buffer->data());

  // checks for valid header, footer, and file length
  CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
  CUDF_EXPECTS(header->magic == cudf::io::parquet::parquet_magic &&
                 ender->magic == cudf::io::parquet::parquet_magic,
               "Corrupted header or footer");
  CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
               "Incorrect footer length");

  // parquet files end with 4-byte footer_length and 4-byte magic == "PAR1"
  // seek backwards from the end of the file (footer_length + 8 bytes of ender)
  const auto footer_buffer =
    source->host_read(len - ender->footer_len - ender_len, ender->footer_len);
  cudf::io::parquet::CompactProtocolReader cp(footer_buffer->data(), ender->footer_len);

  // returns true on success
  bool res = cp.read(file_meta_data);
  CUDF_EXPECTS(res, "Cannot parse file metadata");
}

// returns the number of bits used for dictionary encoding data at the given page location.
// this assumes the data is uncompressed.
// throws cudf::logic_error if the page_loc data is invalid.
int read_dict_bits(const std::unique_ptr<cudf::io::datasource>& source,
                   const cudf::io::parquet::PageLocation& page_loc)
{
  CUDF_EXPECTS(page_loc.offset > 0, "Cannot find page header");
  CUDF_EXPECTS(page_loc.compressed_page_size > 0, "Invalid page header length");

  cudf::io::parquet::PageHeader page_hdr;
  const auto page_buf = source->host_read(page_loc.offset, page_loc.compressed_page_size);
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
  const std::unique_ptr<cudf::io::datasource>& source, const cudf::io::parquet::ColumnChunk& chunk)
{
  CUDF_EXPECTS(chunk.column_index_offset > 0, "Cannot find column index");
  CUDF_EXPECTS(chunk.column_index_length > 0, "Invalid column index length");

  cudf::io::parquet::ColumnIndex colidx;
  const auto ci_buf = source->host_read(chunk.column_index_offset, chunk.column_index_length);
  cudf::io::parquet::CompactProtocolReader cp(ci_buf->data(), ci_buf->size());
  bool res = cp.read(&colidx);
  CUDF_EXPECTS(res, "Cannot parse column index");
  return colidx;
}

// read offset index from datasource at location indicated by chunk,
// parse and return as an OffsetIndex struct.
// throws cudf::logic_error if the chunk data is invalid.
cudf::io::parquet::OffsetIndex read_offset_index(
  const std::unique_ptr<cudf::io::datasource>& source, const cudf::io::parquet::ColumnChunk& chunk)
{
  CUDF_EXPECTS(chunk.offset_index_offset > 0, "Cannot find offset index");
  CUDF_EXPECTS(chunk.offset_index_length > 0, "Invalid offset index length");

  cudf::io::parquet::OffsetIndex offidx;
  const auto oi_buf = source->host_read(chunk.offset_index_offset, chunk.offset_index_length);
  cudf::io::parquet::CompactProtocolReader cp(oi_buf->data(), oi_buf->size());
  bool res = cp.read(&offidx);
  CUDF_EXPECTS(res, "Cannot parse offset index");
  return offidx;
}

// read page header from datasource at location indicated by page_loc,
// parse and return as a PageHeader struct.
// throws cudf::logic_error if the page_loc data is invalid.
cudf::io::parquet::PageHeader read_page_header(const std::unique_ptr<cudf::io::datasource>& source,
                                               const cudf::io::parquet::PageLocation& page_loc)
{
  CUDF_EXPECTS(page_loc.offset > 0, "Cannot find page header");
  CUDF_EXPECTS(page_loc.compressed_page_size > 0, "Invalid page header length");

  cudf::io::parquet::PageHeader page_hdr;
  const auto page_buf = source->host_read(page_loc.offset, page_loc.compressed_page_size);
  cudf::io::parquet::CompactProtocolReader cp(page_buf->data(), page_buf->size());
  bool res = cp.read(&page_hdr);
  CUDF_EXPECTS(res, "Cannot parse page header");
  return page_hdr;
}

// Base test fixture for tests
struct ParquetWriterTest : public cudf::test::BaseFixture {};

// Base test fixture for tests
struct ParquetReaderTest : public cudf::test::BaseFixture {};

TEST_F(ParquetReaderTest, FixedLenBinary)
{
  std::vector<char const*> strings{"AB", "12"};
  column_wrapper<cudf::string_view> col{strings.begin(), strings.end()};

  auto write_tbl = table_view{{col}};

  cudf::io::table_input_metadata expected_metadata(write_tbl);
  expected_metadata.column_metadata[0].set_name("col").set_output_as_binary(true);

  auto filepath = temp_env->get_temp_filepath("toto.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, write_tbl)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER)
      .metadata(&expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema({cudf::io::reader_column_schema().set_convert_binary_to_strings(false)});
  auto res = cudf::io::read_parquet(in_opts);
  cudf::test::print(res.tbl->view().column(0));

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{"./FIXED_BIN_300_TEST.parquet"})
      .set_column_schema({cudf::io::reader_column_schema().set_convert_binary_to_strings(false),
                          cudf::io::reader_column_schema().set_convert_binary_to_strings(false)});
  auto result = cudf::io::read_parquet(read_opts);

  // we should only get back 4 rows
  EXPECT_EQ(result.tbl->view().column(0).type().id(), cudf::type_id::LIST);
  EXPECT_EQ(result.tbl->view().column(0).size(), 300);

  cudf::test::print(result.tbl->view().column(0));
}

CUDF_TEST_PROGRAM_MAIN()
