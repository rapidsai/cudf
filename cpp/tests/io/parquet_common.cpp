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

#include "parquet_common.hpp"

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

template std::unique_ptr<cudf::table> create_random_fixed_table<int>(cudf::size_type num_columns,
                                                                     cudf::size_type num_rows,
                                                                     bool include_validity);
template std::unique_ptr<cudf::table> create_random_fixed_table<float>(cudf::size_type num_columns,
                                                                       cudf::size_type num_rows,
                                                                       bool include_validity);

template std::unique_ptr<cudf::table> create_compressible_fixed_table<int>(
  cudf::size_type num_columns,
  cudf::size_type num_rows,
  cudf::size_type period,
  bool include_validity);

template std::unique_ptr<cudf::table> create_compressible_fixed_table<float>(
  cudf::size_type num_columns,
  cudf::size_type num_rows,
  cudf::size_type period,
  bool include_validity);

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

template std::unique_ptr<cudf::column> make_parquet_list_list_col<int>(
  int skip_rows, int num_rows, int lists_per_row, int list_size, bool include_validity);

// given a datasource pointing to a parquet file, read the footer
// of the file to populate the FileMetaData pointed to by file_meta_data.
// throws cudf::logic_error if the file or metadata is invalid.
void read_footer(std::unique_ptr<cudf::io::datasource> const& source,
                 cudf::io::parquet::detail::FileMetaData* file_meta_data)
{
  constexpr auto header_len = sizeof(cudf::io::parquet::detail::file_header_s);
  constexpr auto ender_len  = sizeof(cudf::io::parquet::detail::file_ender_s);

  auto const len           = source->size();
  auto const header_buffer = source->host_read(0, header_len);
  auto const header =
    reinterpret_cast<cudf::io::parquet::detail::file_header_s const*>(header_buffer->data());
  auto const ender_buffer = source->host_read(len - ender_len, ender_len);
  auto const ender =
    reinterpret_cast<cudf::io::parquet::detail::file_ender_s const*>(ender_buffer->data());

  // checks for valid header, footer, and file length
  ASSERT_GT(len, header_len + ender_len);
  ASSERT_TRUE(header->magic == cudf::io::parquet::detail::parquet_magic &&
              ender->magic == cudf::io::parquet::detail::parquet_magic);
  ASSERT_TRUE(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len));

  // parquet files end with 4-byte footer_length and 4-byte magic == "PAR1"
  // seek backwards from the end of the file (footer_length + 8 bytes of ender)
  auto const footer_buffer =
    source->host_read(len - ender->footer_len - ender_len, ender->footer_len);
  cudf::io::parquet::detail::CompactProtocolReader cp(footer_buffer->data(), ender->footer_len);

  cp.read(file_meta_data);
}

// returns the number of bits used for dictionary encoding data at the given page location.
// this assumes the data is uncompressed.
// throws cudf::logic_error if the page_loc data is invalid.
int read_dict_bits(std::unique_ptr<cudf::io::datasource> const& source,
                   cudf::io::parquet::detail::PageLocation const& page_loc)
{
  CUDF_EXPECTS(page_loc.offset > 0, "Cannot find page header");
  CUDF_EXPECTS(page_loc.compressed_page_size > 0, "Invalid page header length");

  cudf::io::parquet::detail::PageHeader page_hdr;
  auto const page_buf = source->host_read(page_loc.offset, page_loc.compressed_page_size);
  cudf::io::parquet::detail::CompactProtocolReader cp(page_buf->data(), page_buf->size());
  cp.read(&page_hdr);

  // cp should be pointing at the start of page data now. the first byte
  // should be the encoding bit size
  return cp.getb();
}

// read column index from datasource at location indicated by chunk,
// parse and return as a ColumnIndex struct.
// throws cudf::logic_error if the chunk data is invalid.
cudf::io::parquet::detail::ColumnIndex read_column_index(
  std::unique_ptr<cudf::io::datasource> const& source,
  cudf::io::parquet::detail::ColumnChunk const& chunk)
{
  CUDF_EXPECTS(chunk.column_index_offset > 0, "Cannot find column index");
  CUDF_EXPECTS(chunk.column_index_length > 0, "Invalid column index length");

  cudf::io::parquet::detail::ColumnIndex colidx;
  auto const ci_buf = source->host_read(chunk.column_index_offset, chunk.column_index_length);
  cudf::io::parquet::detail::CompactProtocolReader cp(ci_buf->data(), ci_buf->size());
  cp.read(&colidx);
  return colidx;
}

// read offset index from datasource at location indicated by chunk,
// parse and return as an OffsetIndex struct.
// throws cudf::logic_error if the chunk data is invalid.
cudf::io::parquet::detail::OffsetIndex read_offset_index(
  std::unique_ptr<cudf::io::datasource> const& source,
  cudf::io::parquet::detail::ColumnChunk const& chunk)
{
  CUDF_EXPECTS(chunk.offset_index_offset > 0, "Cannot find offset index");
  CUDF_EXPECTS(chunk.offset_index_length > 0, "Invalid offset index length");

  cudf::io::parquet::detail::OffsetIndex offidx;
  auto const oi_buf = source->host_read(chunk.offset_index_offset, chunk.offset_index_length);
  cudf::io::parquet::detail::CompactProtocolReader cp(oi_buf->data(), oi_buf->size());
  cp.read(&offidx);
  return offidx;
}

// Return as a Statistics from the column chunk
cudf::io::parquet::detail::Statistics const& get_statistics(
  cudf::io::parquet::detail::ColumnChunk const& chunk)
{
  return chunk.meta_data.statistics;
}

// read page header from datasource at location indicated by page_loc,
// parse and return as a PageHeader struct.
// throws cudf::logic_error if the page_loc data is invalid.
cudf::io::parquet::detail::PageHeader read_page_header(
  std::unique_ptr<cudf::io::datasource> const& source,
  cudf::io::parquet::detail::PageLocation const& page_loc)
{
  CUDF_EXPECTS(page_loc.offset > 0, "Cannot find page header");
  CUDF_EXPECTS(page_loc.compressed_page_size > 0, "Invalid page header length");

  cudf::io::parquet::detail::PageHeader page_hdr;
  auto const page_buf = source->host_read(page_loc.offset, page_loc.compressed_page_size);
  cudf::io::parquet::detail::CompactProtocolReader cp(page_buf->data(), page_buf->size());
  cp.read(&page_hdr);
  return page_hdr;
}