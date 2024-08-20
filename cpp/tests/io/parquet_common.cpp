/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/io/parquet.hpp>

// Global environment for temporary files
cudf::test::TempDirTestEnvironment* const temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
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

template <typename T>
std::vector<T> random_values(size_t size)
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

template std::vector<float> random_values<float>(size_t size);
template std::vector<double> random_values<double>(size_t size);
template std::vector<int8_t> random_values<int8_t>(size_t size);
template std::vector<int16_t> random_values<int16_t>(size_t size);
template std::vector<int32_t> random_values<int32_t>(size_t size);
template std::vector<int64_t> random_values<int64_t>(size_t size);
template std::vector<__int128_t> random_values<__int128_t>(size_t size);
template std::vector<uint8_t> random_values<uint8_t>(size_t size);
template std::vector<uint16_t> random_values<uint16_t>(size_t size);
template std::vector<uint32_t> random_values<uint32_t>(size_t size);
template std::vector<uint64_t> random_values<uint64_t>(size_t size);

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

// =============================================================================
// ---- test data for stats sort order tests

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

#define FIXED_WIDTH_ORDERED_DATA(type)                                      \
  template cudf::test::fixed_width_column_wrapper<type> ascending<type>();  \
  template cudf::test::fixed_width_column_wrapper<type> descending<type>(); \
  template cudf::test::fixed_width_column_wrapper<type> unordered<type>()

FIXED_WIDTH_ORDERED_DATA(bool);
FIXED_WIDTH_ORDERED_DATA(int8_t);
FIXED_WIDTH_ORDERED_DATA(int16_t);
FIXED_WIDTH_ORDERED_DATA(int32_t);
FIXED_WIDTH_ORDERED_DATA(int64_t);
FIXED_WIDTH_ORDERED_DATA(uint8_t);
FIXED_WIDTH_ORDERED_DATA(uint16_t);
FIXED_WIDTH_ORDERED_DATA(uint32_t);
FIXED_WIDTH_ORDERED_DATA(uint64_t);
FIXED_WIDTH_ORDERED_DATA(float);
FIXED_WIDTH_ORDERED_DATA(double);
FIXED_WIDTH_ORDERED_DATA(cudf::duration_D);
FIXED_WIDTH_ORDERED_DATA(cudf::duration_s);
FIXED_WIDTH_ORDERED_DATA(cudf::duration_ms);
FIXED_WIDTH_ORDERED_DATA(cudf::duration_us);
FIXED_WIDTH_ORDERED_DATA(cudf::duration_ns);
FIXED_WIDTH_ORDERED_DATA(cudf::timestamp_D);
FIXED_WIDTH_ORDERED_DATA(cudf::timestamp_s);
FIXED_WIDTH_ORDERED_DATA(cudf::timestamp_ms);
FIXED_WIDTH_ORDERED_DATA(cudf::timestamp_us);
FIXED_WIDTH_ORDERED_DATA(cudf::timestamp_ns);
FIXED_WIDTH_ORDERED_DATA(numeric::decimal32);
FIXED_WIDTH_ORDERED_DATA(numeric::decimal64);
FIXED_WIDTH_ORDERED_DATA(numeric::decimal128);

template cudf::test::strings_column_wrapper ascending<cudf::string_view>();
template cudf::test::strings_column_wrapper descending<cudf::string_view>();
template cudf::test::strings_column_wrapper unordered<cudf::string_view>();

}  // namespace testdata

template <typename T>
std::unique_ptr<cudf::column> make_parquet_list_col(std::mt19937& engine,
                                                    int num_rows,
                                                    int max_vals_per_row,
                                                    bool include_validity)
{
  std::vector<cudf::size_type> row_sizes(num_rows);

  auto const min_values_per_row = include_validity ? 0 : 1;
  std::uniform_int_distribution<cudf::size_type> dist{min_values_per_row, max_vals_per_row};
  std::generate_n(row_sizes.begin(), num_rows, [&]() { return cudf::size_type{dist(engine)}; });

  std::vector<cudf::size_type> offsets(num_rows + 1);
  std::exclusive_scan(row_sizes.begin(), row_sizes.end(), offsets.begin(), 0);
  offsets[num_rows] = offsets[num_rows - 1] + row_sizes.back();

  std::vector<T> values = random_values<T>(offsets[num_rows]);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets_col(offsets.begin(),
                                                                      offsets.end());

  if (include_validity) {
    auto valids = random_validity(engine);
    auto values_col =
      cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end(), valids);
    auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids, valids + num_rows);

    auto col = cudf::make_lists_column(
      num_rows, offsets_col.release(), values_col.release(), null_count, std::move(null_mask));
    return cudf::purge_nonempty_nulls(*col);
  } else {
    auto values_col = cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end());
    return cudf::make_lists_column(num_rows,
                                   offsets_col.release(),
                                   values_col.release(),
                                   0,
                                   cudf::create_null_mask(num_rows, cudf::mask_state::ALL_VALID));
  }
}

template std::unique_ptr<cudf::column> make_parquet_list_col<int8_t>(std::mt19937& engine,
                                                                     int num_rows,
                                                                     int max_vals_per_row,
                                                                     bool include_validity);
template std::unique_ptr<cudf::column> make_parquet_list_col<int16_t>(std::mt19937& engine,
                                                                      int num_rows,
                                                                      int max_vals_per_row,
                                                                      bool include_validity);
template std::unique_ptr<cudf::column> make_parquet_list_col<int32_t>(std::mt19937& engine,
                                                                      int num_rows,
                                                                      int max_vals_per_row,
                                                                      bool include_validity);
template std::unique_ptr<cudf::column> make_parquet_list_col<int64_t>(std::mt19937& engine,
                                                                      int num_rows,
                                                                      int max_vals_per_row,
                                                                      bool include_validity);

std::vector<std::string> string_values(std::mt19937& engine, int num_rows, int max_string_len)
{
  static std::uniform_int_distribution<char> char_dist{'a', 'z'};
  static std::uniform_int_distribution<cudf::size_type> strlen_dist{1, max_string_len};

  std::vector<std::string> values(num_rows);
  std::generate_n(values.begin(), values.size(), [&]() {
    int str_len     = strlen_dist(engine);
    std::string res = "";
    for (int i = 0; i < str_len; i++) {
      res += char_dist(engine);
    }
    return res;
  });

  return values;
}

// make a random list<string> column, with random string lengths of 0..max_string_len,
// and up to max_vals_per_row strings in each list.
std::unique_ptr<cudf::column> make_parquet_string_list_col(std::mt19937& engine,
                                                           int num_rows,
                                                           int max_vals_per_row,
                                                           int max_string_len,
                                                           bool include_validity)
{
  auto const range_min = include_validity ? 0 : 1;

  std::uniform_int_distribution<cudf::size_type> dist{range_min, max_vals_per_row};

  std::vector<cudf::size_type> row_sizes(num_rows);
  std::generate_n(row_sizes.begin(), num_rows, [&]() { return cudf::size_type{dist(engine)}; });

  std::vector<cudf::size_type> offsets(num_rows + 1);
  std::exclusive_scan(row_sizes.begin(), row_sizes.end(), offsets.begin(), 0);
  offsets[num_rows] = offsets[num_rows - 1] + row_sizes.back();

  std::uniform_int_distribution<cudf::size_type> strlen_dist{range_min, max_string_len};
  auto const values = string_values(engine, offsets[num_rows], max_string_len);

  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets_col(offsets.begin(),
                                                                      offsets.end());

  if (include_validity) {
    auto valids     = random_validity(engine);
    auto values_col = cudf::test::strings_column_wrapper(values.begin(), values.end(), valids);
    auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids, valids + num_rows);

    auto col = cudf::make_lists_column(
      num_rows, offsets_col.release(), values_col.release(), null_count, std::move(null_mask));
    return cudf::purge_nonempty_nulls(*col);
  } else {
    auto values_col = cudf::test::strings_column_wrapper(values.begin(), values.end());
    return cudf::make_lists_column(num_rows,
                                   offsets_col.release(),
                                   values_col.release(),
                                   0,
                                   cudf::create_null_mask(num_rows, cudf::mask_state::ALL_VALID));
  }
}

template <typename T>
std::pair<cudf::table, std::string> create_parquet_typed_with_stats(std::string const& filename)
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

#define TYPED_WITH_STATS(type)                                                        \
  template std::pair<cudf::table, std::string> create_parquet_typed_with_stats<type>( \
    std::string const& filename)

TYPED_WITH_STATS(cudf::string_view);
TYPED_WITH_STATS(bool);
TYPED_WITH_STATS(int8_t);
TYPED_WITH_STATS(int16_t);
TYPED_WITH_STATS(int32_t);
TYPED_WITH_STATS(int64_t);
TYPED_WITH_STATS(uint8_t);
TYPED_WITH_STATS(uint16_t);
TYPED_WITH_STATS(uint32_t);
TYPED_WITH_STATS(uint64_t);
TYPED_WITH_STATS(float);
TYPED_WITH_STATS(double);
// TYPED_WITH_STATS(cudf::duration_D);
// TYPED_WITH_STATS(cudf::duration_s);
TYPED_WITH_STATS(cudf::duration_ms);
TYPED_WITH_STATS(cudf::duration_us);
TYPED_WITH_STATS(cudf::duration_ns);
TYPED_WITH_STATS(cudf::timestamp_D);
// TYPED_WITH_STATS(cudf::timestamp_s);
TYPED_WITH_STATS(cudf::timestamp_ms);
TYPED_WITH_STATS(cudf::timestamp_us);
TYPED_WITH_STATS(cudf::timestamp_ns);
TYPED_WITH_STATS(numeric::decimal32);
TYPED_WITH_STATS(numeric::decimal64);
TYPED_WITH_STATS(numeric::decimal128);

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
                       cudf::io::parquet::detail::Type ptype,
                       cuda::std::optional<cudf::io::parquet::detail::ConvertedType> const& ctype)
{
  auto ctype_val = ctype.value_or(cudf::io::parquet::detail::UNKNOWN);
  switch (ptype) {
    case cudf::io::parquet::detail::INT32:
      switch (ctype_val) {
        case cudf::io::parquet::detail::UINT_8:
        case cudf::io::parquet::detail::UINT_16:
        case cudf::io::parquet::detail::UINT_32:
          return compare(*(reinterpret_cast<uint32_t const*>(v1.data())),
                         *(reinterpret_cast<uint32_t const*>(v2.data())));
        default:
          return compare(*(reinterpret_cast<int32_t const*>(v1.data())),
                         *(reinterpret_cast<int32_t const*>(v2.data())));
      }

    case cudf::io::parquet::detail::INT64:
      if (ctype_val == cudf::io::parquet::detail::UINT_64) {
        return compare(*(reinterpret_cast<uint64_t const*>(v1.data())),
                       *(reinterpret_cast<uint64_t const*>(v2.data())));
      }
      return compare(*(reinterpret_cast<int64_t const*>(v1.data())),
                     *(reinterpret_cast<int64_t const*>(v2.data())));

    case cudf::io::parquet::detail::FLOAT:
      return compare(*(reinterpret_cast<float const*>(v1.data())),
                     *(reinterpret_cast<float const*>(v2.data())));

    case cudf::io::parquet::detail::DOUBLE:
      return compare(*(reinterpret_cast<double const*>(v1.data())),
                     *(reinterpret_cast<double const*>(v2.data())));

    case cudf::io::parquet::detail::BYTE_ARRAY: {
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

void expect_compression_stats_empty(std::shared_ptr<cudf::io::writer_compression_statistics> stats)
{
  EXPECT_EQ(stats->num_compressed_bytes(), 0);
  EXPECT_EQ(stats->num_failed_bytes(), 0);
  EXPECT_EQ(stats->num_skipped_bytes(), 0);
  EXPECT_TRUE(std::isnan(stats->compression_ratio()));
}
