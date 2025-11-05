/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>

#include <string>
#include <vector>

namespace {

cudf::host_span<uint8_t const> fetch_footer_bytes(cudf::host_span<uint8_t const> buffer)
{
  using namespace cudf::io::parquet;

  constexpr auto header_len = sizeof(file_header_s);
  constexpr auto ender_len  = sizeof(file_ender_s);
  size_t const len          = buffer.size();

  auto const header_buffer = cudf::host_span<uint8_t const>(buffer.data(), header_len);
  auto const header        = reinterpret_cast<file_header_s const*>(header_buffer.data());
  auto const ender_buffer =
    cudf::host_span<uint8_t const>(buffer.data() + len - ender_len, ender_len);
  auto const ender = reinterpret_cast<file_ender_s const*>(ender_buffer.data());
  CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
  constexpr uint32_t parquet_magic = (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24));
  CUDF_EXPECTS(header->magic == parquet_magic && ender->magic == parquet_magic,
               "Corrupted header or footer");
  CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
               "Incorrect footer length");

  return cudf::host_span<uint8_t const>(buffer.data() + len - ender->footer_len - ender_len,
                                        ender->footer_len);
}

cudf::host_span<uint8_t const> fetch_page_index_bytes(
  cudf::host_span<uint8_t const> buffer, cudf::io::text::byte_range_info const page_index_bytes)
{
  return cudf::host_span<uint8_t const>(
    reinterpret_cast<uint8_t const*>(buffer.data()) + page_index_bytes.offset(),
    page_index_bytes.size());
}

std::vector<rmm::device_buffer> fetch_byte_ranges(
  cudf::host_span<uint8_t const> host_buffer,
  cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<rmm::device_buffer> buffers{};
  buffers.reserve(byte_ranges.size());

  std::transform(
    byte_ranges.begin(),
    byte_ranges.end(),
    std::back_inserter(buffers),
    [&](auto const& byte_range) {
      auto const chunk_offset = host_buffer.data() + byte_range.offset();
      auto const chunk_size   = byte_range.size();
      auto buffer             = rmm::device_buffer(chunk_size, stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        buffer.data(), chunk_offset, chunk_size, cudaMemcpyHostToDevice, stream.value()));
      return buffer;
    });

  stream.synchronize_no_throw();
  return buffers;
}

template <typename... UniqPtrs>
std::vector<std::unique_ptr<cudf::column>> make_uniqueptrs_vector(UniqPtrs&&... uniqptrs)
{
  std::vector<std::unique_ptr<cudf::column>> ptrsvec;
  (ptrsvec.push_back(std::forward<UniqPtrs>(uniqptrs)), ...);
  return ptrsvec;
}

cudf::table construct_table()
{
  constexpr auto num_rows = 10;

  std::vector<size_t> zeros(num_rows, 0);
  std::vector<size_t> ones(num_rows, 1);

  cudf::test::fixed_width_column_wrapper<bool> col0(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int8_t> col1(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int16_t> col2(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col3(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<float> col4(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<double> col5(zeros.begin(), zeros.end());
  cudf::test::fixed_point_column_wrapper<numeric::decimal128::rep> col6(
    ones.begin(), ones.end(), numeric::scale_type{12});
  cudf::test::fixed_point_column_wrapper<numeric::decimal128::rep> col7(
    ones.begin(), ones.end(), numeric::scale_type{-12});

  cudf::test::lists_column_wrapper<int64_t> col8{
    {1, 1}, {1, 1, 1}, {}, {1}, {1, 1, 1, 1}, {1, 1, 1, 1, 1}, {}, {1, -1}, {}, {-1, -1}};

  cudf::test::structs_column_wrapper col9 = [&ones] {
    cudf::test::fixed_width_column_wrapper<int32_t> child_col(ones.begin(), ones.end());
    return cudf::test::structs_column_wrapper{child_col};
  }();

  cudf::test::strings_column_wrapper col10 = [] {
    std::vector<std::string> col10_data(num_rows, "rapids");
    return cudf::test::strings_column_wrapper(col10_data.begin(), col10_data.end());
  }();

  auto colsptr = make_uniqueptrs_vector(col0.release(),
                                        col1.release(),
                                        col2.release(),
                                        col3.release(),
                                        col4.release(),
                                        col5.release(),
                                        col6.release(),
                                        col7.release(),
                                        col8.release(),
                                        col9.release(),
                                        col10.release());
  return cudf::table(std::move(colsptr));
}
}  // namespace

class HybridScanTest : public cudf::test::BaseFixture {};

TEST_F(HybridScanTest, DictionaryPageFiltering)
{
  auto tab    = construct_table();
  auto buffer = std::vector<char>();
  cudf::io::table_input_metadata out_metadata(tab);
  out_metadata.column_metadata[0].set_name("col0");
  out_metadata.column_metadata[3].set_name("col3");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, tab)
      .metadata(out_metadata)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS);
  cudf::io::write_parquet(out_opts, cudf::test::get_default_stream());

  auto col3_ref      = cudf::ast::column_name_reference("col3");
  auto literal_value = cudf::numeric_scalar<int32_t>(0, true, cudf::test::get_default_stream());
  auto literal       = cudf::ast::literal(literal_value);
  auto expr1         = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col3_ref, literal);

  auto col0_ref = cudf::ast::column_name_reference("col0");
  auto expr2    = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, col0_ref);
  auto expr3    = cudf::ast::operation(cudf::ast::ast_operator::NOT, expr2);

  auto filter_expr = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr1, expr3);
  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{}).filter(filter_expr);

  auto const file_buffer_span =
    cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(buffer.data()), buffer.size());
  auto const footer_buffer = fetch_footer_bytes(file_buffer_span);

  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(footer_buffer, in_opts);

  auto const page_index_byte_range = reader->page_index_byte_range();
  auto const page_index_buffer = fetch_page_index_bytes(file_buffer_span, page_index_byte_range);
  reader->setup_page_index(page_index_buffer);

  auto input_row_group_indices = reader->all_row_groups(in_opts);

  auto const dict_byte_ranges =
    std::get<1>(reader->secondary_filters_byte_ranges(input_row_group_indices, in_opts));
  std::vector<rmm::device_buffer> dictionary_page_buffers =
    fetch_byte_ranges(file_buffer_span,
                      dict_byte_ranges,
                      cudf::test::get_default_stream(),
                      cudf::get_current_device_resource_ref());

  auto result = reader->filter_row_groups_with_dictionary_pages(
    dictionary_page_buffers, input_row_group_indices, in_opts, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
