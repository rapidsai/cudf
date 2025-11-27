/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

constexpr cudf::size_type num_cols = 8;

namespace {

/**
 * @brief Fetches a host span of Parquet footer bytes from the input buffer span
 *
 * @param buffer Input buffer span
 * @return A host span of the footer bytes
 */
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

/**
 * @brief Fetches a host span of Parquet page index bytes from the input buffer span
 *
 * @param buffer Input buffer span
 * @param page_index_bytes Byte range of page index to fetch
 * @return A host span of the page index bytes
 */
cudf::host_span<uint8_t const> fetch_page_index_bytes(
  cudf::host_span<uint8_t const> buffer, cudf::io::text::byte_range_info const page_index_bytes)
{
  return cudf::host_span<uint8_t const>(
    reinterpret_cast<uint8_t const*>(buffer.data()) + page_index_bytes.offset(),
    page_index_bytes.size());
}

/**
 * @brief Fetches a list of byte ranges from a host buffer into a vector of device buffers
 *
 * @param host_buffer Host buffer span
 * @param byte_ranges Byte ranges to fetch
 * @param stream CUDA stream
 *
 * @return Vector of device buffers
 */
std::vector<rmm::device_buffer> fetch_byte_ranges(
  cudf::host_span<uint8_t const> host_buffer,
  cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
  rmm::cuda_stream_view stream)
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
      auto buffer             = rmm::device_buffer(chunk_size, stream);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        buffer.data(), chunk_offset, chunk_size, cudaMemcpyHostToDevice, stream.value()));
      return buffer;
    });

  stream.synchronize_no_throw();
  return buffers;
}

}  // namespace

void BM_parquet_filter_string_row_groups_with_dicts_common(nvbench::state& state,
                                                           data_profile const& table_profile,
                                                           cudf::ast::operation const& filter_expr,
                                                           cudf::size_type num_row_groups,
                                                           double average_str_length,
                                                           cudf::size_type cardinality)
{
  auto constexpr rows_per_row_group = 2000;
  auto const num_rows               = num_row_groups * rows_per_row_group;

  std::vector<char> parquet_buffer;

  // Write table to parquet
  {
    auto const table = create_random_table(
      cycle_dtypes({cudf::type_id::STRING}, num_cols), row_count{num_rows}, table_profile);

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&parquet_buffer), table->view())
        .row_group_size_rows(rows_per_row_group)
        .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
        .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
        .compression(cudf::io::compression_type::AUTO);
    cudf::io::write_parquet(write_opts);
  }

  // Read table from parquet
  auto const file_buffer_span = cudf::host_span<uint8_t const>(
    reinterpret_cast<uint8_t const*>(parquet_buffer.data()), parquet_buffer.size());

  auto const stream = cudf::get_default_stream();

  // Fetch footer and page index bytes from the buffer.
  auto const footer_buffer = fetch_footer_bytes(file_buffer_span);

  auto const read_opts = cudf::io::parquet_reader_options::builder().filter(filter_expr).build();
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(footer_buffer, read_opts);

  // Get page index byte range from the reader
  auto const page_index_byte_range = reader->page_index_byte_range();

  // Fetch page index bytes from the input buffer
  auto const page_index_buffer = fetch_page_index_bytes(file_buffer_span, page_index_byte_range);

  // Setup page index
  reader->setup_page_index(page_index_buffer);

  // Get all row groups from the reader
  auto input_row_group_indices = reader->all_row_groups(read_opts);

  // Get dictionary page byte ranges from the reader
  auto const dict_page_byte_ranges =
    std::get<1>(reader->secondary_filters_byte_ranges(input_row_group_indices, read_opts));

  // If we have dictionary page byte ranges, filter row groups with dictionary pages
  CUDF_EXPECTS(dict_page_byte_ranges.size() > 0, "No dictionary page byte ranges found");

  // Fetch dictionary page buffers from the input file buffer
  std::vector<rmm::device_buffer> dictionary_page_buffers =
    fetch_byte_ranges(file_buffer_span, dict_page_byte_ranges, stream);

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               try_drop_l3_cache();
               timer.start();
               std::ignore = reader->filter_row_groups_with_dictionary_pages(
                 dictionary_page_buffers, input_row_group_indices, read_opts, stream);
               timer.stop();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(
    static_cast<double>(cardinality * num_row_groups * average_str_length) / time,
    "strings_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void BM_parquet_filter_string_rowgroups_with_dicts(nvbench::state& state)
{
  auto const num_row_groups = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const min_length     = static_cast<cudf::size_type>(state.get_int64("min_length"));
  auto const max_length     = static_cast<cudf::size_type>(state.get_int64("max_length"));
  auto const cardinality    = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const is_inline_eval = static_cast<bool>(state.get_int64("is_inline"));

  auto table_profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_length, max_length)
      .cardinality(cardinality);

  auto col_ref = cudf::ast::column_name_reference("_col0");
  auto scalar  = cudf::string_scalar("000010000");
  auto literal = cudf::ast::literal(scalar);
  auto expr1   = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal);
  auto expr2   = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col_ref, literal);
  auto expr3   = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal);

  auto filter_expr_few_literals =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr1, expr2);

  auto filter_expr_many_literals =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, filter_expr_few_literals, expr3);

  return BM_parquet_filter_string_row_groups_with_dicts_common(
    state,
    table_profile,
    is_inline_eval ? filter_expr_few_literals : filter_expr_many_literals,
    num_row_groups,
    (static_cast<double>(min_length) + static_cast<double>(max_length)) / 2,
    cardinality);
}

NVBENCH_BENCH(BM_parquet_filter_string_rowgroups_with_dicts)
  .set_name("parquet_filter_string_rowgroups_with_dicts")
  .set_min_samples(4)
  .add_int64_axis("num_row_groups", {32, 64, 128})
  .add_int64_axis("min_length", {4})
  .add_int64_axis("max_length", {32, 64, 128})
  .add_int64_axis("cardinality", {100, 1'000, 10'000})
  .add_int64_axis("is_inline", {true, false});
