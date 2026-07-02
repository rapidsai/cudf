/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/csv.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

constexpr cudf::size_type num_cols = 64;

// Quoting density: controls the fraction of columns that are STRING type
// (with special chars that force CSV quoting via the 4-state FSM).
enum class quote_pct : int32_t {
  QUOTE_0_PCT   = 0,    // All integral columns — no quoting
  QUOTE_25_PCT  = 25,   // 25% string columns with special chars, 75% integral
  QUOTE_100_PCT = 100,  // All string columns with special chars
};

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  quote_pct,
  [](quote_pct value) {
    switch (value) {
      case quote_pct::QUOTE_0_PCT: return "QUOTE_0_PCT";
      case quote_pct::QUOTE_25_PCT: return "QUOTE_25_PCT";
      case quote_pct::QUOTE_100_PCT: return "QUOTE_100_PCT";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

// Profile for string columns: ASCII 32-126 includes comma, double-quote, and hash.
// The CSV writer will be forced to quote fields containing these characters.
data_profile const quoted_string_profile =
  data_profile_builder().string_char_range(' ', '~');  // ASCII 32-126

template <quote_pct QuotePct>
void BM_csv_read_quoting(nvbench::state& state,
                         nvbench::type_list<nvbench::enum_type<QuotePct>>)
{
  auto const data_size_mb = static_cast<size_t>(state.get_int64("data_size_mb"));
  auto const data_size    = data_size_mb << 20;

  // Determine column type mix based on quoting percentage
  auto const string_cols = static_cast<cudf::size_type>(
    static_cast<int32_t>(QuotePct) * num_cols / 100);
  auto const integral_cols = num_cols - string_cols;

  // Build column types: integral columns first, then string columns
  std::vector<cudf::type_id> col_types;
  col_types.reserve(num_cols);

  if (integral_cols > 0) {
    auto const int_types = get_type_or_group(static_cast<int32_t>(data_type::INTEGRAL));
    auto const cycled_ints = cycle_dtypes(int_types, integral_cols);
    col_types.insert(col_types.end(), cycled_ints.begin(), cycled_ints.end());
  }
  for (cudf::size_type i = 0; i < string_cols; ++i) {
    col_types.push_back(cudf::type_id::STRING);
  }

  auto const tbl =
    create_random_table(col_types, table_size_bytes{data_size}, quoted_string_profile);
  auto const view = tbl->view();

  // Write to CSV — the writer will auto-quote string fields containing commas/quotes
  cuio_source_sink_pair source_sink(io_type::DEVICE_BUFFER);
  cudf::io::csv_writer_options write_options =
    cudf::io::csv_writer_options::builder(source_sink.make_sink_info(), view).include_header(true);
  cudf::io::write_csv(write_options);

  // Extract column types for explicit dtype specification (avoids type inference cost)
  std::vector<cudf::data_type> column_types;
  column_types.reserve(view.num_columns());
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    column_types.push_back(view.column(i).type());
  }

  cudf::io::csv_reader_options const read_options =
    cudf::io::csv_reader_options::builder(source_sink.make_source_info())
      .compression(cudf::io::compression_type::NONE)
      .dtypes(column_types);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      timer.start();
      auto const result = cudf::io::read_csv(read_options);
      timer.stop();

      CUDF_EXPECTS(result.tbl->num_columns() == view.num_columns(), "Unexpected number of columns");
      CUDF_EXPECTS(result.tbl->num_rows() == view.num_rows(), "Unexpected number of rows");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

using quote_pct_list =
  nvbench::enum_type_list<quote_pct::QUOTE_0_PCT, quote_pct::QUOTE_25_PCT, quote_pct::QUOTE_100_PCT>;

NVBENCH_BENCH_TYPES(BM_csv_read_quoting, NVBENCH_TYPE_AXES(quote_pct_list))
  .set_name("csv_read_quoting")
  .set_type_axes_names({"quote_pct"})
  .set_min_samples(4)
  .add_int64_axis("data_size_mb", {64, 256});
