/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/column/column.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Benchmark for the parquet-dictionary -> cudf DICTIONARY32 transcode fast path enabled by
// `parquet_reader_options::output_dict_columns`. It reads a fully dictionary-encoded set of
// low-cardinality string columns under three modes, selected by the `mode` axis, so the transcode
// path can be judged against both a lower and an upper reference:
//
//   - "plain_string":  reader default; columns materialize as STRING. This is the cheapest possible
//                      read (no dictionary built) and serves as the lower-bound reference -- it does
//                      strictly less work and produces a different (STRING) output.
//   - "decode_encode": read as STRING, then `cudf::dictionary::encode` each column to DICTIONARY32.
//                      This is the pre-existing way to obtain a dictionary column and is the fair
//                      apples-to-apples baseline the transcode fast path aims to beat.
//   - "transcode":     `output_dict_columns=true`; the reader keeps the dictionary representation and
//                      emits DICTIONARY32 directly, skipping string materialization.
//
// Both "decode_encode" and "transcode" produce DICTIONARY32 output, so their times and peak memory
// are directly comparable; "plain_string" shows the floor cost of just decoding.

namespace {

enum class bench_mode { plain_string, decode_encode, transcode };

[[nodiscard]] bench_mode parse_mode(std::string const& mode)
{
  if (mode == "plain_string") { return bench_mode::plain_string; }
  if (mode == "decode_encode") { return bench_mode::decode_encode; }
  if (mode == "transcode") { return bench_mode::transcode; }
  CUDF_FAIL("Unknown benchmark mode: " + mode);
}

// The transcode fast path requires every data page of an eligible column to be dictionary-encoded.
// Forcing `dictionary_policy::ALWAYS` together with low cardinality guarantees this for the
// generated data.
void write_dict_encoded_parquet(cudf::table_view const& view,
                                cuio_source_sink_pair& source_sink,
                                int64_t row_group_size_rows)
{
  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .compression(cudf::io::compression_type::NONE)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);
  // Sentinel 0 == use cuDF default row-group sizing.
  if (row_group_size_rows > 0) { write_opts.set_row_group_size_rows(row_group_size_rows); }
  cudf::io::write_parquet(write_opts);
}

}  // namespace

void BM_parquet_read_dict_transcode(nvbench::state& state)
{
  auto const cardinality       = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const data_size         = static_cast<size_t>(state.get_int64("data_size"));
  auto const num_cols          = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const rg_size_rows      = state.get_int64("row_group_size_rows");
  auto const mode              = parse_mode(state.get_string("mode"));
  auto const avg_string_length = static_cast<cudf::size_type>(state.get_int64("avg_string_length"));
  auto const source_type       = retrieve_io_type_enum(state.get_string("io_type"));

  // nvbench axes form a full Cartesian product, so the (large data_size x single-column) cell is
  // generated even though we don't want it: a single ~2 GB STRING column would overflow 32-bit
  // string offsets. Skip that combination and keep the large data size to multi-column runs only.
  if (num_cols == 1 and data_size > (std::size_t{512} << 20)) {
    state.skip(
      "Single-column reads above 512 MiB would overflow 32-bit string offsets; the large "
      "data_size point is restricted to multi-column configurations");
    return;
  }

  // corresponds to 3 sigma (full width 6 sigma: 99.7% of range)
  auto const half_width = avg_string_length >> 3;
  auto const length_min = avg_string_length - half_width;
  auto const length_max = avg_string_length + half_width;

  data_profile const profile =
    data_profile_builder()
      .cardinality(cardinality)
      .avg_run_length(1)
      .distribution(data_type::STRING, distribution_id::NORMAL, length_min, length_max);

  auto const d_type = get_type_or_group(static_cast<int32_t>(data_type::STRING));
  auto const tbl =
    create_random_table(cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, profile);
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(source_type);
  write_dict_encoded_parquet(view, source_sink, rg_size_rows);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info())
      .output_dict_columns(mode == bench_mode::transcode);

  // Perform the full work for the selected mode: read, and for `decode_encode` additionally encode
  // each STRING column to DICTIONARY32. Returns the resulting table so it can be reused for both the
  // outside-the-timed-region verification and the timed measurement.
  auto const run_mode = [&]() -> std::unique_ptr<cudf::table> {
    auto result = cudf::io::read_parquet(read_opts);
    if (mode == bench_mode::decode_encode) {
      std::vector<std::unique_ptr<cudf::column>> encoded;
      encoded.reserve(result.tbl->num_columns());
      for (auto const& col : result.tbl->view()) {
        encoded.push_back(cudf::dictionary::encode(col));
      }
      return std::make_unique<cudf::table>(std::move(encoded));
    }
    return std::move(result.tbl);
  };

  // Sanity check (outside the timed region, run for every mode so warm-up is symmetric): confirm the
  // produced column types match the mode, otherwise the benchmark would silently measure the wrong
  // path (e.g. `transcode` falling back to STRING because the data is not fully dictionary-encoded).
  {
    auto const probe = run_mode();
    CUDF_EXPECTS(probe->num_columns() == num_cols, "Unexpected number of columns");
    auto const expected_id =
      (mode == bench_mode::plain_string) ? cudf::type_id::STRING : cudf::type_id::DICTIONARY32;
    for (auto const& col : probe->view()) {
      CUDF_EXPECTS(col.type().id() == expected_id,
                   "Produced column type does not match the benchmark mode; check that the "
                   "generated data is fully dictionary-encoded");
    }
  }

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               drop_page_cache_if_enabled(read_opts.get_source().filepaths());

               timer.start();
               auto const result = run_mode();
               timer.stop();

               CUDF_EXPECTS(result->num_columns() == num_cols, "Unexpected number of columns");
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_element_count(static_cast<double>(view.num_rows()) / time, "rows_per_sec");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

NVBENCH_BENCH(BM_parquet_read_dict_transcode)
  .set_name("parquet_read_dict_transcode")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_string_axis("mode", {"plain_string", "decode_encode", "transcode"})
  .add_int64_axis("cardinality", {100, 1'000, 10'000})
  .add_int64_axis("num_cols", {1, 8})
  // 2 GiB point is single-column-skipped in the body (see state.skip): a lone ~2 GB string column
  // would overflow 32-bit string offsets, so the large size only runs with num_cols > 1.
  .add_int64_axis("data_size", {std::int64_t{512} << 20, std::int64_t{2} << 30})
  .add_int64_axis("avg_string_length", {16})
  // Sentinel 0 == default row groups; small values force multiple row groups, exercising the
  // per-row-group key concatenation / index remapping path.
  .add_int64_axis("row_group_size_rows", {0, 100'000});
