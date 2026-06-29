/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

// Benchmark for the parquet-dictionary -> cudf DICTIONARY32 transcode fast path enabled by
// `parquet_reader_options::try_output_dict_columns`. It reads a fully dictionary-encoded set of
// low-cardinality string columns both with the option off (the column materializes as STRING) and
// with the option on (the reader keeps the dictionary representation and emits DICTIONARY32). The
// `try_output_dict_columns` axis lets the two paths be compared directly.

namespace {

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
  auto const try_dict          = static_cast<bool>(state.get_int64("try_output_dict_columns"));
  auto const avg_string_length = static_cast<cudf::size_type>(state.get_int64("avg_string_length"));
  auto const source_type       = retrieve_io_type_enum(state.get_string("io_type"));

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
      .try_output_dict_columns(try_dict);

  // Sanity check (outside the timed region): when the option is on the eligible string columns must
  // come back as DICTIONARY32, otherwise the benchmark would silently measure the plain path.
  if (try_dict) {
    auto const probe = cudf::io::read_parquet(read_opts);
    CUDF_EXPECTS(probe.tbl->num_columns() == num_cols, "Unexpected number of columns");
    CUDF_EXPECTS(probe.tbl->view().column(0).type().id() == cudf::type_id::DICTIONARY32,
                 "try_output_dict_columns did not produce a DICTIONARY32 column; check that the "
                 "generated data is fully dictionary-encoded");
  }

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               drop_page_cache_if_enabled(read_opts.get_source().filepaths());

               timer.start();
               auto const result = cudf::io::read_parquet(read_opts);
               timer.stop();

               CUDF_EXPECTS(result.tbl->num_columns() == num_cols, "Unexpected number of columns");
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
  .add_int64_axis("try_output_dict_columns", {0, 1})
  .add_int64_axis("cardinality", {100, 1'000, 10'000})
  .add_int64_axis("num_cols", {1, 8})
  .add_int64_axis("data_size", {512 << 20})
  .add_int64_axis("avg_string_length", {16})
  // Sentinel 0 == default row groups; small values force multiple row groups, exercising the
  // per-row-group key concatenation / index remapping path.
  .add_int64_axis("row_group_size_rows", {0, 100'000});
