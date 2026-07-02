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

constexpr cudf::size_type num_cols = 16;

void BM_csv_write_scale(nvbench::state& state)
{
  auto const data_size_mb = state.get_int64("data_size_mb");
  auto const data_size    = static_cast<std::size_t>(data_size_mb) << 20;

  // Mixed-type columns: INTEGRAL, FLOAT, TIMESTAMP, STRING — 16 columns cycled
  auto const data_types = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                             static_cast<int32_t>(data_type::FLOAT),
                                             static_cast<int32_t>(data_type::TIMESTAMP),
                                             static_cast<int32_t>(data_type::STRING)});

  auto const tbl =
    create_random_table(cycle_dtypes(data_types, num_cols), table_size_bytes{data_size});
  auto const view = tbl->view();

  std::size_t encoded_file_size = 0;

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);

               timer.start();
               cudf::io::csv_writer_options options =
                 cudf::io::csv_writer_options::builder(source_sink.make_sink_info(), view);
               cudf::io::write_csv(options);
               timer.stop();

               encoded_file_size = source_sink.size();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(encoded_file_size, "encoded_file_size", "encoded_file_size");
}

NVBENCH_BENCH(BM_csv_write_scale)
  .set_name("csv_write_scale")
  .set_min_samples(2)
  .add_int64_axis("data_size_mb", {256, 512, 1024, 2048, 4096});
