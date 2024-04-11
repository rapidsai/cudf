/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/orc.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::io::statistics_freq,
  [](auto value) {
    switch (value) {
      case cudf::io::statistics_freq::STATISTICS_NONE: return "STATISTICS_NONE";
      case cudf::io::statistics_freq::STATISTICS_ROWGROUP: return "ORC_STATISTICS_STRIPE";
      case cudf::io::statistics_freq::STATISTICS_PAGE: return "ORC_STATISTICS_ROW_GROUP";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

// Size of the data in the benchmark dataframe; chosen to be low enough to allow benchmarks to
// run on most GPUs, but large enough to allow highest throughput
constexpr int64_t data_size        = 512 << 20;
constexpr cudf::size_type num_cols = 64;

template <data_type DataType>
void BM_orc_write_encode(nvbench::state& state, nvbench::type_list<nvbench::enum_type<DataType>>)
{
  auto const d_type                 = get_type_or_group(static_cast<int32_t>(DataType));
  cudf::size_type const cardinality = state.get_int64("cardinality");
  cudf::size_type const run_length  = state.get_int64("run_length");
  auto const compression            = cudf::io::compression_type::SNAPPY;
  auto const sink_type              = io_type::VOID;

  auto const tbl =
    create_random_table(cycle_dtypes(d_type, num_cols),
                        table_size_bytes{data_size},
                        data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
  auto const view = tbl->view();

  std::size_t encoded_file_size = 0;

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               cuio_source_sink_pair source_sink(sink_type);

               timer.start();
               cudf::io::orc_writer_options options =
                 cudf::io::orc_writer_options::builder(source_sink.make_sink_info(), view)
                   .compression(compression);
               cudf::io::write_orc(options);
               timer.stop();

               encoded_file_size = source_sink.size();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(encoded_file_size, "encoded_file_size", "encoded_file_size");
}

template <cudf::io::io_type IO, cudf::io::compression_type Compression>
void BM_orc_write_io_compression(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<IO>, nvbench::enum_type<Compression>>)
{
  auto const d_type = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL_SIGNED),
                                         static_cast<int32_t>(data_type::FLOAT),
                                         static_cast<int32_t>(data_type::DECIMAL),
                                         static_cast<int32_t>(data_type::TIMESTAMP),
                                         static_cast<int32_t>(data_type::STRING),
                                         static_cast<int32_t>(data_type::LIST),
                                         static_cast<int32_t>(data_type::STRUCT)});

  cudf::size_type const cardinality = state.get_int64("cardinality");
  cudf::size_type const run_length  = state.get_int64("run_length");
  auto const compression            = Compression;
  auto const sink_type              = IO;

  auto const tbl =
    create_random_table(cycle_dtypes(d_type, num_cols),
                        table_size_bytes{data_size},
                        data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
  auto const view = tbl->view();

  std::size_t encoded_file_size = 0;

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               cuio_source_sink_pair source_sink(sink_type);

               timer.start();
               cudf::io::orc_writer_options options =
                 cudf::io::orc_writer_options::builder(source_sink.make_sink_info(), view)
                   .compression(compression);
               cudf::io::write_orc(options);
               timer.stop();

               encoded_file_size = source_sink.size();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(encoded_file_size, "encoded_file_size", "encoded_file_size");
}

template <cudf::io::statistics_freq Statistics, cudf::io::compression_type Compression>
void BM_orc_write_statistics(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<Statistics>, nvbench::enum_type<Compression>>)
{
  auto const d_type = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL_SIGNED),
                                         static_cast<int32_t>(data_type::FLOAT),
                                         static_cast<int32_t>(data_type::DECIMAL),
                                         static_cast<int32_t>(data_type::TIMESTAMP),
                                         static_cast<int32_t>(data_type::STRING),
                                         static_cast<int32_t>(data_type::LIST)});

  auto const compression = Compression;
  auto const stats_freq  = Statistics;

  auto const tbl  = create_random_table(d_type, table_size_bytes{data_size});
  auto const view = tbl->view();

  std::size_t encoded_file_size = 0;

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               cuio_source_sink_pair source_sink(io_type::FILEPATH);

               timer.start();
               cudf::io::orc_writer_options const options =
                 cudf::io::orc_writer_options::builder(source_sink.make_sink_info(), view)
                   .compression(compression)
                   .enable_statistics(stats_freq);
               cudf::io::write_orc(options);
               timer.stop();

               encoded_file_size = source_sink.size();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(encoded_file_size, "encoded_file_size", "encoded_file_size");
}

using d_type_list = nvbench::enum_type_list<data_type::INTEGRAL_SIGNED,
                                            data_type::FLOAT,
                                            data_type::DECIMAL,
                                            data_type::TIMESTAMP,
                                            data_type::STRING,
                                            data_type::LIST,
                                            data_type::STRUCT>;

using io_list = nvbench::enum_type_list<cudf::io::io_type::FILEPATH,
                                        cudf::io::io_type::HOST_BUFFER,
                                        cudf::io::io_type::VOID>;

using compression_list =
  nvbench::enum_type_list<cudf::io::compression_type::SNAPPY, cudf::io::compression_type::NONE>;

using stats_list = nvbench::enum_type_list<cudf::io::STATISTICS_NONE,
                                           cudf::io::ORC_STATISTICS_STRIPE,
                                           cudf::io::ORC_STATISTICS_ROW_GROUP>;

NVBENCH_BENCH_TYPES(BM_orc_write_encode, NVBENCH_TYPE_AXES(d_type_list))
  .set_name("orc_write_encode")
  .set_type_axes_names({"data_type"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});

NVBENCH_BENCH_TYPES(BM_orc_write_io_compression, NVBENCH_TYPE_AXES(io_list, compression_list))
  .set_name("orc_write_io_compression")
  .set_type_axes_names({"io", "compression"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});

NVBENCH_BENCH_TYPES(BM_orc_write_statistics, NVBENCH_TYPE_AXES(stats_list, compression_list))
  .set_name("orc_write_statistics")
  .set_type_axes_names({"statistics", "compression"})
  .set_min_samples(4);
