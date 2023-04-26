/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <benchmarks/fixture/rmm_pool_raii.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

// Size of the data in the the benchmark dataframe; chosen to be low enough to allow benchmarks to
// run on most GPUs, but large enough to allow highest throughput
constexpr size_t data_size         = 512 << 20;
constexpr cudf::size_type num_cols = 64;

void parquet_read_common(cudf::io::parquet_writer_options const& write_opts,
                         cuio_source_sink_pair& source_sink,
                         nvbench::state& state)
{
  cudf::io::write_parquet(write_opts);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               try_drop_l3_cache();

               timer.start();
               cudf::io::read_parquet(read_opts);
               timer.stop();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

template <data_type DataType, cudf::io::io_type IOType>
void BM_parquet_read_data(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<DataType>, nvbench::enum_type<IOType>>)
{
  auto const d_type                 = get_type_or_group(static_cast<int32_t>(DataType));
  cudf::size_type const cardinality = state.get_int64("cardinality");
  cudf::size_type const run_length  = state.get_int64("run_length");
  auto const compression            = cudf::io::compression_type::SNAPPY;

  auto const tbl =
    create_random_table(cycle_dtypes(d_type, num_cols),
                        table_size_bytes{data_size},
                        data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(IOType);
  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .compression(compression);

  parquet_read_common(write_opts, source_sink, state);
}

template <cudf::io::io_type IOType, cudf::io::compression_type Compression>
void BM_parquet_read_io_compression(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<IOType>, nvbench::enum_type<Compression>>)
{
  auto const d_type = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                         static_cast<int32_t>(data_type::FLOAT),
                                         static_cast<int32_t>(data_type::DECIMAL),
                                         static_cast<int32_t>(data_type::TIMESTAMP),
                                         static_cast<int32_t>(data_type::DURATION),
                                         static_cast<int32_t>(data_type::STRING),
                                         static_cast<int32_t>(data_type::LIST),
                                         static_cast<int32_t>(data_type::STRUCT)});

  cudf::size_type const cardinality = state.get_int64("cardinality");
  cudf::size_type const run_length  = state.get_int64("run_length");
  auto const compression            = Compression;
  auto const source_type            = IOType;

  auto const tbl =
    create_random_table(cycle_dtypes(d_type, num_cols),
                        table_size_bytes{data_size},
                        data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(source_type);
  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .compression(compression);

  parquet_read_common(write_opts, source_sink, state);
}

template <data_type DataType, cudf::io::io_type IOType>
void BM_parquet_read_chunks(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<DataType>, nvbench::enum_type<IOType>>)
{
  auto const d_type                 = get_type_or_group(static_cast<int32_t>(DataType));
  cudf::size_type const cardinality = state.get_int64("cardinality");
  cudf::size_type const run_length  = state.get_int64("run_length");
  cudf::size_type const byte_limit  = state.get_int64("byte_limit");
  auto const compression            = cudf::io::compression_type::SNAPPY;

  auto const tbl =
    create_random_table(cycle_dtypes(d_type, num_cols),
                        table_size_bytes{data_size},
                        data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(IOType);
  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .compression(compression);

  cudf::io::write_parquet(write_opts);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info());
  auto reader = cudf::io::chunked_parquet_reader(byte_limit, read_opts);

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               try_drop_l3_cache();

               timer.start();
               do {
                 auto chunk = reader.read_chunk();
               } while (reader.has_next());
               timer.stop();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

using d_type_list = nvbench::enum_type_list<data_type::INTEGRAL,
                                            data_type::FLOAT,
                                            data_type::DECIMAL,
                                            data_type::TIMESTAMP,
                                            data_type::DURATION,
                                            data_type::STRING,
                                            data_type::LIST,
                                            data_type::STRUCT>;

using io_list = nvbench::enum_type_list<cudf::io::io_type::FILEPATH,
                                        cudf::io::io_type::HOST_BUFFER,
                                        cudf::io::io_type::DEVICE_BUFFER>;

using compression_list =
  nvbench::enum_type_list<cudf::io::compression_type::SNAPPY, cudf::io::compression_type::NONE>;

NVBENCH_BENCH_TYPES(BM_parquet_read_data,
                    NVBENCH_TYPE_AXES(d_type_list,
                                      nvbench::enum_type_list<cudf::io::io_type::DEVICE_BUFFER>))
  .set_name("parquet_read_decode")
  .set_type_axes_names({"data_type", "io"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});

NVBENCH_BENCH_TYPES(BM_parquet_read_io_compression, NVBENCH_TYPE_AXES(io_list, compression_list))
  .set_name("parquet_read_io_compression")
  .set_type_axes_names({"io", "compression"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});

NVBENCH_BENCH_TYPES(BM_parquet_read_chunks,
                    NVBENCH_TYPE_AXES(d_type_list,
                                      nvbench::enum_type_list<cudf::io::io_type::DEVICE_BUFFER>))
  .set_name("parquet_read_chunks")
  .set_type_axes_names({"data_type", "io"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("byte_limit", {0, 500'000});
