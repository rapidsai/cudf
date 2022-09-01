/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/io/orc.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

constexpr int64_t data_size        = 512 << 20;
constexpr cudf::size_type num_cols = 64;

void orc_read_common(cudf::io::orc_writer_options const& opts,
                     cuio_source_sink_pair& source_sink,
                     nvbench::state& state)
{
  cudf::io::write_orc(opts);

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(source_sink.make_source_info());

  auto mem_stats_logger = cudf::memory_stats_logger();  // init stats logger
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::default_stream_value.value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               try_drop_l3_cache();

               timer.start();
               cudf::io::read_orc(read_opts);
               timer.stop();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

template <data_type DataType>
void BM_orc_read_data(nvbench::state& state, nvbench::type_list<nvbench::enum_type<DataType>>)
{
  cudf::rmm_pool_raii rmm_pool;

  auto const d_type                 = get_type_or_group(static_cast<int32_t>(DataType));
  cudf::size_type const cardinality = state.get_int64("cardinality");
  cudf::size_type const run_length  = state.get_int64("run_length");

  auto const tbl =
    create_random_table(cycle_dtypes(d_type, num_cols),
                        table_size_bytes{data_size},
                        data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  cudf::io::orc_writer_options opts =
    cudf::io::orc_writer_options::builder(source_sink.make_sink_info(), view);

  orc_read_common(opts, source_sink, state);
}

template <cudf::io::io_type IO, cudf::io::compression_type Compression>
void BM_orc_read_io_compression(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<IO>, nvbench::enum_type<Compression>>)
{
  cudf::rmm_pool_raii rmm_pool;

  auto const d_type = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL_SIGNED),
                                         static_cast<int32_t>(data_type::FLOAT),
                                         static_cast<int32_t>(data_type::DECIMAL),
                                         static_cast<int32_t>(data_type::TIMESTAMP),
                                         static_cast<int32_t>(data_type::STRING),
                                         static_cast<int32_t>(data_type::LIST),
                                         static_cast<int32_t>(data_type::STRUCT)});

  cudf::size_type const cardinality = state.get_int64("cardinality");
  cudf::size_type const run_length  = state.get_int64("run_length");

  auto const tbl =
    create_random_table(cycle_dtypes(d_type, num_cols),
                        table_size_bytes{data_size},
                        data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(IO);
  cudf::io::orc_writer_options opts =
    cudf::io::orc_writer_options::builder(source_sink.make_sink_info(), view)
      .compression(Compression);

  orc_read_common(opts, source_sink, state);
}

using d_type_list = nvbench::enum_type_list<data_type::INTEGRAL_SIGNED,
                                            data_type::FLOAT,
                                            data_type::DECIMAL,
                                            data_type::TIMESTAMP,
                                            data_type::STRING,
                                            data_type::LIST,
                                            data_type::STRUCT>;

using io_list =
  nvbench::enum_type_list<cudf::io::io_type::FILEPATH, cudf::io::io_type::HOST_BUFFER>;

using compression_list =
  nvbench::enum_type_list<cudf::io::compression_type::SNAPPY, cudf::io::compression_type::NONE>;

NVBENCH_BENCH_TYPES(BM_orc_read_data, NVBENCH_TYPE_AXES(d_type_list))
  .set_name("orc_read_decode")
  .set_type_axes_names({"data_type"})
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});

NVBENCH_BENCH_TYPES(BM_orc_read_io_compression, NVBENCH_TYPE_AXES(io_list, compression_list))
  .set_name("orc_read_io_compression")
  .set_type_axes_names({"io", "compression"})
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});
