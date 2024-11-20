/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/io/json.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

// Default size of the data in the benchmark dataframe; chosen to be low enough to allow benchmarks
// to run on most GPUs, but large enough to allow highest throughput
constexpr size_t default_data_size = 512 << 20;
constexpr cudf::size_type num_cols = 64;

void json_read_common(cuio_source_sink_pair& source_sink,
                      cudf::size_type num_rows_to_read,
                      nvbench::state& state,
                      cudf::io::compression_type comptype = cudf::io::compression_type::NONE,
                      size_t data_size                    = default_data_size)
{
  cudf::io::json_reader_options read_opts =
    cudf::io::json_reader_options::builder(source_sink.make_source_info()).compression(comptype);

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      try_drop_l3_cache();

      timer.start();
      auto const result = cudf::io::read_json(read_opts);
      timer.stop();

      CUDF_EXPECTS(result.tbl->num_columns() == num_cols, "Unexpected number of columns");
      CUDF_EXPECTS(result.tbl->num_rows() == num_rows_to_read, "Unexpected number of rows");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

cudf::size_type json_write_bm_data(
  cudf::io::sink_info sink,
  std::vector<cudf::type_id> const& dtypes,
  cudf::io::compression_type comptype = cudf::io::compression_type::NONE,
  size_t data_size                    = default_data_size)
{
  auto const tbl = create_random_table(
    cycle_dtypes(dtypes, num_cols), table_size_bytes{data_size}, data_profile_builder());
  auto const view = tbl->view();

  cudf::io::json_writer_options const write_opts =
    cudf::io::json_writer_options::builder(sink, view)
      .na_rep("null")
      .rows_per_chunk(100'000)
      .compression(comptype);
  cudf::io::write_json(write_opts);
  return view.num_rows();
}

template <io_type IO>
void BM_json_read_io(nvbench::state& state, nvbench::type_list<nvbench::enum_type<IO>>)
{
  cuio_source_sink_pair source_sink(IO);
  auto const d_type   = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                           static_cast<int32_t>(data_type::FLOAT),
                                           static_cast<int32_t>(data_type::DECIMAL),
                                           static_cast<int32_t>(data_type::TIMESTAMP),
                                           static_cast<int32_t>(data_type::DURATION),
                                           static_cast<int32_t>(data_type::STRING),
                                           static_cast<int32_t>(data_type::LIST),
                                           static_cast<int32_t>(data_type::STRUCT)});
  auto const num_rows = json_write_bm_data(source_sink.make_sink_info(), d_type);

  json_read_common(source_sink, num_rows, state);
}

template <cudf::io::compression_type comptype, io_type IO>
void BM_json_read_compressed_io(
  nvbench::state& state, nvbench::type_list<nvbench::enum_type<comptype>, nvbench::enum_type<IO>>)
{
  size_t const data_size = state.get_int64("data_size");
  cuio_source_sink_pair source_sink(IO);
  auto const d_type = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                         static_cast<int32_t>(data_type::FLOAT),
                                         static_cast<int32_t>(data_type::DECIMAL),
                                         static_cast<int32_t>(data_type::TIMESTAMP),
                                         static_cast<int32_t>(data_type::DURATION),
                                         static_cast<int32_t>(data_type::STRING),
                                         static_cast<int32_t>(data_type::LIST),
                                         static_cast<int32_t>(data_type::STRUCT)});
  auto const num_rows =
    json_write_bm_data(source_sink.make_sink_info(), d_type, comptype, data_size);

  json_read_common(source_sink, num_rows, state, comptype, data_size);
}

template <data_type DataType, io_type IO>
void BM_json_read_data_type(
  nvbench::state& state, nvbench::type_list<nvbench::enum_type<DataType>, nvbench::enum_type<IO>>)
{
  cuio_source_sink_pair source_sink(IO);
  auto const d_type   = get_type_or_group(static_cast<int32_t>(DataType));
  auto const num_rows = json_write_bm_data(source_sink.make_sink_info(), d_type);

  json_read_common(source_sink, num_rows, state);
}

using d_type_list = nvbench::enum_type_list<data_type::INTEGRAL,
                                            data_type::FLOAT,
                                            data_type::DECIMAL,
                                            data_type::TIMESTAMP,
                                            data_type::DURATION,
                                            data_type::STRING,
                                            data_type::LIST,
                                            data_type::STRUCT>;

using io_list =
  nvbench::enum_type_list<io_type::FILEPATH, io_type::HOST_BUFFER, io_type::DEVICE_BUFFER>;

using compression_list = nvbench::enum_type_list<cudf::io::compression_type::GZIP,
                                                 cudf::io::compression_type::SNAPPY,
                                                 cudf::io::compression_type::NONE>;

NVBENCH_BENCH_TYPES(BM_json_read_data_type,
                    NVBENCH_TYPE_AXES(d_type_list, nvbench::enum_type_list<io_type::DEVICE_BUFFER>))
  .set_name("json_read_data_type")
  .set_type_axes_names({"data_type", "io"})
  .set_min_samples(4);

NVBENCH_BENCH_TYPES(BM_json_read_io, NVBENCH_TYPE_AXES(io_list))
  .set_name("json_read_io")
  .set_type_axes_names({"io"})
  .set_min_samples(4);

NVBENCH_BENCH_TYPES(BM_json_read_compressed_io,
                    NVBENCH_TYPE_AXES(compression_list, nvbench::enum_type_list<io_type::FILEPATH>))
  .set_name("json_read_compressed_io")
  .set_type_axes_names({"compression_type", "io"})
  .add_int64_power_of_two_axis("data_size", nvbench::range(20, 29, 1))
  .set_min_samples(4);
