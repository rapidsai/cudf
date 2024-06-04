/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

namespace {

// Size of the data in the benchmark dataframe; chosen to be low enough to allow benchmarks to
// run on most GPUs, but large enough to allow highest throughput
constexpr cudf::size_type num_cols = 64;
constexpr std::size_t data_size    = 512 << 20;
constexpr std::size_t Mbytes       = 1024 * 1024;

template <bool is_chunked_read>
void orc_read_common(cudf::size_type num_rows_to_read,
                     cuio_source_sink_pair& source_sink,
                     nvbench::state& state)
{
  auto const read_opts =
    cudf::io::orc_reader_options::builder(source_sink.make_source_info()).build();

  auto mem_stats_logger = cudf::memory_stats_logger();  // init stats logger
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  if constexpr (is_chunked_read) {
    state.exec(
      nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
        try_drop_l3_cache();
        auto const output_limit_MB =
          static_cast<std::size_t>(state.get_int64("chunk_read_limit_MB"));
        auto const read_limit_MB = static_cast<std::size_t>(state.get_int64("pass_read_limit_MB"));

        auto reader =
          cudf::io::chunked_orc_reader(output_limit_MB * Mbytes, read_limit_MB * Mbytes, read_opts);
        cudf::size_type num_rows{0};

        timer.start();
        do {
          auto chunk = reader.read_chunk();
          num_rows += chunk.tbl->num_rows();
        } while (reader.has_next());
        timer.stop();

        CUDF_EXPECTS(num_rows == num_rows_to_read, "Unexpected number of rows");
      });
  } else {  // not is_chunked_read
    state.exec(
      nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
        try_drop_l3_cache();

        timer.start();
        auto const result = cudf::io::read_orc(read_opts);
        timer.stop();

        CUDF_EXPECTS(result.tbl->num_columns() == num_cols, "Unexpected number of columns");
        CUDF_EXPECTS(result.tbl->num_rows() == num_rows_to_read, "Unexpected number of rows");
      });
  }

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

}  // namespace

template <data_type DataType, io_type IOType>
void BM_orc_read_data(nvbench::state& state,
                      nvbench::type_list<nvbench::enum_type<DataType>, nvbench::enum_type<IOType>>)
{
  auto const d_type                 = get_type_or_group(static_cast<int32_t>(DataType));
  cudf::size_type const cardinality = state.get_int64("cardinality");
  cudf::size_type const run_length  = state.get_int64("run_length");
  cuio_source_sink_pair source_sink(IOType);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, num_cols),
      table_size_bytes{data_size},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::orc_writer_options opts =
      cudf::io::orc_writer_options::builder(source_sink.make_sink_info(), view);
    cudf::io::write_orc(opts);
    return view.num_rows();
  }();

  orc_read_common<false>(num_rows_written, source_sink, state);
}

template <io_type IOType, cudf::io::compression_type Compression, bool chunked_read>
void orc_read_io_compression(nvbench::state& state)
{
  auto const d_type = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL_SIGNED),
                                         static_cast<int32_t>(data_type::FLOAT),
                                         static_cast<int32_t>(data_type::DECIMAL),
                                         static_cast<int32_t>(data_type::TIMESTAMP),
                                         static_cast<int32_t>(data_type::STRING),
                                         static_cast<int32_t>(data_type::LIST),
                                         static_cast<int32_t>(data_type::STRUCT)});

  auto const [cardinality, run_length] = [&]() -> std::pair<cudf::size_type, cudf::size_type> {
    if constexpr (chunked_read) {
      return {0, 4};
    } else {
      return {static_cast<cudf::size_type>(state.get_int64("cardinality")),
              static_cast<cudf::size_type>(state.get_int64("run_length"))};
    }
  }();
  cuio_source_sink_pair source_sink(IOType);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, num_cols),
      table_size_bytes{data_size},
      data_profile_builder{}.cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::orc_writer_options opts =
      cudf::io::orc_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(Compression);
    cudf::io::write_orc(opts);
    return view.num_rows();
  }();

  orc_read_common<chunked_read>(num_rows_written, source_sink, state);
}

template <io_type IOType, cudf::io::compression_type Compression>
void BM_orc_read_io_compression(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<IOType>, nvbench::enum_type<Compression>>)
{
  return orc_read_io_compression<IOType, Compression, false>(state);
}

template <cudf::io::compression_type Compression>
void BM_orc_chunked_read_io_compression(nvbench::state& state,
                                        nvbench::type_list<nvbench::enum_type<Compression>>)
{
  // Only run benchmark using HOST_BUFFER IO.
  return orc_read_io_compression<io_type::HOST_BUFFER, Compression, true>(state);
}

using d_type_list = nvbench::enum_type_list<data_type::INTEGRAL_SIGNED,
                                            data_type::FLOAT,
                                            data_type::DECIMAL,
                                            data_type::TIMESTAMP,
                                            data_type::STRING,
                                            data_type::LIST,
                                            data_type::STRUCT>;

using io_list =
  nvbench::enum_type_list<io_type::FILEPATH, io_type::HOST_BUFFER, io_type::DEVICE_BUFFER>;

using compression_list =
  nvbench::enum_type_list<cudf::io::compression_type::SNAPPY, cudf::io::compression_type::NONE>;

NVBENCH_BENCH_TYPES(BM_orc_read_data,
                    NVBENCH_TYPE_AXES(d_type_list, nvbench::enum_type_list<io_type::DEVICE_BUFFER>))
  .set_name("orc_read_decode")
  .set_type_axes_names({"data_type", "io"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});

NVBENCH_BENCH_TYPES(BM_orc_read_io_compression, NVBENCH_TYPE_AXES(io_list, compression_list))
  .set_name("orc_read_io_compression")
  .set_type_axes_names({"io", "compression"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});

// Should have the same parameters as `BM_orc_read_io_compression` for comparison.
NVBENCH_BENCH_TYPES(BM_orc_chunked_read_io_compression, NVBENCH_TYPE_AXES(compression_list))
  .set_name("orc_chunked_read_io_compression")
  .set_type_axes_names({"compression"})
  .set_min_samples(4)
  // The input has approximately 520MB and 127K rows.
  // The limits below are given in MBs.
  .add_int64_axis("chunk_read_limit_MB", {50, 250, 700})
  .add_int64_axis("pass_read_limit_MB", {50, 250, 700});
