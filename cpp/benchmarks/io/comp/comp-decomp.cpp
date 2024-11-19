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

#include "io/comp/comp.hpp"

#include "io/comp/io_uncomp.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/json.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

// Size of the data in the benchmark dataframe; chosen to be low enough to allow benchmarks to
// run on most GPUs, but large enough to allow highest throughput
constexpr size_t data_size         = 512 << 20;
constexpr cudf::size_type num_cols = 64;

template <cudf::io::compression_type comptype>
void BM_comp_json_data(nvbench::state& state, nvbench::type_list<nvbench::enum_type<comptype>>)
{
  size_t const data_size = state.get_int64("data_size");
  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  auto const d_type = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                         static_cast<int32_t>(data_type::FLOAT),
                                         static_cast<int32_t>(data_type::DECIMAL),
                                         static_cast<int32_t>(data_type::TIMESTAMP),
                                         static_cast<int32_t>(data_type::DURATION),
                                         static_cast<int32_t>(data_type::STRING),
                                         static_cast<int32_t>(data_type::LIST),
                                         static_cast<int32_t>(data_type::STRUCT)});
  auto const tbl    = create_random_table(
    cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, data_profile_builder());
  auto const view = tbl->view();
  cudf::io::json_writer_options const write_opts =
    cudf::io::json_writer_options::builder(source_sink.make_sink_info(), view)
      .na_rep("null")
      .rows_per_chunk(100'000);
  cudf::io::write_json(write_opts);

  auto hbufs = source_sink.make_source_info().host_buffers();
  CUDF_EXPECTS(hbufs.size() == 1, "something is wrong");

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               try_drop_l3_cache();

               timer.start();
               auto chbuf = cudf::io::detail::compress(
                 comptype,
                 cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(hbufs[0].data()),
                                                hbufs[0].size()),
                 cudf::get_default_stream());
               timer.stop();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

template <cudf::io::compression_type comptype>
void BM_decomp_json_data(nvbench::state& state, nvbench::type_list<nvbench::enum_type<comptype>>)
{
  size_t const data_size = state.get_int64("data_size");
  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  auto const d_type = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                         static_cast<int32_t>(data_type::FLOAT),
                                         static_cast<int32_t>(data_type::DECIMAL),
                                         static_cast<int32_t>(data_type::TIMESTAMP),
                                         static_cast<int32_t>(data_type::DURATION),
                                         static_cast<int32_t>(data_type::STRING),
                                         static_cast<int32_t>(data_type::LIST),
                                         static_cast<int32_t>(data_type::STRUCT)});
  auto const tbl    = create_random_table(
    cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, data_profile_builder());
  auto const view = tbl->view();
  cudf::io::json_writer_options const write_opts =
    cudf::io::json_writer_options::builder(source_sink.make_sink_info(), view)
      .na_rep("null")
      .rows_per_chunk(100'000);
  cudf::io::write_json(write_opts);

  auto hbufs = source_sink.make_source_info().host_buffers();
  CUDF_EXPECTS(hbufs.size() == 1, "something is wrong");
  auto chbuf = cudf::io::detail::compress(
    comptype,
    cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(hbufs[0].data()),
                                   hbufs[0].size()),
    cudf::get_default_stream());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               try_drop_l3_cache();

               timer.start();
               auto hbuf = cudf::io::detail::decompress(comptype, chbuf);
               timer.stop();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

using compression_list =
  nvbench::enum_type_list<cudf::io::compression_type::SNAPPY, cudf::io::compression_type::GZIP>;

NVBENCH_BENCH_TYPES(BM_comp_json_data, NVBENCH_TYPE_AXES(compression_list))
  .set_name("comp_json_data")
  .set_type_axes_names({"compression_type"})
  .add_int64_power_of_two_axis("data_size", nvbench::range(10, 25, 1))
  .set_min_samples(4);

NVBENCH_BENCH_TYPES(BM_decomp_json_data, NVBENCH_TYPE_AXES(compression_list))
  .set_name("decomp_json_data")
  .set_type_axes_names({"compression_type"})
  .add_int64_power_of_two_axis("data_size", nvbench::range(10, 25, 1))
  .set_min_samples(4);
