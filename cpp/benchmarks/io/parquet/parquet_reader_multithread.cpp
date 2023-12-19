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
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/thread_pool.hpp>

#include <cudf/detail/utilities/stream_pool.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

template <data_type DataType>
void BM_parquet_multithreaded_read(nvbench::state& state,
                                   nvbench::type_list<nvbench::enum_type<DataType>>)
{
  auto const d_type                 = get_type_or_group(static_cast<int32_t>(DataType));
  cudf::size_type const cardinality = state.get_int64("cardinality");
  cudf::size_type const run_length  = state.get_int64("run_length");
  cudf::size_type const num_cols    = state.get_int64("num_cols");
  size_t const data_size            = state.get_int64("data_size");
  size_t const chunk_size           = state.get_int64("chunk_size");
  auto const num_threads            = state.get_int64("num_threads");

  int const num_tables = data_size / chunk_size;
  std::vector<cuio_source_sink_pair> source_sink_vector;
  auto streams = cudf::detail::fork_streams(cudf::get_default_stream(), num_threads);
  cudf::detail::thread_pool threads(num_threads);

  size_t total_file_size = 0;

  for (auto i = 0; i < num_tables; ++i) {
    cuio_source_sink_pair source_sink{cudf::io::io_type::HOST_BUFFER};

    auto const tbl = create_random_table(
      cycle_dtypes(d_type, num_cols),
      table_size_bytes{chunk_size},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(cudf::io::compression_type::SNAPPY);

    cudf::io::write_parquet(write_opts);
    total_file_size += source_sink.size();

    source_sink_vector.push_back(std::move(source_sink));
  }

  auto mem_stats_logger = cudf::memory_stats_logger();

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      auto read_func = [&](int index) {
        auto const stream = streams[index % num_threads];
        auto& source_sink = source_sink_vector[index];
        cudf::io::parquet_reader_options read_opts =
          cudf::io::parquet_reader_options::builder(source_sink.make_source_info());
        auto datasources = cudf::io::datasource::create(read_opts.get_source().host_buffers());
        auto reader      = std::make_unique<cudf::io::parquet::detail::reader>(
          std::move(datasources), read_opts, stream, rmm::mr::get_current_device_resource());

        reader->read(read_opts);
      };

      threads.paused = true;
      for (auto i = 0; i < num_tables; ++i) {
        threads.submit(read_func, i);
      }
      timer.start();
      threads.paused = false;
      threads.wait_for_tasks();
      cudf::detail::join_streams(streams, cudf::get_default_stream());
      timer.stop();
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(total_file_size, "encoded_file_size", "encoded_file_size");
}

using d_type_list = nvbench::enum_type_list<data_type::INTEGRAL,
                                            data_type::FLOAT,
                                            data_type::DECIMAL,
                                            data_type::TIMESTAMP,
                                            data_type::DURATION,
                                            data_type::STRING,
                                            data_type::LIST,
                                            data_type::STRUCT>;

NVBENCH_BENCH_TYPES(BM_parquet_multithreaded_read, NVBENCH_TYPE_AXES(d_type_list))
  .set_name("parquet_multithreaded_read_decode")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("num_cols", {872})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("chunk_size", {128ul * 1024 * 1024})
  .add_int64_axis("data_size", {5ul * 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {4});
