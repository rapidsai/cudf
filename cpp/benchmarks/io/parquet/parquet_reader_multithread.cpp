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

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/io/memory_resource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/thread_pool.hpp>

#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <nvtx3/nvtx3.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

// TODO: remove this once pinned/pooled is enabled by default in cuIO
void set_cuio_host_pinned_pool()
{
  using host_pooled_mr = rmm::mr::pool_memory_resource<rmm::mr::pinned_host_memory_resource>;
  static std::shared_ptr<host_pooled_mr> mr = std::make_shared<host_pooled_mr>(
    std::make_shared<rmm::mr::pinned_host_memory_resource>().get(), 256ul * 1024 * 1024);

  cudf::io::set_host_memory_resource(*mr);
}

size_t get_num_reads(nvbench::state const& state)
{
  size_t const data_size          = state.get_int64("total_data_size");
  size_t const per_file_data_size = state.get_int64("per_file_data_size");
  return data_size / per_file_data_size;
}

std::string get_label(std::string const& test_name, nvbench::state const& state)
{
  auto const num_reads = get_num_reads(state);
  auto const num_cols  = state.get_int64("num_cols");
  return {test_name + ", " + std::to_string(num_cols) + "columns, " + std::to_string(num_reads) +
          "reads, " + std::to_string(state.get_int64("num_threads")) + " threads"};
}

std::tuple<std::vector<cuio_source_sink_pair>, size_t, size_t> write_file_data(
  nvbench::state& state, std::vector<cudf::type_id> const& d_types)
{
  cudf::size_type const cardinality = state.get_int64("cardinality");
  cudf::size_type const run_length  = state.get_int64("run_length");
  cudf::size_type const num_cols    = state.get_int64("num_cols");
  size_t const per_file_data_size   = state.get_int64("per_file_data_size");

  size_t const num_tables = get_num_reads(state);
  std::vector<cuio_source_sink_pair> source_sink_vector;

  size_t total_file_size = 0;

  for (size_t i = 0; i < num_tables; ++i) {
    cuio_source_sink_pair source_sink{cudf::io::io_type::HOST_BUFFER};

    auto const tbl = create_random_table(
      cycle_dtypes(d_types, num_cols),
      table_size_bytes{per_file_data_size},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(cudf::io::compression_type::SNAPPY);

    cudf::io::write_parquet(write_opts);
    total_file_size += source_sink.size();

    source_sink_vector.push_back(std::move(source_sink));
  }

  return {std::move(source_sink_vector), total_file_size, num_tables};
}

void BM_parquet_multithreaded_read_common(nvbench::state& state,
                                          std::vector<cudf::type_id> const& d_types)
{
  size_t const data_size = state.get_int64("total_data_size");
  auto const num_threads = state.get_int64("num_threads");

  set_cuio_host_pinned_pool();

  auto streams = cudf::detail::fork_streams(cudf::get_default_stream(), num_threads);
  cudf::detail::thread_pool threads(num_threads);

  auto [source_sink_vector, total_file_size, num_tables] = write_file_data(state, d_types);

  auto mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               auto read_func = [&](int index) {
                 auto const stream = streams[index % num_threads];
                 auto& source_sink = source_sink_vector[index];
                 cudf::io::parquet_reader_options read_opts =
                   cudf::io::parquet_reader_options::builder(source_sink.make_source_info());
                 cudf::io::read_parquet(read_opts, stream, rmm::mr::get_current_device_resource());
               };

               threads.paused = true;
               for (size_t i = 0; i < num_tables; ++i) {
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

void BM_parquet_multithreaded_read_mixed(nvbench::state& state)
{
  nvtxRangePushA(get_label("mixed", state).c_str());
  BM_parquet_multithreaded_read_common(
    state,
    {cudf::type_id::INT32, cudf::type_id::DECIMAL64, cudf::type_id::STRING, cudf::type_id::LIST});
  nvtxRangePop();
}

void BM_parquet_multithreaded_read_fixed_width(nvbench::state& state)
{
  nvtxRangePushA(get_label("fixed width", state).c_str());
  BM_parquet_multithreaded_read_common(state, {cudf::type_id::INT32});
  nvtxRangePop();
}

void BM_parquet_multithreaded_read_string(nvbench::state& state)
{
  nvtxRangePushA(get_label("string", state).c_str());
  BM_parquet_multithreaded_read_common(state, {cudf::type_id::STRING});
  nvtxRangePop();
}

void BM_parquet_multithreaded_read_list(nvbench::state& state)
{
  nvtxRangePushA(get_label("list", state).c_str());
  BM_parquet_multithreaded_read_common(state, {cudf::type_id::LIST});
  nvtxRangePop();
}

void BM_parquet_multithreaded_read_chunked_common(nvbench::state& state,
                                                  std::vector<cudf::type_id> const& d_types)
{
  size_t const data_size    = state.get_int64("total_data_size");
  size_t const input_limit  = state.get_int64("input_limit");
  size_t const output_limit = state.get_int64("output_limit");
  auto const num_threads    = state.get_int64("num_threads");

  set_cuio_host_pinned_pool();

  auto streams = cudf::detail::fork_streams(cudf::get_default_stream(), num_threads);
  cudf::detail::thread_pool threads(num_threads);

  auto [source_sink_vector, total_file_size, num_tables] = write_file_data(state, d_types);

  auto mem_stats_logger = cudf::memory_stats_logger();

  std::vector<cudf::io::table_with_metadata> chunks;
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      auto read_func = [&](int index) {
        auto const stream = streams[index % num_threads];
        auto& source_sink = source_sink_vector[index];
        cudf::io::parquet_reader_options read_opts =
          cudf::io::parquet_reader_options::builder(source_sink.make_source_info());
        auto reader = cudf::io::chunked_parquet_reader(output_limit, input_limit, read_opts);

        // read all the chunks
        do {
          auto table = reader.read_chunk();
        } while (reader.has_next());
      };

      threads.paused = true;
      for (size_t i = 0; i < num_tables; ++i) {
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

void BM_parquet_multithreaded_read_chunked_mixed(nvbench::state& state)
{
  nvtxRangePushA(get_label("mixed", state).c_str());
  BM_parquet_multithreaded_read_chunked_common(
    state, {cudf::type_id::INT32, cudf::type_id::STRING, cudf::type_id::LIST});
  nvtxRangePop();
}

void BM_parquet_multithreaded_read_chunked_string(nvbench::state& state)
{
  nvtxRangePushA(get_label("string", state).c_str());
  BM_parquet_multithreaded_read_chunked_common(state, {cudf::type_id::STRING});
  nvtxRangePop();
}

void BM_parquet_multithreaded_read_chunked_list(nvbench::state& state)
{
  nvtxRangePushA(get_label("list", state).c_str());
  BM_parquet_multithreaded_read_chunked_common(state, {cudf::type_id::LIST});
  nvtxRangePop();
}

// mixed data types, covering the 3 main families : fixed width, strings, and lists
NVBENCH_BENCH(BM_parquet_multithreaded_read_mixed)
  .set_name("parquet_multithreaded_read_decode_mixed")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("num_cols", {4, 16})
  .add_int64_axis("run_length", {8})
  .add_int64_axis("per_file_data_size", {128ul * 1024 * 1024, 512ul * 1024 * 1024})
  .add_int64_axis("total_data_size", {4ul * 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {2, 4, 8});

NVBENCH_BENCH(BM_parquet_multithreaded_read_fixed_width)
  .set_name("parquet_multithreaded_read_decode_fixed_width")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("num_cols", {4, 16})
  .add_int64_axis("run_length", {8})
  .add_int64_axis("per_file_data_size", {128ul * 1024 * 1024, 512ul * 1024 * 1024})
  .add_int64_axis("total_data_size", {4ul * 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {2, 4, 8});

NVBENCH_BENCH(BM_parquet_multithreaded_read_string)
  .set_name("parquet_multithreaded_read_decode_string")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("num_cols", {4, 16})
  .add_int64_axis("run_length", {8})
  .add_int64_axis("per_file_data_size", {128ul * 1024 * 1024, 512ul * 1024 * 1024})
  .add_int64_axis("total_data_size", {4ul * 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {2, 4, 8});

NVBENCH_BENCH(BM_parquet_multithreaded_read_list)
  .set_name("parquet_multithreaded_read_decode_list")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("num_cols", {4, 16})
  .add_int64_axis("run_length", {8})
  .add_int64_axis("per_file_data_size", {128ul * 1024 * 1024, 512ul * 1024 * 1024})
  .add_int64_axis("total_data_size", {4ul * 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {2, 4, 8});

// mixed data types, covering the 3 main families : fixed width, strings, and lists
NVBENCH_BENCH(BM_parquet_multithreaded_read_chunked_mixed)
  .set_name("parquet_multithreaded_read_decode_chunked_mixed")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("num_cols", {6})
  .add_int64_axis("run_length", {8})
  // divides into 10 GB exactly 8 times
  .add_int64_axis("per_file_data_size", {1280ul * 1024 * 1024})
  .add_int64_axis("total_data_size", {10ul * 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {2, 4, 8})
  .add_int64_axis("input_limit", {768 * 1024 * 1024})
  .add_int64_axis("output_limit", {512 * 1024 * 1024});

NVBENCH_BENCH(BM_parquet_multithreaded_read_chunked_string)
  .set_name("parquet_multithreaded_read_decode_chunked_mixed")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("num_cols", {6})
  .add_int64_axis("run_length", {8})
  // divides into 10 GB exactly 8 times
  .add_int64_axis("per_file_data_size", {1280ul * 1024 * 1024})
  .add_int64_axis("total_data_size", {10ul * 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {2, 4, 8})
  .add_int64_axis("input_limit", {768 * 1024 * 1024})
  .add_int64_axis("output_limit", {512 * 1024 * 1024});

NVBENCH_BENCH(BM_parquet_multithreaded_read_chunked_list)
  .set_name("parquet_multithreaded_read_decode_chunked_mixed")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("num_cols", {6})
  .add_int64_axis("run_length", {8})
  // divides into 10 GB exactly 8 times
  .add_int64_axis("per_file_data_size", {1280ul * 1024 * 1024})
  .add_int64_axis("total_data_size", {10ul * 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {2, 4, 8})
  .add_int64_axis("input_limit", {768 * 1024 * 1024})
  .add_int64_axis("output_limit", {512 * 1024 * 1024});
