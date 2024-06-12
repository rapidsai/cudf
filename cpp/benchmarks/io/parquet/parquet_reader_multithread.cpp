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
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/pinned_memory.hpp>
#include <cudf/utilities/thread_pool.hpp>

#include <nvtx3/nvtx3.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

size_t get_num_reads(nvbench::state const& state) { return state.get_int64("num_threads"); }

size_t get_read_size(nvbench::state const& state)
{
  auto const num_reads = get_num_reads(state);
  return state.get_int64("total_data_size") / num_reads;
}

std::string get_label(std::string const& test_name, nvbench::state const& state)
{
  auto const num_cols       = state.get_int64("num_cols");
  size_t const read_size_mb = get_read_size(state) / (1024 * 1024);
  return {test_name + ", " + std::to_string(num_cols) + " columns, " +
          std::to_string(state.get_int64("num_threads")) + " threads " + " (" +
          std::to_string(read_size_mb) + " MB each)"};
}

std::tuple<std::vector<cuio_source_sink_pair>, size_t, size_t> write_file_data(
  nvbench::state& state, std::vector<cudf::type_id> const& d_types)
{
  cudf::size_type const cardinality = state.get_int64("cardinality");
  cudf::size_type const run_length  = state.get_int64("run_length");
  cudf::size_type const num_cols    = state.get_int64("num_cols");
  size_t const num_files            = get_num_reads(state);
  size_t const per_file_data_size   = get_read_size(state);

  std::vector<cuio_source_sink_pair> source_sink_vector;

  size_t total_file_size = 0;

  for (size_t i = 0; i < num_files; ++i) {
    cuio_source_sink_pair source_sink{io_type::HOST_BUFFER};

    auto const tbl = create_random_table(
      cycle_dtypes(d_types, num_cols),
      table_size_bytes{per_file_data_size},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(cudf::io::compression_type::SNAPPY)
        .max_page_size_rows(50000)
        .max_page_size_bytes(1024 * 1024);

    cudf::io::write_parquet(write_opts);
    total_file_size += source_sink.size();

    source_sink_vector.push_back(std::move(source_sink));
  }

  return {std::move(source_sink_vector), total_file_size, num_files};
}

void BM_parquet_multithreaded_read_common(nvbench::state& state,
                                          std::vector<cudf::type_id> const& d_types,
                                          std::string const& label)
{
  size_t const data_size = state.get_int64("total_data_size");
  auto const num_threads = state.get_int64("num_threads");

  auto streams = cudf::detail::fork_streams(cudf::get_default_stream(), num_threads);
  cudf::detail::thread_pool threads(num_threads);

  auto [source_sink_vector, total_file_size, num_files] = write_file_data(state, d_types);
  std::vector<cudf::io::source_info> source_info_vector;
  std::transform(source_sink_vector.begin(),
                 source_sink_vector.end(),
                 std::back_inserter(source_info_vector),
                 [](auto& source_sink) { return source_sink.make_source_info(); });

  auto mem_stats_logger = cudf::memory_stats_logger();

  nvtxRangePushA(("(read) " + label).c_str());
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               auto read_func = [&](int index) {
                 auto const stream = streams[index % num_threads];
                 cudf::io::parquet_reader_options read_opts =
                   cudf::io::parquet_reader_options::builder(source_info_vector[index]);
                 cudf::io::read_parquet(read_opts, stream, rmm::mr::get_current_device_resource());
               };

               threads.paused = true;
               for (size_t i = 0; i < num_files; ++i) {
                 threads.submit(read_func, i);
               }
               timer.start();
               threads.paused = false;
               threads.wait_for_tasks();
               cudf::detail::join_streams(streams, cudf::get_default_stream());
               timer.stop();
             });
  nvtxRangePop();

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(total_file_size, "encoded_file_size", "encoded_file_size");
}

void BM_parquet_multithreaded_read_mixed(nvbench::state& state)
{
  auto label = get_label("mixed", state);
  nvtxRangePushA(label.c_str());
  BM_parquet_multithreaded_read_common(
    state, {cudf::type_id::INT32, cudf::type_id::DECIMAL64, cudf::type_id::STRING}, label);
  nvtxRangePop();
}

void BM_parquet_multithreaded_read_fixed_width(nvbench::state& state)
{
  auto label = get_label("fixed width", state);
  nvtxRangePushA(label.c_str());
  BM_parquet_multithreaded_read_common(state, {cudf::type_id::INT32}, label);
  nvtxRangePop();
}

void BM_parquet_multithreaded_read_string(nvbench::state& state)
{
  auto label = get_label("string", state);
  nvtxRangePushA(label.c_str());
  BM_parquet_multithreaded_read_common(state, {cudf::type_id::STRING}, label);
  nvtxRangePop();
}

void BM_parquet_multithreaded_read_list(nvbench::state& state)
{
  auto label = get_label("list", state);
  nvtxRangePushA(label.c_str());
  BM_parquet_multithreaded_read_common(state, {cudf::type_id::LIST}, label);
  nvtxRangePop();
}

void BM_parquet_multithreaded_read_chunked_common(nvbench::state& state,
                                                  std::vector<cudf::type_id> const& d_types,
                                                  std::string const& label)
{
  size_t const data_size    = state.get_int64("total_data_size");
  auto const num_threads    = state.get_int64("num_threads");
  size_t const input_limit  = state.get_int64("input_limit");
  size_t const output_limit = state.get_int64("output_limit");

  auto streams = cudf::detail::fork_streams(cudf::get_default_stream(), num_threads);
  cudf::detail::thread_pool threads(num_threads);
  auto [source_sink_vector, total_file_size, num_files] = write_file_data(state, d_types);
  std::vector<cudf::io::source_info> source_info_vector;
  std::transform(source_sink_vector.begin(),
                 source_sink_vector.end(),
                 std::back_inserter(source_info_vector),
                 [](auto& source_sink) { return source_sink.make_source_info(); });

  auto mem_stats_logger = cudf::memory_stats_logger();

  nvtxRangePushA(("(read) " + label).c_str());
  std::vector<cudf::io::table_with_metadata> chunks;
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               auto read_func = [&](int index) {
                 auto const stream = streams[index % num_threads];
                 cudf::io::parquet_reader_options read_opts =
                   cudf::io::parquet_reader_options::builder(source_info_vector[index]);
                 // divide chunk limits by number of threads so the number of chunks produced is the
                 // same for all cases. this seems better than the alternative, which is to keep the
                 // limits the same. if we do that, as the number of threads goes up, the number of
                 // chunks goes down - so are actually benchmarking the same thing in that case?
                 auto reader = cudf::io::chunked_parquet_reader(
                   output_limit / num_threads, input_limit / num_threads, read_opts, stream);

                 // read all the chunks
                 do {
                   auto table = reader.read_chunk();
                 } while (reader.has_next());
               };

               threads.paused = true;
               for (size_t i = 0; i < num_files; ++i) {
                 threads.submit(read_func, i);
               }
               timer.start();
               threads.paused = false;
               threads.wait_for_tasks();
               cudf::detail::join_streams(streams, cudf::get_default_stream());
               timer.stop();
             });
  nvtxRangePop();

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(total_file_size, "encoded_file_size", "encoded_file_size");
}

void BM_parquet_multithreaded_read_chunked_mixed(nvbench::state& state)
{
  auto label = get_label("mixed", state);
  nvtxRangePushA(label.c_str());
  BM_parquet_multithreaded_read_chunked_common(
    state, {cudf::type_id::INT32, cudf::type_id::DECIMAL64, cudf::type_id::STRING}, label);
  nvtxRangePop();
}

void BM_parquet_multithreaded_read_chunked_fixed_width(nvbench::state& state)
{
  auto label = get_label("mixed", state);
  nvtxRangePushA(label.c_str());
  BM_parquet_multithreaded_read_chunked_common(state, {cudf::type_id::INT32}, label);
  nvtxRangePop();
}

void BM_parquet_multithreaded_read_chunked_string(nvbench::state& state)
{
  auto label = get_label("string", state);
  nvtxRangePushA(label.c_str());
  BM_parquet_multithreaded_read_chunked_common(state, {cudf::type_id::STRING}, label);
  nvtxRangePop();
}

void BM_parquet_multithreaded_read_chunked_list(nvbench::state& state)
{
  auto label = get_label("list", state);
  nvtxRangePushA(label.c_str());
  BM_parquet_multithreaded_read_chunked_common(state, {cudf::type_id::LIST}, label);
  nvtxRangePop();
}

// mixed data types: fixed width and strings
NVBENCH_BENCH(BM_parquet_multithreaded_read_mixed)
  .set_name("parquet_multithreaded_read_decode_mixed")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("total_data_size", {512 * 1024 * 1024, 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {1, 2, 4, 8})
  .add_int64_axis("num_cols", {4})
  .add_int64_axis("run_length", {8});

NVBENCH_BENCH(BM_parquet_multithreaded_read_fixed_width)
  .set_name("parquet_multithreaded_read_decode_fixed_width")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("total_data_size", {512 * 1024 * 1024, 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {1, 2, 4, 8})
  .add_int64_axis("num_cols", {4})
  .add_int64_axis("run_length", {8});

NVBENCH_BENCH(BM_parquet_multithreaded_read_string)
  .set_name("parquet_multithreaded_read_decode_string")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("total_data_size", {512 * 1024 * 1024, 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {1, 2, 4, 8})
  .add_int64_axis("num_cols", {4})
  .add_int64_axis("run_length", {8});

NVBENCH_BENCH(BM_parquet_multithreaded_read_list)
  .set_name("parquet_multithreaded_read_decode_list")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("total_data_size", {512 * 1024 * 1024, 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {1, 2, 4, 8})
  .add_int64_axis("num_cols", {4})
  .add_int64_axis("run_length", {8});

// mixed data types: fixed width, strings
NVBENCH_BENCH(BM_parquet_multithreaded_read_chunked_mixed)
  .set_name("parquet_multithreaded_read_decode_chunked_mixed")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("total_data_size", {512 * 1024 * 1024, 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {1, 2, 4, 8})
  .add_int64_axis("num_cols", {4})
  .add_int64_axis("run_length", {8})
  .add_int64_axis("input_limit", {640 * 1024 * 1024})
  .add_int64_axis("output_limit", {640 * 1024 * 1024});

NVBENCH_BENCH(BM_parquet_multithreaded_read_chunked_fixed_width)
  .set_name("parquet_multithreaded_read_decode_chunked_fixed_width")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("total_data_size", {512 * 1024 * 1024, 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {1, 2, 4, 8})
  .add_int64_axis("num_cols", {4})
  .add_int64_axis("run_length", {8})
  .add_int64_axis("input_limit", {640 * 1024 * 1024})
  .add_int64_axis("output_limit", {640 * 1024 * 1024});

NVBENCH_BENCH(BM_parquet_multithreaded_read_chunked_string)
  .set_name("parquet_multithreaded_read_decode_chunked_string")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("total_data_size", {512 * 1024 * 1024, 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {1, 2, 4, 8})
  .add_int64_axis("num_cols", {4})
  .add_int64_axis("run_length", {8})
  .add_int64_axis("input_limit", {640 * 1024 * 1024})
  .add_int64_axis("output_limit", {640 * 1024 * 1024});

NVBENCH_BENCH(BM_parquet_multithreaded_read_chunked_list)
  .set_name("parquet_multithreaded_read_decode_chunked_list")
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("total_data_size", {512 * 1024 * 1024, 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {1, 2, 4, 8})
  .add_int64_axis("num_cols", {4})
  .add_int64_axis("run_length", {8})
  .add_int64_axis("input_limit", {640 * 1024 * 1024})
  .add_int64_axis("output_limit", {640 * 1024 * 1024});
