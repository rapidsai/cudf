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

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

// Size of the data in the benchmark dataframe; chosen to be low enough to allow benchmarks to
// run on most GPUs, but large enough to allow highest throughput
constexpr size_t data_size         = 512 << 20;
constexpr cudf::size_type num_cols = 64;

void parquet_read_common(cudf::size_type num_rows_to_read,
                         cudf::size_type num_cols_to_read,
                         cuio_source_sink_pair& source_sink,
                         nvbench::state& state,
                         size_t table_data_size = data_size)
{
  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      // Comment out to cache file read
      // try_drop_l3_cache();

      timer.start();
      auto const result = cudf::io::read_parquet(read_opts);
      timer.stop();

      CUDF_EXPECTS(result.tbl->num_columns() == num_cols_to_read, "Unexpected number of columns");
      CUDF_EXPECTS(result.tbl->num_rows() == num_rows_to_read, "Unexpected number of rows");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(table_data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

template <data_type DataType>
void BM_parquet_read_data_common(nvbench::state& state,
                                 data_profile const& profile,
                                 nvbench::type_list<nvbench::enum_type<DataType>>)
{
  auto const d_type      = get_type_or_group(static_cast<int32_t>(DataType));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const compression = cudf::io::compression_type::SNAPPY;
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl =
      create_random_table(cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, profile);
    auto const view = tbl->view();

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(compression);
    cudf::io::write_parquet(write_opts);
    return view.num_rows();
  }();

  parquet_read_common(num_rows_written, num_cols, source_sink, state);
}

template <data_type DataType>
void BM_parquet_read_data(nvbench::state& state,
                          nvbench::type_list<nvbench::enum_type<DataType>> type_list)
{
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length  = static_cast<cudf::size_type>(state.get_int64("run_length"));
  BM_parquet_read_data_common<DataType>(
    state, data_profile_builder().cardinality(cardinality).avg_run_length(run_length), type_list);
}

template <data_type DataType>
void BM_parquet_read_fixed_width_struct(nvbench::state& state,
                                        nvbench::type_list<nvbench::enum_type<DataType>> type_list)
{
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length  = static_cast<cudf::size_type>(state.get_int64("run_length"));
  std::vector<cudf::type_id> s_types{
    cudf::type_id::INT32, cudf::type_id::FLOAT32, cudf::type_id::INT64};
  BM_parquet_read_data_common<DataType>(state,
                                        data_profile_builder()
                                          .cardinality(cardinality)
                                          .avg_run_length(run_length)
                                          .struct_types(s_types),
                                        type_list);
}

void BM_parquet_read_io_compression(nvbench::state& state)
{
  auto const d_type = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                         static_cast<int32_t>(data_type::FLOAT),
                                         static_cast<int32_t>(data_type::BOOL8),
                                         static_cast<int32_t>(data_type::DECIMAL),
                                         static_cast<int32_t>(data_type::TIMESTAMP),
                                         static_cast<int32_t>(data_type::DURATION),
                                         static_cast<int32_t>(data_type::STRING),
                                         static_cast<int32_t>(data_type::LIST),
                                         static_cast<int32_t>(data_type::STRUCT)});

  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length  = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const compression = retrieve_compression_type_enum(state.get_string("compression_type"));
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, num_cols),
      table_size_bytes{data_size},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(compression);
    cudf::io::write_parquet(write_opts);
    return view.num_rows();
  }();

  parquet_read_common(num_rows_written, num_cols, source_sink, state);
}

void BM_parquet_read_io_small_mixed(nvbench::state& state)
{
  auto const d_type =
    std::pair<cudf::type_id, cudf::type_id>{cudf::type_id::STRING, cudf::type_id::INT32};

  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length  = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const num_strings = static_cast<cudf::size_type>(state.get_int64("num_string_cols"));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  cuio_source_sink_pair source_sink(source_type);

  // want 80 pages total, across 4 columns, so 20 pages per column
  cudf::size_type constexpr n_col          = 4;
  cudf::size_type constexpr page_size_rows = 10'000;
  cudf::size_type constexpr num_rows       = page_size_rows * (80 / n_col);

  {
    auto const tbl = create_random_table(
      mix_dtypes(d_type, n_col, num_strings),
      row_count{num_rows},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .max_page_size_rows(10'000)
        .compression(cudf::io::compression_type::NONE);
    cudf::io::write_parquet(write_opts);
  }

  parquet_read_common(num_rows, n_col, source_sink, state);
}

template <data_type DataType>
void BM_parquet_read_chunks(nvbench::state& state, nvbench::type_list<nvbench::enum_type<DataType>>)
{
  auto const d_type      = get_type_or_group(static_cast<int32_t>(DataType));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length  = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const byte_limit  = static_cast<cudf::size_type>(state.get_int64("byte_limit"));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const compression = cudf::io::compression_type::SNAPPY;
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, num_cols),
      table_size_bytes{data_size},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(compression);

    cudf::io::write_parquet(write_opts);
    return view.num_rows();
  }();

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      try_drop_l3_cache();

      timer.start();
      auto reader                   = cudf::io::chunked_parquet_reader(byte_limit, read_opts);
      cudf::size_type num_rows_read = 0;
      do {
        auto const result = reader.read_chunk();
        num_rows_read += result.tbl->num_rows();
        CUDF_EXPECTS(result.tbl->num_columns() == num_cols, "Unexpected number of columns");
      } while (reader.has_next());
      timer.stop();

      CUDF_EXPECTS(num_rows_read == num_rows_written, "Benchmark did not read the entire table");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

template <data_type DataType>
void BM_parquet_read_wide_tables(nvbench::state& state,
                                 nvbench::type_list<nvbench::enum_type<DataType>> type_list)
{
  auto const d_type = get_type_or_group(static_cast<int32_t>(DataType));

  auto const n_col           = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const data_size_bytes = static_cast<size_t>(state.get_int64("data_size_mb") << 20);
  auto const cardinality     = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length      = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const source_type     = io_type::DEVICE_BUFFER;
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, n_col),
      table_size_bytes{data_size_bytes},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(cudf::io::compression_type::NONE);
    cudf::io::write_parquet(write_opts);
    return view.num_rows();
  }();

  parquet_read_common(num_rows_written, n_col, source_sink, state, data_size_bytes);
}

void BM_parquet_read_wide_tables_mixed(nvbench::state& state)
{
  auto const d_type = []() {
    auto d_type1 = get_type_or_group(static_cast<int32_t>(data_type::INTEGRAL));
    auto d_type2 = get_type_or_group(static_cast<int32_t>(data_type::FLOAT));
    d_type1.reserve(d_type1.size() + d_type2.size());
    std::move(d_type2.begin(), d_type2.end(), std::back_inserter(d_type1));
    return d_type1;
  }();

  auto const n_col           = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const data_size_bytes = static_cast<size_t>(state.get_int64("data_size_mb") << 20);
  auto const cardinality     = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length      = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const source_type     = io_type::DEVICE_BUFFER;
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, n_col),
      table_size_bytes{data_size_bytes},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(cudf::io::compression_type::NONE);
    cudf::io::write_parquet(write_opts);
    return view.num_rows();
  }();

  parquet_read_common(num_rows_written, n_col, source_sink, state, data_size_bytes);
}

using d_type_list = nvbench::enum_type_list<data_type::INTEGRAL,
                                            data_type::FLOAT,
                                            data_type::BOOL8,
                                            data_type::DECIMAL,
                                            data_type::TIMESTAMP,
                                            data_type::DURATION,
                                            data_type::STRING,
                                            data_type::LIST,
                                            data_type::STRUCT>;

NVBENCH_BENCH_TYPES(BM_parquet_read_data, NVBENCH_TYPE_AXES(d_type_list))
  .set_name("parquet_read_decode")
  .set_type_axes_names({"data_type"})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});

NVBENCH_BENCH(BM_parquet_read_io_compression)
  .set_name("parquet_read_io_compression")
  .add_string_axis("io_type", {"FILEPATH", "HOST_BUFFER", "DEVICE_BUFFER"})
  .add_string_axis("compression_type", {"SNAPPY", "NONE"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});

NVBENCH_BENCH_TYPES(BM_parquet_read_chunks, NVBENCH_TYPE_AXES(d_type_list))
  .set_name("parquet_read_chunks")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("byte_limit", {0, 500'000});

NVBENCH_BENCH(BM_parquet_read_io_small_mixed)
  .set_name("parquet_read_io_small_mixed")
  .add_string_axis("io_type", {"FILEPATH"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("num_string_cols", {1, 2, 3});

using d_type_list_wide_table = nvbench::enum_type_list<data_type::DECIMAL, data_type::STRING>;
NVBENCH_BENCH_TYPES(BM_parquet_read_wide_tables, NVBENCH_TYPE_AXES(d_type_list_wide_table))
  .set_name("parquet_read_wide_tables")
  .set_min_samples(4)
  .set_type_axes_names({"data_type"})
  .add_int64_axis("data_size_mb", {1024, 2048, 4096})
  .add_int64_axis("num_cols", {256, 512, 1024})
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});

NVBENCH_BENCH(BM_parquet_read_wide_tables_mixed)
  .set_name("parquet_read_wide_tables_mixed")
  .set_min_samples(4)
  .add_int64_axis("data_size_mb", {1024, 2048, 4096})
  .add_int64_axis("num_cols", {256, 512, 1024})
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});

// a benchmark for structs that only contain fixed-width types
using d_type_list_struct_only = nvbench::enum_type_list<data_type::STRUCT>;
NVBENCH_BENCH_TYPES(BM_parquet_read_fixed_width_struct, NVBENCH_TYPE_AXES(d_type_list_struct_only))
  .set_name("parquet_read_fixed_width_struct")
  .set_type_axes_names({"data_type"})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});
