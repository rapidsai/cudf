/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

constexpr cudf::size_type num_cols = 64;

void parquet_read_common(cudf::size_type num_rows_to_read,
                         cudf::size_type num_cols_to_read,
                         cuio_source_sink_pair& source_sink,
                         nvbench::state& state)
{
  auto const data_size = static_cast<size_t>(state.get_int64("data_size"));
  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      try_drop_l3_cache();

      timer.start();
      auto const result = cudf::io::read_parquet(read_opts);
      timer.stop();

      CUDF_EXPECTS(result.tbl->num_columns() == num_cols_to_read, "Unexpected number of columns");
      CUDF_EXPECTS(result.tbl->num_rows() == num_rows_to_read, "Unexpected number of rows");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
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
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));
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

void BM_parquet_read_long_strings(nvbench::state& state)
{
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));

  auto const d_type      = get_type_or_group(static_cast<int32_t>(data_type::STRING));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const compression = cudf::io::compression_type::SNAPPY;
  cuio_source_sink_pair source_sink(source_type);

  auto const avg_string_length = static_cast<cudf::size_type>(state.get_int64("avg_string_length"));
  // corresponds to 3 sigma (full width 6 sigma: 99.7% of range)
  auto const half_width =
    avg_string_length >> 3;  // 32 +/- 4, 128 +/- 16, 1024 +/- 128, 8k +/- 1k, etc.
  auto const length_min = avg_string_length - half_width;
  auto const length_max = avg_string_length + half_width;

  data_profile profile =
    data_profile_builder()
      .cardinality(cardinality)
      .avg_run_length(1)
      .distribution(data_type::STRING, distribution_id::NORMAL, length_min, length_max);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, profile);  // THIS
    auto const view = tbl->view();

    // set smaller threshold to reduce file size and execution time
    auto const threshold = 1;
    setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", std::to_string(threshold).c_str(), 1);

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(compression);
    cudf::io::write_parquet(write_opts);
    return view.num_rows();
  }();

  parquet_read_common(num_rows_written, num_cols, source_sink, state);
  unsetenv("LIBCUDF_LARGE_STRINGS_THRESHOLD");
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
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));
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
  auto const d_type           = get_type_or_group(static_cast<int32_t>(DataType));
  auto const cardinality      = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length       = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const chunk_read_limit = static_cast<cudf::size_type>(state.get_int64("chunk_read_limit"));
  auto const source_type      = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size        = static_cast<size_t>(state.get_int64("data_size"));
  auto const compression      = cudf::io::compression_type::SNAPPY;
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
      auto reader                   = cudf::io::chunked_parquet_reader(chunk_read_limit, read_opts);
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
void BM_parquet_read_subrowgroup_chunks(nvbench::state& state,
                                        nvbench::type_list<nvbench::enum_type<DataType>>)
{
  auto const d_type           = get_type_or_group(static_cast<int32_t>(DataType));
  auto const cardinality      = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length       = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const chunk_read_limit = static_cast<cudf::size_type>(state.get_int64("chunk_read_limit"));
  auto const pass_read_limit  = static_cast<cudf::size_type>(state.get_int64("pass_read_limit"));
  auto const source_type      = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size        = static_cast<size_t>(state.get_int64("data_size"));
  auto const compression      = cudf::io::compression_type::SNAPPY;
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
      auto reader = cudf::io::chunked_parquet_reader(chunk_read_limit, pass_read_limit, read_opts);
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
  auto const data_size_bytes = static_cast<size_t>(state.get_int64("data_size"));
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

  parquet_read_common(num_rows_written, n_col, source_sink, state);
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
  auto const data_size_bytes = static_cast<size_t>(state.get_int64("data_size"));
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

  parquet_read_common(num_rows_written, n_col, source_sink, state);
}

void BM_parquet_read_file_shape(nvbench::state& state)
{
  // Currently the parquet reader only reads the page index if there are string columns
  auto constexpr d_type = cudf::type_id::STRING;

  auto const source_type    = retrieve_io_type_enum(state.get_string("io_type"));
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_row_groups = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const num_pages_per_row_group =
    static_cast<cudf::size_type>(state.get_int64("pages_per_row_group"));
  auto const has_page_idx = static_cast<bool>(state.get_int64("has_page_idx"));

  cuio_source_sink_pair source_sink(source_type);

  auto const tbl =
    create_random_table({d_type},
                        row_count{num_rows},
                        data_profile_builder().cardinality(num_rows / 10).avg_run_length(4));
  auto const view = tbl->view();

  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .compression(cudf::io::compression_type::NONE)
      .row_group_size_rows(num_rows / num_row_groups)
      .max_page_size_rows(num_rows / (num_row_groups * num_pages_per_row_group))
      // Write page index by setting stats_level to STATISTICS_COLUMN
      .stats_level(has_page_idx ? cudf::io::statistics_freq::STATISTICS_COLUMN
                                : cudf::io::statistics_freq::STATISTICS_ROWGROUP);
  cudf::io::write_parquet(write_opts);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               try_drop_l3_cache();

               timer.start();
               auto const result = cudf::io::read_parquet(read_opts);
               timer.stop();

               CUDF_EXPECTS(result.tbl->num_columns() == 1, "Unexpected number of columns");
               CUDF_EXPECTS(result.tbl->num_rows() == num_rows, "Unexpected number of rows");
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_rows) / time, "rows_per_sec");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
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
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("data_size", {512 << 20});

NVBENCH_BENCH(BM_parquet_read_io_compression)
  .set_name("parquet_read_io_compression")
  .add_string_axis("io_type", {"FILEPATH", "HOST_BUFFER", "DEVICE_BUFFER"})
  .add_string_axis("compression_type", {"SNAPPY", "ZSTD", "NONE"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("data_size", {512 << 20});

NVBENCH_BENCH_TYPES(BM_parquet_read_chunks, NVBENCH_TYPE_AXES(d_type_list))
  .set_name("parquet_read_chunks")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("chunk_read_limit", {0, 500'000})
  .add_int64_axis("data_size", {512 << 20});

NVBENCH_BENCH_TYPES(BM_parquet_read_subrowgroup_chunks, NVBENCH_TYPE_AXES(d_type_list))
  .set_name("parquet_read_subrowgroup_chunks")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("chunk_read_limit", {0, 500'000})
  .add_int64_axis("pass_read_limit", {0, 500'000})
  .add_int64_axis("data_size", {512 << 20});

NVBENCH_BENCH(BM_parquet_read_io_small_mixed)
  .set_name("parquet_read_io_small_mixed")
  .add_string_axis("io_type", {"FILEPATH"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("num_string_cols", {1, 2, 3})
  .add_int64_axis("data_size", {512 << 20});

using d_type_list_wide_table = nvbench::enum_type_list<data_type::DECIMAL, data_type::STRING>;
NVBENCH_BENCH_TYPES(BM_parquet_read_wide_tables, NVBENCH_TYPE_AXES(d_type_list_wide_table))
  .set_name("parquet_read_wide_tables")
  .set_min_samples(4)
  .set_type_axes_names({"data_type"})
  .add_int64_axis("data_size", {1024L << 20, 2048L << 20})
  .add_int64_axis("num_cols", {256, 512, 1024})
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});

NVBENCH_BENCH(BM_parquet_read_wide_tables_mixed)
  .set_name("parquet_read_wide_tables_mixed")
  .set_min_samples(4)
  .add_int64_axis("data_size", {1024L << 20, 2048L << 20})
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
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("data_size", {512 << 20});

NVBENCH_BENCH(BM_parquet_read_long_strings)
  .set_name("parquet_read_long_strings")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("data_size", {512 << 20})
  .add_int64_power_of_two_axis("avg_string_length",
                               nvbench::range(4, 16, 2));  // 16, 64, ... -> 64k

NVBENCH_BENCH(BM_parquet_read_file_shape)
  .set_name("parquet_read_file_shape")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("num_rows", {10'000'000, 100'000'000})
  .add_int64_axis("num_row_groups", {1, 10})
  .add_int64_axis("pages_per_row_group", {1'000, 10'000})
  .add_int64_axis("has_page_idx", {true, false});
