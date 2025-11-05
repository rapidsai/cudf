/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/table_format.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

constexpr cudf::size_type num_cols = 64;

void table_read_common(cudf::size_type num_rows_to_read,
                       cudf::size_type num_cols_to_read,
                       cuio_source_sink_pair& source_sink,
                       nvbench::state& state)
{
  auto const data_size = static_cast<size_t>(state.get_int64("data_size"));

  // Create source_info outside the lambda to ensure it stays alive for the entire benchmark
  // This matches the pattern used in other IO benchmarks (parquet, orc, csv)
  auto source_info = source_sink.make_source_info();
  // For DEVICE_BUFFER, make_source_info does async copy, so synchronize to ensure it completes
  if (source_info.type() == cudf::io::io_type::DEVICE_BUFFER) {
    cudf::get_default_stream().synchronize();
  }

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      try_drop_l3_cache();

      timer.start();
      auto result = cudf::io::read_table(source_info);
      // read_table already synchronizes internally, so result is ready
      timer.stop();

      // Access result.table while result is still in scope
      // result owns the packed_columns which contains the memory that table_view points to
      CUDF_EXPECTS(result.table.num_columns() == num_cols_to_read, "Unexpected number of columns");
      CUDF_EXPECTS(result.table.num_rows() == num_rows_to_read, "Unexpected number of rows");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

// ============================================================================
// IO Path Comparison Benchmarks
// ============================================================================

void BM_table_read_io_paths(nvbench::state& state)
{
  auto const d_type      = get_type_or_group(static_cast<int32_t>(data_type::INTEGRAL));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl =
      create_random_table(cycle_dtypes(d_type, num_cols), table_size_bytes{data_size});
    auto const view = tbl->view();

    cudf::io::write_table(view, source_sink.make_sink_info());
    return view.num_rows();
  }();

  table_read_common(num_rows_written, num_cols, source_sink, state);
}

// ============================================================================
// Data Size Scaling Benchmarks
// ============================================================================

void BM_table_read_size_scaling(nvbench::state& state)
{
  auto const d_type      = get_type_or_group(static_cast<int32_t>(data_type::INTEGRAL));
  auto const source_type = io_type::DEVICE_BUFFER;  // Use DEVICE_BUFFER to isolate unpack overhead
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl =
      create_random_table(cycle_dtypes(d_type, num_cols), table_size_bytes{data_size});
    auto const view = tbl->view();

    cudf::io::write_table(view, source_sink.make_sink_info());
    return view.num_rows();
  }();

  table_read_common(num_rows_written, num_cols, source_sink, state);
}

// ============================================================================
// Column Complexity Benchmarks
// ============================================================================

void BM_table_read_column_complexity(nvbench::state& state)
{
  auto const d_type      = get_type_or_group(static_cast<int32_t>(data_type::INTEGRAL));
  auto const n_col       = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const source_type = io_type::DEVICE_BUFFER;          // Focus on unpack overhead
  auto const data_size   = static_cast<size_t>(512 << 20);  // Fixed 512MB
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl  = create_random_table(cycle_dtypes(d_type, n_col), table_size_bytes{data_size});
    auto const view = tbl->view();

    cudf::io::write_table(view, source_sink.make_sink_info());
    return view.num_rows();
  }();

  table_read_common(num_rows_written, n_col, source_sink, state);
}

// ============================================================================
// Basic Data Type Benchmarks
// ============================================================================

template <data_type DataType>
void BM_table_read_data_common(nvbench::state& state,
                               data_profile const& profile,
                               nvbench::type_list<nvbench::enum_type<DataType>>)
{
  auto const d_type      = get_type_or_group(static_cast<int32_t>(DataType));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl =
      create_random_table(cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, profile);
    auto const view = tbl->view();

    cudf::io::write_table(view, source_sink.make_sink_info());
    return view.num_rows();
  }();

  table_read_common(num_rows_written, num_cols, source_sink, state);
}

template <data_type DataType>
void BM_table_read_data(nvbench::state& state,
                        nvbench::type_list<nvbench::enum_type<DataType>> type_list)
{
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length  = static_cast<cudf::size_type>(state.get_int64("run_length"));
  BM_table_read_data_common<DataType>(
    state, data_profile_builder().cardinality(cardinality).avg_run_length(run_length), type_list);
}

// ============================================================================
// Wide Tables Benchmarks (focusing on metadata parsing overhead)
// ============================================================================

template <data_type DataType>
void BM_table_read_wide_tables(nvbench::state& state,
                               nvbench::type_list<nvbench::enum_type<DataType>> type_list)
{
  auto const d_type = get_type_or_group(static_cast<int32_t>(DataType));

  auto const n_col           = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const data_size_bytes = static_cast<size_t>(state.get_int64("data_size"));
  auto const cardinality     = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length      = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const source_type     = io_type::DEVICE_BUFFER;  // Focus on unpack overhead
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, n_col),
      table_size_bytes{data_size_bytes},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::write_table(view, source_sink.make_sink_info());
    return view.num_rows();
  }();

  table_read_common(num_rows_written, n_col, source_sink, state);
}

void BM_table_read_wide_tables_mixed(nvbench::state& state)
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
  auto const source_type     = io_type::DEVICE_BUFFER;  // Focus on unpack overhead
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, n_col),
      table_size_bytes{data_size_bytes},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::write_table(view, source_sink.make_sink_info());
    return view.num_rows();
  }();

  table_read_common(num_rows_written, n_col, source_sink, state);
}

// ============================================================================
// Nested Types Benchmarks
// ============================================================================

template <data_type DataType>
void BM_table_read_fixed_width_struct(nvbench::state& state,
                                      nvbench::type_list<nvbench::enum_type<DataType>> type_list)
{
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length  = static_cast<cudf::size_type>(state.get_int64("run_length"));
  std::vector<cudf::type_id> s_types{
    cudf::type_id::INT32, cudf::type_id::FLOAT32, cudf::type_id::INT64};
  BM_table_read_data_common<DataType>(state,
                                      data_profile_builder()
                                        .cardinality(cardinality)
                                        .avg_run_length(run_length)
                                        .struct_types(s_types),
                                      type_list);
}

void BM_table_read_nested_struct_depth(nvbench::state& state)
{
  auto const cardinality  = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length   = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const struct_depth = static_cast<cudf::size_type>(state.get_int64("struct_depth"));
  auto const source_type  = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size    = static_cast<size_t>(state.get_int64("data_size"));
  cuio_source_sink_pair source_sink(source_type);

  std::vector<cudf::type_id> s_types{
    cudf::type_id::INT32, cudf::type_id::FLOAT32, cudf::type_id::INT64};

  auto const num_rows_written = [&]() {
    auto const tbl  = create_random_table(cycle_dtypes({cudf::type_id::STRUCT}, num_cols),
                                         table_size_bytes{data_size},
                                         data_profile_builder()
                                           .cardinality(cardinality)
                                           .avg_run_length(run_length)
                                           .struct_types(s_types)
                                           .struct_depth(struct_depth));
    auto const view = tbl->view();

    cudf::io::write_table(view, source_sink.make_sink_info());
    return view.num_rows();
  }();

  table_read_common(num_rows_written, num_cols, source_sink, state);
}

void BM_table_read_nested_list_depth(nvbench::state& state)
{
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const list_depth  = static_cast<cudf::size_type>(state.get_int64("list_depth"));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl =
      create_random_table(cycle_dtypes({cudf::type_id::LIST}, num_cols),
                          table_size_bytes{data_size},
                          data_profile_builder().cardinality(cardinality).list_depth(list_depth));
    auto const view = tbl->view();

    cudf::io::write_table(view, source_sink.make_sink_info());
    return view.num_rows();
  }();

  table_read_common(num_rows_written, num_cols, source_sink, state);
}

// ============================================================================
// Null Density Benchmarks
// ============================================================================

void BM_table_read_null_density(nvbench::state& state)
{
  auto const null_prob   = state.get_float64("null_probability");
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));
  cuio_source_sink_pair source_sink(source_type);

  auto const d_type = get_type_or_group(static_cast<int32_t>(data_type::INTEGRAL));

  auto const num_rows_written = [&]() {
    auto const tbl  = create_random_table(cycle_dtypes(d_type, num_cols),
                                         table_size_bytes{data_size},
                                         data_profile_builder().null_probability(null_prob));
    auto const view = tbl->view();

    cudf::io::write_table(view, source_sink.make_sink_info());
    return view.num_rows();
  }();

  table_read_common(num_rows_written, num_cols, source_sink, state);
}

// ============================================================================
// Small Tables Benchmarks (testing fixed overhead)
// ============================================================================

void BM_table_read_small_tables(nvbench::state& state)
{
  auto const d_type      = get_type_or_group(static_cast<int32_t>(data_type::INTEGRAL));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl =
      create_random_table(cycle_dtypes(d_type, num_cols), table_size_bytes{data_size});
    auto const view = tbl->view();

    cudf::io::write_table(view, source_sink.make_sink_info());
    return view.num_rows();
  }();

  table_read_common(num_rows_written, num_cols, source_sink, state);
}

// ============================================================================
// String Variation Benchmarks
// ============================================================================

void BM_table_read_string_variations(nvbench::state& state)
{
  auto const cardinality       = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const data_size         = static_cast<size_t>(state.get_int64("data_size"));
  auto const avg_string_length = static_cast<cudf::size_type>(state.get_int64("avg_string_length"));
  auto const source_type       = retrieve_io_type_enum(state.get_string("io_type"));
  cuio_source_sink_pair source_sink(source_type);

  auto const d_type = get_type_or_group(static_cast<int32_t>(data_type::STRING));

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
    auto const tbl =
      create_random_table(cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, profile);
    auto const view = tbl->view();

    cudf::io::write_table(view, source_sink.make_sink_info());
    return view.num_rows();
  }();

  table_read_common(num_rows_written, num_cols, source_sink, state);
}

void BM_table_read_string_column_ratio(nvbench::state& state)
{
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));
  auto const num_strings = static_cast<cudf::size_type>(state.get_int64("num_string_cols"));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  cuio_source_sink_pair source_sink(source_type);

  auto const d_type =
    std::pair<cudf::type_id, cudf::type_id>{cudf::type_id::STRING, cudf::type_id::INT32};

  cudf::size_type constexpr n_col = 8;

  auto const num_rows_written = [&]() {
    auto const tbl  = create_random_table(mix_dtypes(d_type, n_col, num_strings),
                                         table_size_bytes{data_size},
                                         data_profile_builder().cardinality(cardinality));
    auto const view = tbl->view();

    cudf::io::write_table(view, source_sink.make_sink_info());
    return view.num_rows();
  }();

  table_read_common(num_rows_written, n_col, source_sink, state);
}

// ============================================================================
// Benchmark Registrations
// ============================================================================

using d_type_list = nvbench::enum_type_list<data_type::INTEGRAL,
                                            data_type::FLOAT,
                                            data_type::BOOL8,
                                            data_type::DECIMAL,
                                            data_type::TIMESTAMP,
                                            data_type::DURATION,
                                            data_type::STRING,
                                            data_type::LIST,
                                            data_type::STRUCT>;

// IO Path Comparison
NVBENCH_BENCH(BM_table_read_io_paths)
  .set_name("table_read_io_paths")
  .add_string_axis("io_type", {"FILEPATH", "HOST_BUFFER", "DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("data_size", {512 << 20, 1024 << 20, 2048 << 20});

// Data Size Scaling
NVBENCH_BENCH(BM_table_read_size_scaling)
  .set_name("table_read_size_scaling")
  .set_min_samples(4)
  .add_int64_axis("data_size",
                  {64 << 20, 256 << 20, 512 << 20, 1024 << 20, 2048 << 20, 4096 << 20});

// Column Complexity
NVBENCH_BENCH(BM_table_read_column_complexity)
  .set_name("table_read_column_complexity")
  .set_min_samples(4)
  .add_int64_axis("num_cols", {8, 32, 64, 128, 256, 512, 1024});

// Basic Data Types
NVBENCH_BENCH_TYPES(BM_table_read_data, NVBENCH_TYPE_AXES(d_type_list))
  .set_name("table_read_decode")
  .set_type_axes_names({"data_type"})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("data_size", {512 << 20});

// Wide Tables
using d_type_list_wide_table = nvbench::enum_type_list<data_type::DECIMAL, data_type::STRING>;
NVBENCH_BENCH_TYPES(BM_table_read_wide_tables, NVBENCH_TYPE_AXES(d_type_list_wide_table))
  .set_name("table_read_wide_tables")
  .set_min_samples(4)
  .set_type_axes_names({"data_type"})
  .add_int64_axis("data_size", {1024 << 20, 2048 << 20})
  .add_int64_axis("num_cols", {256, 512, 1024, 2048})
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});

NVBENCH_BENCH(BM_table_read_wide_tables_mixed)
  .set_name("table_read_wide_tables_mixed")
  .set_min_samples(4)
  .add_int64_axis("data_size", {1024 << 20, 2048 << 20})
  .add_int64_axis("num_cols", {256, 512, 1024, 2048})
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32});

// Nested Types - Fixed Width Struct
using d_type_list_struct_only = nvbench::enum_type_list<data_type::STRUCT>;
NVBENCH_BENCH_TYPES(BM_table_read_fixed_width_struct, NVBENCH_TYPE_AXES(d_type_list_struct_only))
  .set_name("table_read_fixed_width_struct")
  .set_type_axes_names({"data_type"})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("data_size", {512 << 20});

// Nested Types - Struct Depth
NVBENCH_BENCH(BM_table_read_nested_struct_depth)
  .set_name("table_read_nested_struct_depth")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("struct_depth", {1, 2, 3})
  .add_int64_axis("data_size", {512 << 20});

// Nested Types - List Depth
NVBENCH_BENCH(BM_table_read_nested_list_depth)
  .set_name("table_read_nested_list_depth")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("list_depth", {1, 2, 3})
  .add_int64_axis("data_size", {512 << 20});

// Null Density
NVBENCH_BENCH(BM_table_read_null_density)
  .set_name("table_read_null_density")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_float64_axis("null_probability", {0.0, 0.1, 0.5, 0.9})
  .add_int64_axis("data_size", {512 << 20});

// Small Tables
NVBENCH_BENCH(BM_table_read_small_tables)
  .set_name("table_read_small_tables")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("data_size", {1 << 10, 10 << 10, 100 << 10, 1 << 20});  // 1KB, 10KB, 100KB, 1MB

// String Variations
NVBENCH_BENCH(BM_table_read_string_variations)
  .set_name("table_read_string_variations")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("data_size", {512 << 20})
  .add_int64_power_of_two_axis("avg_string_length",
                               nvbench::range(4, 16, 2));  // 16, 64, ... -> 64k

NVBENCH_BENCH(BM_table_read_string_column_ratio)
  .set_name("table_read_string_column_ratio")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("num_string_cols", {0, 2, 4, 6, 8})
  .add_int64_axis("data_size", {512 << 20});
