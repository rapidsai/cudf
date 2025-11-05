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
  auto const source_type = io_type::DEVICE_BUFFER;
  auto const data_size   = static_cast<size_t>(512 << 20);
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
  // Cardinality and run_length don't affect table format read performance
  BM_table_read_data_common<DataType>(state, data_profile{}, type_list);
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
  auto const source_type     = io_type::DEVICE_BUFFER;
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    // Cardinality and run_length don't affect table format read performance
    auto const tbl =
      create_random_table(cycle_dtypes(d_type, n_col), table_size_bytes{data_size_bytes});
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
  // Cardinality and run_length don't affect table format read performance
  std::vector<cudf::type_id> s_types{
    cudf::type_id::INT32, cudf::type_id::FLOAT32, cudf::type_id::INT64};
  BM_table_read_data_common<DataType>(
    state, data_profile_builder().struct_types(s_types), type_list);
}

// ============================================================================
// String Variation Benchmarks
// ============================================================================

void BM_table_read_string_variations(nvbench::state& state)
{
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

  data_profile profile = data_profile_builder().avg_run_length(1).distribution(
    data_type::STRING, distribution_id::NORMAL, length_min, length_max);

  auto const num_rows_written = [&]() {
    auto const tbl =
      create_random_table(cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, profile);
    auto const view = tbl->view();

    cudf::io::write_table(view, source_sink.make_sink_info());
    return view.num_rows();
  }();

  table_read_common(num_rows_written, num_cols, source_sink, state);
}

// ============================================================================
// Benchmark Registrations
// ============================================================================

NVBENCH_BENCH(BM_table_read_io_paths)
  .set_name("table_read_io_paths")
  .add_string_axis("io_type", {"FILEPATH", "HOST_BUFFER", "DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("data_size", {512LL << 20, 1024LL << 20});

NVBENCH_BENCH(BM_table_read_size_scaling)
  .set_name("table_read_size_scaling")
  .set_min_samples(4)
  .add_int64_axis("data_size", {256LL << 20, 512LL << 20, 1024LL << 20, 2048LL << 20});

NVBENCH_BENCH(BM_table_read_column_complexity)
  .set_name("table_read_column_complexity")
  .set_min_samples(4)
  .add_int64_axis("num_cols", {32, 128, 512, 1024})
  .add_int64_axis("data_size", {512 << 20});  // Fixed size for column complexity test

using d_type_list_reduced = nvbench::enum_type_list<data_type::INTEGRAL,
                                                    data_type::FLOAT,
                                                    data_type::STRING,
                                                    data_type::LIST,
                                                    data_type::STRUCT>;
NVBENCH_BENCH_TYPES(BM_table_read_data, NVBENCH_TYPE_AXES(d_type_list_reduced))
  .set_name("table_read_decode")
  .set_type_axes_names({"data_type"})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("data_size", {512 << 20});

NVBENCH_BENCH_TYPES(BM_table_read_wide_tables,
                    NVBENCH_TYPE_AXES(nvbench::enum_type_list<data_type::STRING>))
  .set_name("table_read_wide_tables")
  .set_type_axes_names({"data_type"})
  .set_min_samples(4)
  .add_int64_axis("data_size", {1024 << 20})
  .add_int64_axis("num_cols", {256, 512, 1024});

using d_type_list_struct_only = nvbench::enum_type_list<data_type::STRUCT>;
NVBENCH_BENCH_TYPES(BM_table_read_fixed_width_struct, NVBENCH_TYPE_AXES(d_type_list_struct_only))
  .set_name("table_read_fixed_width_struct")
  .set_type_axes_names({"data_type"})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("data_size", {512 << 20});

NVBENCH_BENCH(BM_table_read_string_variations)
  .set_name("table_read_string_variations")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("data_size", {512 << 20})
  .add_int64_power_of_two_axis("avg_string_length", nvbench::range(5, 13, 2));
