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
                                                    data_type::BOOL8,
                                                    data_type::DECIMAL,
                                                    data_type::TIMESTAMP,
                                                    data_type::DURATION,
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
  .add_int64_axis("data_size", {512 << 20})
  .add_int64_axis("num_cols", {256, 512, 1024, 2048});
