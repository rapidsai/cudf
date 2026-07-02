/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/csv.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

// Column type profiles for the type inference benchmark
enum class type_profile : int32_t {
  ALL_INTEGRAL,
  ALL_FLOAT,
  ALL_STRING,
  MIXED,
};

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  type_profile,
  [](auto value) {
    switch (value) {
      case type_profile::ALL_INTEGRAL: return "ALL_INTEGRAL";
      case type_profile::ALL_FLOAT: return "ALL_FLOAT";
      case type_profile::ALL_STRING: return "ALL_STRING";
      case type_profile::MIXED: return "MIXED";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

// Use alphanumeric character range to avoid CSV special characters (comma, quote, hash)
// that can trigger quoting issues.
data_profile const ti_profile = data_profile_builder().string_char_range('0', 'z');  // ASCII 48-122

/**
 * @brief Returns the type IDs for the given type profile.
 */
std::vector<cudf::type_id> get_type_ids_for_profile(type_profile profile)
{
  switch (profile) {
    case type_profile::ALL_INTEGRAL:
      return get_type_or_group(static_cast<int32_t>(data_type::INTEGRAL));
    case type_profile::ALL_FLOAT:
      return get_type_or_group(static_cast<int32_t>(data_type::FLOAT));
    case type_profile::ALL_STRING:
      return get_type_or_group(static_cast<int32_t>(data_type::STRING));
    case type_profile::MIXED:
      return get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                static_cast<int32_t>(data_type::FLOAT),
                                static_cast<int32_t>(data_type::DECIMAL),
                                static_cast<int32_t>(data_type::TIMESTAMP),
                                static_cast<int32_t>(data_type::DURATION),
                                static_cast<int32_t>(data_type::STRING)});
    default: CUDF_FAIL("Unsupported type profile");
  }
}

/**
 * @brief Benchmark CSV reader with type inference enabled (no explicit dtypes).
 */
template <type_profile TypeProfile>
void csv_read_with_inference(nvbench::state& state,
                             nvbench::type_list<nvbench::enum_type<TypeProfile>>)
{
  auto const data_size_mb = state.get_int64("data_size_mb");
  auto const num_cols     = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  size_t const data_size  = static_cast<size_t>(data_size_mb) << 20;

  auto const base_types = get_type_ids_for_profile(TypeProfile);
  auto const data_types = cycle_dtypes(base_types, num_cols);

  auto const tbl  = create_random_table(data_types, table_size_bytes{data_size}, ti_profile);
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(io_type::DEVICE_BUFFER);
  cudf::io::csv_writer_options write_options =
    cudf::io::csv_writer_options::builder(source_sink.make_sink_info(), view).include_header(true);
  cudf::io::write_csv(write_options);

  // Read WITHOUT specifying dtypes -- forces type inference
  cudf::io::csv_reader_options const read_options =
    cudf::io::csv_reader_options::builder(source_sink.make_source_info())
      .compression(cudf::io::compression_type::NONE);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      timer.start();
      auto const result = cudf::io::read_csv(read_options);
      timer.stop();
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

/**
 * @brief Benchmark CSV reader with explicit dtypes (type inference skipped).
 */
template <type_profile TypeProfile>
void csv_read_without_inference(nvbench::state& state,
                                nvbench::type_list<nvbench::enum_type<TypeProfile>>)
{
  auto const data_size_mb = state.get_int64("data_size_mb");
  auto const num_cols     = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  size_t const data_size  = static_cast<size_t>(data_size_mb) << 20;

  auto const base_types = get_type_ids_for_profile(TypeProfile);
  auto const data_types = cycle_dtypes(base_types, num_cols);

  auto const tbl  = create_random_table(data_types, table_size_bytes{data_size}, ti_profile);
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(io_type::DEVICE_BUFFER);
  cudf::io::csv_writer_options write_options =
    cudf::io::csv_writer_options::builder(source_sink.make_sink_info(), view).include_header(true);
  cudf::io::write_csv(write_options);

  // Extract column types from the source table to skip type inference
  std::vector<cudf::data_type> column_types;
  column_types.reserve(view.num_columns());
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    column_types.push_back(view.column(i).type());
  }

  // Read WITH explicit dtypes -- type inference is skipped
  cudf::io::csv_reader_options const read_options =
    cudf::io::csv_reader_options::builder(source_sink.make_source_info())
      .compression(cudf::io::compression_type::NONE)
      .dtypes(column_types);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      timer.start();
      auto const result = cudf::io::read_csv(read_options);
      timer.stop();
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

using type_profiles = nvbench::enum_type_list<type_profile::ALL_INTEGRAL,
                                              type_profile::ALL_FLOAT,
                                              type_profile::ALL_STRING,
                                              type_profile::MIXED>;

NVBENCH_BENCH_TYPES(csv_read_with_inference, NVBENCH_TYPE_AXES(type_profiles))
  .set_name("csv_read_with_inference")
  .set_type_axes_names({"type_profile"})
  .set_min_samples(4)
  .add_int64_axis("data_size_mb", {64, 256})
  .add_int64_axis("num_cols", {8, 64});

NVBENCH_BENCH_TYPES(csv_read_without_inference, NVBENCH_TYPE_AXES(type_profiles))
  .set_name("csv_read_without_inference")
  .set_type_axes_names({"type_profile"})
  .set_min_samples(4)
  .add_int64_axis("data_size_mb", {64, 256})
  .add_int64_axis("num_cols", {8, 64});
