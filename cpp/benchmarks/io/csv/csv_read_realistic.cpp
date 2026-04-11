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

// Realistic mixed-type column profiles matching real-world datasets.
// These profiles expose warp divergence from mixed parsing paths that
// single-type benchmarks miss.

enum class csv_profile : int32_t { TAXI, LOGS, ANALYTICS };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  csv_profile,
  [](csv_profile value) {
    switch (value) {
      case csv_profile::TAXI: return "TAXI";
      case csv_profile::LOGS: return "LOGS";
      case csv_profile::ANALYTICS: return "ANALYTICS";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

// NYC taxi trip data: 14 columns with ints, floats, timestamps, and strings.
std::vector<cudf::type_id> const taxi_dtypes = {
  cudf::type_id::INT32,                   // VendorID
  cudf::type_id::TIMESTAMP_MILLISECONDS,  // pickup_datetime
  cudf::type_id::TIMESTAMP_MILLISECONDS,  // dropoff_datetime
  cudf::type_id::INT8,                    // passenger_count
  cudf::type_id::FLOAT64,                 // trip_distance
  cudf::type_id::FLOAT64,                 // pickup_longitude
  cudf::type_id::FLOAT64,                 // pickup_latitude
  cudf::type_id::INT32,                   // RatecodeID
  cudf::type_id::STRING,                  // store_and_fwd_flag
  cudf::type_id::FLOAT64,                 // dropoff_longitude
  cudf::type_id::FLOAT64,                 // dropoff_latitude
  cudf::type_id::INT8,                    // payment_type
  cudf::type_id::FLOAT64,                 // fare_amount
  cudf::type_id::FLOAT64,                 // total_amount
};

// Server log data: 6 columns, timestamp-heavy with strings.
std::vector<cudf::type_id> const logs_dtypes = {
  cudf::type_id::TIMESTAMP_MILLISECONDS,
  cudf::type_id::STRING,
  cudf::type_id::STRING,
  cudf::type_id::INT32,
  cudf::type_id::INT32,
  cudf::type_id::STRING,
};

// Analytics/numeric data: 8 columns, all numeric (ints + doubles).
std::vector<cudf::type_id> const analytics_dtypes = {
  cudf::type_id::INT64,
  cudf::type_id::INT64,
  cudf::type_id::FLOAT64,
  cudf::type_id::FLOAT64,
  cudf::type_id::FLOAT64,
  cudf::type_id::FLOAT64,
  cudf::type_id::FLOAT64,
  cudf::type_id::FLOAT64,
};

// Avoid CSV special characters (comma, quote, hash) in generated strings.
data_profile const profile = data_profile_builder().string_char_range('0', 'z');  // ASCII 48-122

template <csv_profile Profile>
void BM_csv_read_realistic(nvbench::state& state,
                           nvbench::type_list<nvbench::enum_type<Profile>>)
{
  auto const data_size_mb = static_cast<size_t>(state.get_int64("data_size_mb"));
  auto const data_size    = data_size_mb * 1024UL * 1024UL;

  // Select dtype vector based on profile
  std::vector<cudf::type_id> const* dtypes_ptr = nullptr;
  switch (Profile) {
    case csv_profile::TAXI: dtypes_ptr = &taxi_dtypes; break;
    case csv_profile::LOGS: dtypes_ptr = &logs_dtypes; break;
    case csv_profile::ANALYTICS: dtypes_ptr = &analytics_dtypes; break;
  }
  auto const& dtypes = *dtypes_ptr;

  auto const tbl  = create_random_table(dtypes, table_size_bytes{data_size}, profile);
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(io_type::DEVICE_BUFFER);
  cudf::io::csv_writer_options write_options =
    cudf::io::csv_writer_options::builder(source_sink.make_sink_info(), view).include_header(true);
  cudf::io::write_csv(write_options);

  // Extract column types from the generated table for explicit dtype specification.
  // This measures parse throughput, not type inference.
  std::vector<cudf::data_type> column_types;
  column_types.reserve(view.num_columns());
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    column_types.push_back(view.column(i).type());
  }

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

using profile_list =
  nvbench::enum_type_list<csv_profile::TAXI, csv_profile::LOGS, csv_profile::ANALYTICS>;

NVBENCH_BENCH_TYPES(BM_csv_read_realistic, NVBENCH_TYPE_AXES(profile_list))
  .set_name("csv_read_realistic")
  .set_type_axes_names({"profile"})
  .set_min_samples(4)
  .add_int64_axis("data_size_mb", {256, 512, 1024});
