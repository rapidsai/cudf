/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/reduction.hpp>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <http_log_fragments.hpp>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

enum class variant { PRECOMPILED, JIT, LTO };
enum class operation { REQUEST_LINE, COMBINED_LOG };

constexpr auto request_line_output_count = std::size_t{3};
constexpr auto combined_log_output_count = std::size_t{7};

constexpr std::string_view request_line_pattern =
  R"(^([A-Z]+) ([^ ?]+)[^ ]* HTTP/([0-9]+[.][0-9]+)$)";
constexpr std::string_view combined_log_pattern =
  R"regex(^([^ ]+) - [^ ]+ \[([^]]+)\] "([A-Z]+) ([^ ?]+)[^ ]* HTTP/[0-9.]+" ([0-9]{3}) [0-9]+ "([^"]*)" "([^"]*)"$)regex";

struct options {
  std::string input_path;
  std::string output_path;
  variant implementation;
  operation selected_operation;
  cudf::size_type rows;
  int iterations;
};

// Runtime JIT compilation consumes CUDA source strings. Each operation has one UDF that computes
// exact output sizes and another that writes into the resulting character buffers.
constexpr char request_line_sizes_udf[] = R"***(
__device__ void compute_request_line_sizes(int32_t* method_size,
                                           int32_t* path_size,
                                           int32_t* version_size,
                                           cudf::string_view input) {
  auto find_character = [&](char needle, int32_t begin) {
    for (auto index = begin; index < input.size_bytes(); ++index) {
      if (input.data()[index] == needle) { return index; }
    }
    return input.size_bytes();
  };

  auto const method_end  = find_character(' ', 0);
  auto const target_end  = find_character(' ', method_end + 1);
  auto const query_begin = find_character('?', method_end + 1);
  auto const path_end    = query_begin < target_end ? query_begin : target_end;

  *method_size  = method_end;
  *path_size    = path_end - method_end - 1;
  *version_size = input.size_bytes() - target_end - 6;
}
)***";

constexpr char request_line_output_udf[] = R"***(
__device__ void write_request_line(cuda::std::span<char>* method,
                                   cuda::std::span<char>* path,
                                   cuda::std::span<char>* version,
                                   cudf::string_view input) {
  auto find_character = [&](char needle, int32_t begin) {
    for (auto index = begin; index < input.size_bytes(); ++index) {
      if (input.data()[index] == needle) { return index; }
    }
    return input.size_bytes();
  };

  auto copy_field = [&](cuda::std::span<char> out, int32_t begin, int32_t end) {
    for (auto index = begin; index < end; ++index) {
      out[index - begin] = input.data()[index];
    }
  };

  auto const method_end  = find_character(' ', 0);
  auto const target_end  = find_character(' ', method_end + 1);
  auto const query_begin = find_character('?', method_end + 1);
  auto const path_end    = query_begin < target_end ? query_begin : target_end;

  copy_field(*method, 0, method_end);
  copy_field(*path, method_end + 1, path_end);
  copy_field(*version, target_end + 6, input.size_bytes());
}
)***";

constexpr char combined_log_sizes_udf[] = R"***(
__device__ void compute_combined_log_sizes(int32_t* client_ip_size,
                                           int32_t* timestamp_size,
                                           int32_t* method_size,
                                           int32_t* path_size,
                                           int32_t* status_size,
                                           int32_t* referer_size,
                                           int32_t* user_agent_size,
                                           cudf::string_view input) {
  auto find_character = [&](char needle, int32_t begin) {
    for (auto index = begin; index < input.size_bytes(); ++index) {
      if (input.data()[index] == needle) { return index; }
    }
    return input.size_bytes();
  };

  auto const client_ip_end    = find_character(' ', 0);
  auto const timestamp_begin  = find_character('[', client_ip_end) + 1;
  auto const timestamp_end    = find_character(']', timestamp_begin);
  auto const request_begin    = find_character('\"', timestamp_end) + 1;
  auto const method_end       = find_character(' ', request_begin);
  auto const target_end       = find_character(' ', method_end + 1);
  auto const query_begin      = find_character('?', method_end + 1);
  auto const path_end         = query_begin < target_end ? query_begin : target_end;
  auto const request_end      = find_character('\"', target_end);
  auto const status_begin     = request_end + 2;
  auto const status_end       = find_character(' ', status_begin);
  auto const bytes_end        = find_character(' ', status_end + 1);
  auto const referer_begin    = find_character('\"', bytes_end) + 1;
  auto const referer_end      = find_character('\"', referer_begin);
  auto const user_agent_begin = find_character('\"', referer_end + 1) + 1;
  auto const user_agent_end   = find_character('\"', user_agent_begin);

  *client_ip_size  = client_ip_end;
  *timestamp_size  = timestamp_end - timestamp_begin;
  *method_size     = method_end - request_begin;
  *path_size       = path_end - method_end - 1;
  *status_size     = status_end - status_begin;
  *referer_size    = referer_end - referer_begin;
  *user_agent_size = user_agent_end - user_agent_begin;
}
)***";

constexpr char combined_log_output_udf[] = R"***(
__device__ void write_combined_log(cuda::std::span<char>* client_ip,
                                   cuda::std::span<char>* timestamp,
                                   cuda::std::span<char>* method,
                                   cuda::std::span<char>* path,
                                   cuda::std::span<char>* status,
                                   cuda::std::span<char>* referer,
                                   cuda::std::span<char>* user_agent,
                                   cudf::string_view input) {
  auto find_character = [&](char needle, int32_t begin) {
    for (auto index = begin; index < input.size_bytes(); ++index) {
      if (input.data()[index] == needle) { return index; }
    }
    return input.size_bytes();
  };

  auto copy_field = [&](cuda::std::span<char> out, int32_t begin, int32_t end) {
    for (auto index = begin; index < end; ++index) {
      out[index - begin] = input.data()[index];
    }
  };

  auto const client_ip_end    = find_character(' ', 0);
  auto const timestamp_begin  = find_character('[', client_ip_end) + 1;
  auto const timestamp_end    = find_character(']', timestamp_begin);
  auto const request_begin    = find_character('\"', timestamp_end) + 1;
  auto const method_end       = find_character(' ', request_begin);
  auto const target_end       = find_character(' ', method_end + 1);
  auto const query_begin      = find_character('?', method_end + 1);
  auto const path_end         = query_begin < target_end ? query_begin : target_end;
  auto const request_end      = find_character('\"', target_end);
  auto const status_begin     = request_end + 2;
  auto const status_end       = find_character(' ', status_begin);
  auto const bytes_end        = find_character(' ', status_end + 1);
  auto const referer_begin    = find_character('\"', bytes_end) + 1;
  auto const referer_end      = find_character('\"', referer_begin);
  auto const user_agent_begin = find_character('\"', referer_end + 1) + 1;
  auto const user_agent_end   = find_character('\"', user_agent_begin);

  copy_field(*client_ip, 0, client_ip_end);
  copy_field(*timestamp, timestamp_begin, timestamp_end);
  copy_field(*method, request_begin, method_end);
  copy_field(*path, method_end + 1, path_end);
  copy_field(*status, status_begin, status_end);
  copy_field(*referer, referer_begin, referer_end);
  copy_field(*user_agent, user_agent_begin, user_agent_end);
}
)***";

[[nodiscard]] constexpr std::string_view to_string(variant value)
{
  switch (value) {
    case variant::PRECOMPILED: return "precompiled";
    case variant::JIT: return "jit";
    case variant::LTO: return "lto";
  }
  throw std::logic_error("Unknown variant");
}

[[nodiscard]] constexpr std::string_view to_string(operation value)
{
  switch (value) {
    case operation::REQUEST_LINE: return "request-line";
    case operation::COMBINED_LOG: return "combined-log";
  }
  throw std::logic_error("Unknown operation");
}

[[nodiscard]] variant parse_variant(std::string_view name)
{
  if (name == "precompiled") { return variant::PRECOMPILED; }
  if (name == "jit") { return variant::JIT; }
  if (name == "lto") { return variant::LTO; }
  throw std::invalid_argument("variant must be precompiled, jit, or lto");
}

[[nodiscard]] operation parse_operation(std::string_view name)
{
  if (name == "request-line") { return operation::REQUEST_LINE; }
  if (name == "combined-log") { return operation::COMBINED_LOG; }
  throw std::invalid_argument("operation must be request-line or combined-log");
}

constexpr std::string_view usage =
  "usage: http_log_transforms INPUT.csv OUTPUT.csv "
  "<precompiled|jit|lto> <request-line|combined-log> ROWS ITERATIONS\n"
  "       http_log_transforms <usage|--help>\n";

[[nodiscard]] options parse_options(int argc, char const** argv)
{
  if (argc != 7) {
    throw std::invalid_argument("invalid arguments; run http_log_transforms --help for usage");
  }

  auto const implementation     = parse_variant(argv[3]);
  auto const selected_operation = parse_operation(argv[4]);

  auto const rows       = std::stoll(argv[5]);
  auto const iterations = std::stoi(argv[6]);
  if (rows < 0 || rows > std::numeric_limits<cudf::size_type>::max()) {
    throw std::invalid_argument("ROWS is outside the cudf::size_type range");
  }
  if (iterations < 1) { throw std::invalid_argument("ITERATIONS must be positive"); }
  return {argv[1],
          argv[2],
          implementation,
          selected_operation,
          static_cast<cudf::size_type>(rows),
          iterations};
}

[[nodiscard]] constexpr std::size_t output_count(operation selected_operation)
{
  return selected_operation == operation::REQUEST_LINE ? request_line_output_count
                                                       : combined_log_output_count;
}

[[nodiscard]] constexpr cudf::size_type input_column_index(operation selected_operation)
{
  return selected_operation == operation::REQUEST_LINE ? 0 : 1;
}

[[nodiscard]] char const* sizing_udf(operation selected_operation)
{
  return selected_operation == operation::REQUEST_LINE ? request_line_sizes_udf
                                                       : combined_log_sizes_udf;
}

[[nodiscard]] char const* output_udf(operation selected_operation)
{
  return selected_operation == operation::REQUEST_LINE ? request_line_output_udf
                                                       : combined_log_output_udf;
}

[[nodiscard]] std::size_t sizing_fragment(operation selected_operation)
{
  return selected_operation == operation::REQUEST_LINE ? http_log_fragments::request_line_sizes
                                                       : http_log_fragments::combined_log_sizes;
}

[[nodiscard]] std::size_t output_fragment(operation selected_operation)
{
  return selected_operation == operation::REQUEST_LINE ? http_log_fragments::request_line_output
                                                       : http_log_fragments::combined_log_output;
}

[[nodiscard]] std::vector<cudf::transform_output> make_output_specs(std::size_t count,
                                                                    cudf::type_id type)
{
  auto const spec =
    cudf::transform_output{cudf::data_type{type}, cudf::output_nullability::ALL_VALID};
  return std::vector<cudf::transform_output>(count, spec);
}

[[nodiscard]] std::span<uint8_t const> get_fragment(std::size_t id)
{
  auto const range = http_log_fragments::file_ranges[id];
  return http_log_fragments::files.subspan(range[0], range[1]);
}

[[nodiscard]] std::unique_ptr<cudf::column> make_string_offsets(
  cudf::column_view const string_sizes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // A strings column uses N+1 offsets. Scanning the N string sizes produces every run end; adding
  // a leading zero supplies the first offset.
  auto run_ends          = cudf::scan(string_sizes,
                             *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                             cudf::scan_type::INCLUSIVE,
                             cudf::null_policy::EXCLUDE,
                             stream,
                             mr);
  auto const zero_offset = cudf::numeric_scalar<int32_t>{0, true, stream, mr};
  auto first_offset      = cudf::make_column_from_scalar(zero_offset, 1, stream, mr);
  return cudf::concatenate(
    std::vector<cudf::column_view>{first_offset->view(), run_ends->view()}, stream, mr);
}

[[nodiscard]] std::unique_ptr<cudf::table> run_regex(cudf::column_view input,
                                                     operation selected_operation,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  // This public cuDF regex implementation is the non-JIT comparison baseline.
  if (selected_operation == operation::REQUEST_LINE) {
    static auto const program = cudf::strings::regex_program::create(request_line_pattern);
    return cudf::strings::extract(cudf::strings_column_view{input}, *program, stream, mr);
  }
  static auto const program = cudf::strings::regex_program::create(combined_log_pattern);
  return cudf::strings::extract(cudf::strings_column_view{input}, *program, stream, mr);
}

[[nodiscard]] std::unique_ptr<cudf::table> compute_string_sizes(cudf::column_view input,
                                                                operation selected_operation,
                                                                variant implementation,
                                                                rmm::cuda_stream_view stream,
                                                                rmm::device_async_resource_ref mr)
{
  // Pass 1: produce one INT32 byte-count column for each eventual string output.
  auto const outputs = make_output_specs(output_count(selected_operation), cudf::type_id::INT32);
  cudf::transform_input inputs[] = {input};

  if (implementation == variant::JIT) {
    return cudf::multi_transform(sizing_udf(selected_operation),
                                 cudf::udf_source_type::CUDA,
                                 cudf::null_aware::NO,
                                 std::nullopt,
                                 inputs,
                                 outputs,
                                 {},
                                 std::nullopt,
                                 stream,
                                 mr);
  }

  return cudf::transform_lto(get_fragment(sizing_fragment(selected_operation)),
                             cudf::lto_binary_type::FATBIN,
                             cudf::null_aware::NO,
                             std::nullopt,
                             inputs,
                             outputs,
                             {},
                             std::nullopt,
                             stream,
                             mr);
}

[[nodiscard]] std::vector<std::unique_ptr<cudf::column>> make_all_string_offsets(
  cudf::table_view const size_columns,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> offsets;
  offsets.reserve(size_columns.num_columns());
  for (auto const& string_sizes : size_columns) {
    offsets.push_back(make_string_offsets(string_sizes, stream, mr));
  }
  return offsets;
}

[[nodiscard]] std::unique_ptr<cudf::table> write_strings(
  cudf::column_view input,
  operation selected_operation,
  variant implementation,
  std::vector<std::unique_ptr<cudf::column>>&& string_offsets,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Pass 2: use the precomputed offsets to write directly into final strings columns.
  auto const outputs = make_output_specs(output_count(selected_operation), cudf::type_id::STRING);
  cudf::transform_input inputs[] = {input};

  if (implementation == variant::JIT) {
    return cudf::multi_transform(output_udf(selected_operation),
                                 cudf::udf_source_type::CUDA,
                                 cudf::null_aware::NO,
                                 std::nullopt,
                                 inputs,
                                 outputs,
                                 std::move(string_offsets),
                                 std::nullopt,
                                 stream,
                                 mr);
  }

  return cudf::transform_lto(get_fragment(output_fragment(selected_operation)),
                             cudf::lto_binary_type::FATBIN,
                             cudf::null_aware::NO,
                             std::nullopt,
                             inputs,
                             outputs,
                             std::move(string_offsets),
                             std::nullopt,
                             stream,
                             mr);
}

[[nodiscard]] std::unique_ptr<cudf::table> run_two_pass(cudf::column_view input,
                                                        operation selected_operation,
                                                        variant implementation,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  // Pass 1 computes the exact byte count of every output string. Inclusive scans turn those sizes
  // into run-end offsets, allowing pass 2 to write directly into the final character buffers.
  auto sizes   = compute_string_sizes(input, selected_operation, implementation, stream, mr);
  auto offsets = make_all_string_offsets(sizes->view(), stream, mr);
  return write_strings(input, selected_operation, implementation, std::move(offsets), stream, mr);
}

[[nodiscard]] std::unique_ptr<cudf::table> run_transform(cudf::column_view input,
                                                         operation selected_operation,
                                                         variant implementation,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  if (implementation == variant::PRECOMPILED) {
    return run_regex(input, selected_operation, stream, mr);
  }
  return run_two_pass(input, selected_operation, implementation, stream, mr);
}

[[nodiscard]] std::vector<std::string> output_column_names(operation selected_operation)
{
  if (selected_operation == operation::REQUEST_LINE) { return {"method", "path", "http_version"}; }
  return {"client_ip", "timestamp", "method", "path", "status", "referer", "user_agent"};
}

void write_output(cudf::table_view const result,
                  operation selected_operation,
                  std::string const& output_path)
{
  auto options = cudf::io::csv_writer_options::builder(cudf::io::sink_info{output_path}, result)
                   .include_header(true)
                   .names(output_column_names(selected_operation))
                   .build();
  cudf::io::write_csv(options);
}

[[nodiscard]] std::unique_ptr<cudf::table> read_input(options const& opts)
{
  auto read_options =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{opts.input_path}).header(0).build();
  auto input = cudf::io::read_csv(read_options).tbl;

  if (opts.rows == input->num_rows()) { return input; }
  return cudf::sample(input->view(), opts.rows, cudf::sample_with_replacement::TRUE);
}

}  // namespace

int main(int argc, char const** argv)
{
  try {
    if (argc == 2 &&
        (std::string_view{argv[1]} == "--help" || std::string_view{argv[1]} == "usage")) {
      std::cout << usage;
      return EXIT_SUCCESS;
    }

    auto const opts   = parse_options(argc, argv);
    auto const stream = cudf::get_default_stream();
    auto const mr     = cudf::get_current_device_resource_ref();

    auto input             = read_input(opts);
    auto const input_index = input_column_index(opts.selected_operation);
    auto const input_bytes = input->get_column(input_index).alloc_size();
    auto const input_view  = input->get_column(input_index).view();

    rmm::mr::statistics_resource_adaptor stats{mr};
    auto const stats_mr = rmm::device_async_resource_ref{stats};

    stream.synchronize();
    auto const cold_start = std::chrono::steady_clock::now();
    nvtxRangePush("http_log_cold");
    auto cold_result =
      run_transform(input_view, opts.selected_operation, opts.implementation, stream, stats_mr);
    stream.synchronize();
    nvtxRangePop();
    auto const cold_seconds =
      std::chrono::duration<double>{std::chrono::steady_clock::now() - cold_start}.count();
    cold_result.reset();

    std::unique_ptr<cudf::table> result;
    auto const warm_start = std::chrono::steady_clock::now();
    nvtxRangePush("http_log_warm");
    for (auto i = 0; i < opts.iterations; ++i) {
      result.reset();
      result =
        run_transform(input_view, opts.selected_operation, opts.implementation, stream, stats_mr);
    }
    stream.synchronize();
    nvtxRangePop();
    auto const warm_seconds =
      std::chrono::duration<double>{std::chrono::steady_clock::now() - warm_start}.count() /
      opts.iterations;

    if (opts.output_path != "-") {
      write_output(result->view(), opts.selected_operation, opts.output_path);
    }

    auto const bytes        = stats.get_bytes_counter();
    auto const output_bytes = result->alloc_size();
    auto const gib          = static_cast<double>(input_bytes + output_bytes) / (1ULL << 30);

    std::cout << std::fixed << std::setprecision(9)
              << "RESULT variant=" << to_string(opts.implementation)
              << " operation=" << to_string(opts.selected_operation) << " rows=" << opts.rows
              << " cold_seconds=" << cold_seconds << " warm_seconds=" << warm_seconds
              << " rows_per_second=" << static_cast<double>(opts.rows) / warm_seconds
              << " effective_gib_per_second=" << gib / warm_seconds
              << " input_bytes=" << input_bytes << " output_bytes=" << output_bytes
              << " peak_memory_bytes=" << bytes.peak << " total_allocated_bytes=" << bytes.total
              << " allocated_bytes_per_call="
              << bytes.total / static_cast<std::size_t>(opts.iterations + 1) << '\n';
    return EXIT_SUCCESS;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
