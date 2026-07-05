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

struct options {
  std::string input_path;
  std::string output_path;
  variant implementation;
  operation selected_operation;
  cudf::size_type rows;
  int iterations;
};

constexpr char request_line_sizes_udf[] = R"***(
__device__ void size_http_request(int32_t* method_size, int32_t* path_size,
                                  int32_t* version_size, cudf::string_view input) {
  auto find_char = [&](char needle, int32_t begin) {
    for (auto i = begin; i < input.size_bytes(); ++i) if (input.data()[i] == needle) return i;
    return input.size_bytes();
  };
  auto method_end = find_char(' ', 0);
  auto target_end = find_char(' ', method_end + 1);
  auto query_begin = find_char('?', method_end + 1);
  auto path_end = query_begin < target_end ? query_begin : target_end;
  *method_size = method_end;
  *path_size = path_end - method_end - 1;
  *version_size = input.size_bytes() - target_end - 6;
}
)***";

constexpr char request_line_output_udf[] = R"***(
__device__ void extract_http_request(cuda::std::span<char>* method, cuda::std::span<char>* path,
                                     cuda::std::span<char>* version,
                                     cudf::string_view input) {
  auto find_char = [&](char needle, int32_t begin) {
    for (auto i = begin; i < input.size_bytes(); ++i) if (input.data()[i] == needle) return i;
    return input.size_bytes();
  };
  auto copy_field = [&](cuda::std::span<char> out, int32_t begin, int32_t end) {
    for (int32_t i = begin; i < end; ++i) out[i - begin] = input.data()[i];
  };
  auto method_end = find_char(' ', 0);
  auto target_end = find_char(' ', method_end + 1);
  auto query_begin = find_char('?', method_end + 1);
  auto path_end = query_begin < target_end ? query_begin : target_end;
  copy_field(*method, 0, method_end);
  copy_field(*path, method_end + 1, path_end);
  copy_field(*version, target_end + 6, input.size_bytes());
}
)***";

constexpr char combined_log_sizes_udf[] = R"***(
__device__ void size_combined_log(int32_t* ip, int32_t* timestamp, int32_t* method,
                                  int32_t* path, int32_t* status, int32_t* referer,
                                  int32_t* user_agent, cudf::string_view input) {
  auto find_char = [&](char needle, int32_t begin) {
    for (auto i = begin; i < input.size_bytes(); ++i) if (input.data()[i] == needle) return i;
    return input.size_bytes();
  };
  auto ip_end = find_char(' ', 0);
  auto timestamp_begin = find_char('[', ip_end) + 1;
  auto timestamp_end = find_char(']', timestamp_begin);
  auto request_begin = find_char('\"', timestamp_end) + 1;
  auto method_end = find_char(' ', request_begin);
  auto target_end = find_char(' ', method_end + 1);
  auto query_begin = find_char('?', method_end + 1);
  auto path_end = query_begin < target_end ? query_begin : target_end;
  auto request_end = find_char('\"', target_end);
  auto status_begin = request_end + 2;
  auto status_end = find_char(' ', status_begin);
  auto bytes_end = find_char(' ', status_end + 1);
  auto referer_begin = find_char('\"', bytes_end) + 1;
  auto referer_end = find_char('\"', referer_begin);
  auto user_agent_begin = find_char('\"', referer_end + 1) + 1;
  auto user_agent_end = find_char('\"', user_agent_begin);
  *ip = ip_end; *timestamp = timestamp_end - timestamp_begin;
  *method = method_end - request_begin; *path = path_end - method_end - 1;
  *status = status_end - status_begin; *referer = referer_end - referer_begin;
  *user_agent = user_agent_end - user_agent_begin;
}
)***";

constexpr char combined_log_output_udf[] = R"***(
__device__ void extract_combined_log(cuda::std::span<char>* ip,
                                     cuda::std::span<char>* timestamp,
                                     cuda::std::span<char>* method,
                                     cuda::std::span<char>* path,
                                     cuda::std::span<char>* status,
                                     cuda::std::span<char>* referer,
                                     cuda::std::span<char>* user_agent,
                                     cudf::string_view input) {
  auto find_char = [&](char needle, int32_t begin) {
    for (auto i = begin; i < input.size_bytes(); ++i) if (input.data()[i] == needle) return i;
    return input.size_bytes();
  };
  auto copy_field = [&](cuda::std::span<char> out, int32_t begin, int32_t end) {
    for (int32_t i = begin; i < end; ++i) out[i - begin] = input.data()[i];
  };
  auto ip_end = find_char(' ', 0);
  auto timestamp_begin = find_char('[', ip_end) + 1;
  auto timestamp_end = find_char(']', timestamp_begin);
  auto request_begin = find_char('\"', timestamp_end) + 1;
  auto method_end = find_char(' ', request_begin);
  auto target_end = find_char(' ', method_end + 1);
  auto query_begin = find_char('?', method_end + 1);
  auto path_end = query_begin < target_end ? query_begin : target_end;
  auto request_end = find_char('\"', target_end);
  auto status_begin = request_end + 2;
  auto status_end = find_char(' ', status_begin);
  auto bytes_end = find_char(' ', status_end + 1);
  auto referer_begin = find_char('\"', bytes_end) + 1;
  auto referer_end = find_char('\"', referer_begin);
  auto user_agent_begin = find_char('\"', referer_end + 1) + 1;
  auto user_agent_end = find_char('\"', user_agent_begin);
  copy_field(*ip, 0, ip_end); copy_field(*timestamp, timestamp_begin, timestamp_end);
  copy_field(*method, request_begin, method_end); copy_field(*path, method_end + 1, path_end);
  copy_field(*status, status_begin, status_end); copy_field(*referer, referer_begin, referer_end);
  copy_field(*user_agent, user_agent_begin, user_agent_end);
}
)***";

[[nodiscard]] std::string const& to_string(variant value)
{
  static std::string const precompiled{"precompiled"};
  static std::string const jit{"jit"};
  static std::string const lto{"lto"};
  switch (value) {
    case variant::PRECOMPILED: return precompiled;
    case variant::JIT: return jit;
    case variant::LTO: return lto;
  }
  throw std::logic_error("Unknown variant");
}

[[nodiscard]] std::string const& to_string(operation value)
{
  static std::string const request_line{"request-line"};
  static std::string const combined_log{"combined-log"};
  return value == operation::REQUEST_LINE ? request_line : combined_log;
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

  auto const implementation = std::string_view{argv[3]} == "precompiled" ? variant::PRECOMPILED
                              : std::string_view{argv[3]} == "jit"       ? variant::JIT
                                                                         : variant::LTO;
  if (std::string_view{argv[3]} != "precompiled" && std::string_view{argv[3]} != "jit" &&
      std::string_view{argv[3]} != "lto") {
    throw std::invalid_argument("variant must be precompiled, jit, or lto");
  }

  auto const selected_operation =
    std::string_view{argv[4]} == "request-line" ? operation::REQUEST_LINE : operation::COMBINED_LOG;
  if (std::string_view{argv[4]} != "request-line" && std::string_view{argv[4]} != "combined-log") {
    throw std::invalid_argument("operation must be request-line or combined-log");
  }

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

[[nodiscard]] std::unique_ptr<cudf::column> make_offsets(cudf::column_view const sizes,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  auto inclusive  = cudf::scan(sizes,
                              *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                              cudf::scan_type::INCLUSIVE,
                              cudf::null_policy::EXCLUDE,
                              stream,
                              mr);
  auto const zero = cudf::numeric_scalar<int32_t>{0, true, stream, mr};
  auto first      = cudf::make_column_from_scalar(zero, 1, stream, mr);
  return cudf::concatenate(
    std::vector<cudf::column_view>{first->view(), inclusive->view()}, stream, mr);
}

[[nodiscard]] std::vector<cudf::transform_output> output_specs(std::size_t count,
                                                               cudf::type_id type)
{
  return std::vector<cudf::transform_output>(
    count, cudf::transform_output{cudf::data_type{type}, cudf::output_nullability::ALL_VALID});
}

[[nodiscard]] std::span<uint8_t const> fragment(std::size_t id)
{
  auto const range = http_log_fragments::file_ranges[id];
  return http_log_fragments::files.subspan(range[0], range[1]);
}

[[nodiscard]] std::unique_ptr<cudf::table> run_precompiled(cudf::column_view input,
                                                           operation selected_operation,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr)
{
  if (selected_operation == operation::REQUEST_LINE) {
    static auto const program =
      cudf::strings::regex_program::create(R"(^([A-Z]+) ([^ ?]+)[^ ]* HTTP/([0-9]+[.][0-9]+)$)");
    return cudf::strings::extract(cudf::strings_column_view{input}, *program, stream, mr);
  }
  static auto const program = cudf::strings::regex_program::create(
    R"regex(^([^ ]+) - [^ ]+ \[([^]]+)\] "([A-Z]+) ([^ ?]+)[^ ]* HTTP/[0-9.]+" ([0-9]{3}) [0-9]+ "([^"]*)" "([^"]*)"$)regex");
  return cudf::strings::extract(cudf::strings_column_view{input}, *program, stream, mr);
}

[[nodiscard]] std::unique_ptr<cudf::table> run_two_pass(cudf::column_view input,
                                                        operation selected_operation,
                                                        variant implementation,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  // Pass 1 emits the exact byte count for every output string and row. Scanning each size column
  // produces run-end offsets, so multi_transform can allocate each chars child once. Pass 2 then
  // receives a cuda::std::span<char> for every row/output and writes directly into final storage.
  auto const count =
    selected_operation == operation::REQUEST_LINE ? std::size_t{3} : std::size_t{7};
  auto sizes_out                 = output_specs(count, cudf::type_id::INT32);
  cudf::transform_input inputs[] = {input};

  std::unique_ptr<cudf::table> sizes;
  if (implementation == variant::JIT) {
    auto const source = selected_operation == operation::REQUEST_LINE ? request_line_sizes_udf
                                                                      : combined_log_sizes_udf;
    sizes             = cudf::multi_transform(source,
                                  cudf::udf_source_type::CUDA,
                                  cudf::null_aware::NO,
                                  std::nullopt,
                                  inputs,
                                  sizes_out,
                                  std::vector<std::unique_ptr<cudf::column>>{},
                                  std::nullopt,
                                  stream,
                                  mr);
  } else {
    auto const id = selected_operation == operation::REQUEST_LINE
                      ? http_log_fragments::request_line_sizes
                      : http_log_fragments::combined_log_sizes;
    sizes         = cudf::transform_lto(fragment(id),
                                cudf::lto_binary_type::FATBIN,
                                cudf::null_aware::NO,
                                std::nullopt,
                                inputs,
                                sizes_out,
                                std::vector<std::unique_ptr<cudf::column>>{},
                                std::nullopt,
                                stream,
                                mr);
  }

  std::vector<std::unique_ptr<cudf::column>> offsets;
  offsets.reserve(count);
  for (auto const& size_column : sizes->view()) {
    offsets.push_back(make_offsets(size_column, stream, mr));
  }

  auto strings_out = output_specs(count, cudf::type_id::STRING);
  if (implementation == variant::JIT) {
    auto const source = selected_operation == operation::REQUEST_LINE ? request_line_output_udf
                                                                      : combined_log_output_udf;
    return cudf::multi_transform(source,
                                 cudf::udf_source_type::CUDA,
                                 cudf::null_aware::NO,
                                 std::nullopt,
                                 inputs,
                                 strings_out,
                                 std::move(offsets),
                                 std::nullopt,
                                 stream,
                                 mr);
  }

  auto const id = selected_operation == operation::REQUEST_LINE
                    ? http_log_fragments::request_line_output
                    : http_log_fragments::combined_log_output;
  return cudf::transform_lto(fragment(id),
                             cudf::lto_binary_type::FATBIN,
                             cudf::null_aware::NO,
                             std::nullopt,
                             inputs,
                             strings_out,
                             std::move(offsets),
                             std::nullopt,
                             stream,
                             mr);
}

[[nodiscard]] std::unique_ptr<cudf::table> run(cudf::column_view input,
                                               operation selected_operation,
                                               variant implementation,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  return implementation == variant::PRECOMPILED
           ? run_precompiled(input, selected_operation, stream, mr)
           : run_two_pass(input, selected_operation, implementation, stream, mr);
}

void write_output(cudf::table_view const result,
                  operation selected_operation,
                  std::string const& output_path)
{
  auto names   = selected_operation == operation::REQUEST_LINE
                   ? std::vector<std::string>{"method", "path", "http_version"}
                   : std::vector<std::string>{
                     "client_ip", "timestamp", "method", "path", "status", "referer", "user_agent"};
  auto options = cudf::io::csv_writer_options::builder(cudf::io::sink_info{output_path}, result)
                   .include_header(true)
                   .names(names)
                   .build();
  cudf::io::write_csv(options);
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

    auto read_options =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{opts.input_path})
        .header(0)
        .build();
    auto input_data = cudf::io::read_csv(read_options);
    auto input =
      opts.rows == input_data.tbl->num_rows()
        ? std::move(input_data.tbl)
        : cudf::sample(input_data.tbl->view(), opts.rows, cudf::sample_with_replacement::TRUE);
    auto const input_index  = opts.selected_operation == operation::REQUEST_LINE ? 0 : 1;
    auto const input_bytes  = input->get_column(input_index).alloc_size();
    auto const input_column = input->get_column(input_index).view();

    rmm::mr::statistics_resource_adaptor stats{mr};
    auto const stats_mr = rmm::device_async_resource_ref{stats};

    stream.synchronize();
    auto const cold_start = std::chrono::steady_clock::now();
    nvtxRangePush("http_log_cold");
    auto cold_result =
      run(input_column, opts.selected_operation, opts.implementation, stream, stats_mr);
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
      result = run(input_column, opts.selected_operation, opts.implementation, stream, stats_mr);
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
