/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr auto output_count = std::size_t{3};

// Runtime JIT compilation consumes one CUDA source string for each pass.
constexpr char request_line_sizes_udf[] = R"***(
// multi_transform calls this function once per input row. Pointer parameters are output columns;
// writing a byte count to each one lets the host create exact string offsets before allocation.
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

  auto method_end  = find_character(' ', 0);
  auto target_end  = find_character(' ', method_end + 1);
  auto query_begin = find_character('?', method_end + 1);
  // Strip the query string so the path matches the first regex capture workload.
  auto path_end    = query_begin < target_end ? query_begin : target_end;

  *method_size  = method_end;
  *path_size    = path_end - method_end - 1;
  *version_size = input.size_bytes() - target_end - 6;
}
)***";

constexpr char request_line_output_udf[] = R"***(
// Each span points at the final character buffer for one output string in this row. Its size came
// from compute_request_line_sizes, so this pass only copies bytes and performs no allocation.
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

  auto method_end  = find_character(' ', 0);
  auto target_end  = find_character(' ', method_end + 1);
  auto query_begin = find_character('?', method_end + 1);
  auto path_end    = query_begin < target_end ? query_begin : target_end;

  copy_field(*method, 0, method_end);
  copy_field(*path, method_end + 1, path_end);
  copy_field(*version, target_end + 6, input.size_bytes());
}
)***";

constexpr std::string_view usage =
  "usage: http_log_transforms INPUT.csv OUTPUT.csv "
  "<precompiled|jit|lto> ROWS ITERATIONS\n"
  "       http_log_transforms <usage|--help>\n";

[[nodiscard]] std::unique_ptr<cudf::table> run_precompiled(cudf::column_view input,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr)
{
  static auto program =
    cudf::strings::regex_program::create(R"(^([A-Z]+) ([^ ?]+)[^ ]* HTTP/([0-9]+[.][0-9]+)$)");
  return cudf::strings::extract(cudf::strings_column_view{input}, *program, stream, mr);
}

[[nodiscard]] std::unique_ptr<cudf::table> run_jit(cudf::column_view input,
                                                   bool use_lto,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  // Pass 1 produces one byte-count column for each eventual string output.
  cudf::transform_output const size_spec{cudf::data_type{cudf::type_id::INT32},
                                         cudf::output_nullability::ALL_VALID};
  std::vector<cudf::transform_output> const size_outputs(output_count, size_spec);
  cudf::transform_input inputs[] = {input};

  std::unique_ptr<cudf::table> sizes;

  if (use_lto) {
    auto range    = http_log_fragments::file_ranges[http_log_fragments::request_line_sizes];
    auto fragment = http_log_fragments::files.subspan(range[0], range[1]);

    sizes = cudf::transform_lto(fragment,
                                cudf::lto_binary_type::FATBIN,
                                cudf::null_aware::NO,
                                std::nullopt,
                                inputs,
                                size_outputs,
                                {},
                                std::nullopt,
                                stream,
                                mr);
  } else {
    sizes = cudf::multi_transform(request_line_sizes_udf,
                                  cudf::udf_source_type::CUDA,
                                  cudf::null_aware::NO,
                                  std::nullopt,
                                  inputs,
                                  size_outputs,
                                  {},
                                  std::nullopt,
                                  stream,
                                  mr);
  }

  // Inclusive scans turn the sizes into the offsets needed for the final strings columns.
  std::vector<std::unique_ptr<cudf::column>> offsets;
  offsets.reserve(output_count);

  for (auto& string_sizes : sizes->view()) {
    auto run_ends = cudf::scan(string_sizes,
                               *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                               cudf::scan_type::INCLUSIVE,
                               cudf::null_policy::EXCLUDE,
                               stream,
                               mr);

    auto zero  = cudf::numeric_scalar<int32_t>{0, true, stream, mr};
    auto first = cudf::make_column_from_scalar(zero, 1, stream, mr);
    offsets.push_back(cudf::concatenate(
      std::vector<cudf::column_view>{first->view(), run_ends->view()}, stream, mr));
  }

  // Pass 2 writes directly into final character buffers described by those offsets.
  cudf::transform_output const output_spec{cudf::data_type{cudf::type_id::STRING},
                                           cudf::output_nullability::ALL_VALID};
  std::vector<cudf::transform_output> const outputs(output_count, output_spec);

  if (use_lto) {
    auto range    = http_log_fragments::file_ranges[http_log_fragments::request_line_output];
    auto fragment = http_log_fragments::files.subspan(range[0], range[1]);
    return cudf::transform_lto(fragment,
                               cudf::lto_binary_type::FATBIN,
                               cudf::null_aware::NO,
                               std::nullopt,
                               inputs,
                               outputs,
                               std::move(offsets),
                               std::nullopt,
                               stream,
                               mr);
  }

  return cudf::multi_transform(request_line_output_udf,
                               cudf::udf_source_type::CUDA,
                               cudf::null_aware::NO,
                               std::nullopt,
                               inputs,
                               outputs,
                               std::move(offsets),
                               std::nullopt,
                               stream,
                               mr);
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

    if (argc != 6) {
      throw std::invalid_argument("invalid arguments; run http_log_transforms --help for usage");
    }

    auto input_path     = std::string{argv[1]};
    auto output_path    = std::string{argv[2]};
    auto implementation = std::string_view{argv[3]};
    if (implementation != "precompiled" && implementation != "jit" && implementation != "lto") {
      throw std::invalid_argument("variant must be precompiled, jit, or lto");
    }

    auto requested_rows = std::stoll(argv[4]);
    auto iterations     = std::stoi(argv[5]);
    if (requested_rows < 0 || requested_rows > std::numeric_limits<cudf::size_type>::max()) {
      throw std::invalid_argument("ROWS is outside the cudf::size_type range");
    }

    if (iterations < 1) { throw std::invalid_argument("ITERATIONS must be positive"); }

    auto rows           = static_cast<cudf::size_type>(requested_rows);
    auto is_precompiled = implementation == "precompiled";
    auto use_lto        = implementation == "lto";
    auto stream         = cudf::get_default_stream();
    auto mr             = cudf::get_current_device_resource_ref();

    auto read_options = cudf::io::csv_reader_options::builder(cudf::io::source_info{input_path})
                          .header(0)
                          .use_cols_names({"RequestLine"})
                          .build();
    auto input = cudf::io::read_csv(read_options).tbl;
    if (rows != input->num_rows()) {
      // Sampling with replacement scales the small checked-in dataset to the requested size.
      input = cudf::sample(input->view(), rows, cudf::sample_with_replacement::TRUE);
    }

    auto input_bytes = input->get_column(0).alloc_size();
    auto input_view  = input->get_column(0).view();

    // Track allocations made by the transforms without changing the application's upstream memory
    // resource.
    rmm::mr::statistics_resource_adaptor stats{mr};
    auto stats_mr = rmm::device_async_resource_ref{stats};

    stream.synchronize();
    // The cold measurement includes regex setup or JIT compilation/linking performed on first use.
    auto cold_start = std::chrono::steady_clock::now();
    nvtxRangePush("http_log_cold");
    auto cold_result = is_precompiled ? run_precompiled(input_view, stream, stats_mr)
                                      : run_jit(input_view, use_lto, stream, stats_mr);
    stream.synchronize();
    nvtxRangePop();
    auto cold_seconds =
      std::chrono::duration<double>{std::chrono::steady_clock::now() - cold_start}.count();
    cold_result.reset();

    std::unique_ptr<cudf::table> result;
    // Subsequent calls exercise the cached kernel and represent steady-state throughput.
    auto warm_start = std::chrono::steady_clock::now();
    nvtxRangePush("http_log_warm");
    for (auto i = 0; i < iterations; ++i) {
      result.reset();
      result = is_precompiled ? run_precompiled(input_view, stream, stats_mr)
                              : run_jit(input_view, use_lto, stream, stats_mr);
    }
    stream.synchronize();
    nvtxRangePop();
    auto warm_seconds =
      std::chrono::duration<double>{std::chrono::steady_clock::now() - warm_start}.count() /
      iterations;

    // A dash suppresses CSV output so file I/O does not affect benchmark runs.
    if (output_path != "-") {
      auto write_options =
        cudf::io::csv_writer_options::builder(cudf::io::sink_info{output_path}, result->view())
          .include_header(true)
          .names({"method", "path", "http_version"})
          .build();
      cudf::io::write_csv(write_options);
    }

    auto bytes        = stats.get_bytes_counter();
    auto output_bytes = result->alloc_size();
    auto gib          = static_cast<double>(input_bytes + output_bytes) / (1ULL << 30);

    std::cout << std::fixed << std::setprecision(9) << "RESULT variant=" << implementation
              << " rows=" << rows << " cold_seconds=" << cold_seconds
              << " warm_seconds=" << warm_seconds
              << " rows_per_second=" << static_cast<double>(rows) / warm_seconds
              << " effective_gib_per_second=" << gib / warm_seconds
              << " input_bytes=" << input_bytes << " output_bytes=" << output_bytes
              << " peak_memory_bytes=" << bytes.peak << " total_allocated_bytes=" << bytes.total
              << " allocated_bytes_per_call="
              << bytes.total / static_cast<std::size_t>(iterations + 1) << '\n';
    return EXIT_SUCCESS;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
