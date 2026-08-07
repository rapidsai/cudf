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
#include <cudf/replace.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/split/partition.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <url_log_fragments.hpp>

#include <chrono>
#include <cstdlib>
#include <format>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr auto output_count = std::size_t{6};

// Shared CUDA source inserted into both runtime-compiled UDF bodies.
constexpr char parse_url_udf[] = R"***(
  struct range32 {
    int32_t begin{};
    int32_t end{};
  };
  struct url_ranges {
    range32 protocol;
    range32 host;
    range32 port;
    range32 path;
    range32 query;
    range32 fragment;
  };
  // Parses the first URL candidate and records byte ranges for all six components.
  auto const parse_url = [&](url_ranges* out) {
    *out = {};
    auto const n = input.size_bytes();
    auto const is_alpha = [](char c) {
      return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    };
    auto const is_digit = [](char c) { return c >= '0' && c <= '9'; };
    auto const is_scheme_char = [&](char c) {
      return is_alpha(c) || is_digit(c) || c == '+' || c == '-' || c == '.';
    };
    auto const is_hex = [&](char c) {
      return is_digit(c) || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
    };
    auto const is_unreserved = [&](char c) {
      return is_alpha(c) || is_digit(c) || c == '-' || c == '.' || c == '_' || c == '~';
    };
    auto const is_sub_delim = [](char c) {
      return c == '!' || c == '$' || c == '&' || c == '\'' || c == '(' || c == ')' || c == '*' ||
             c == '+' || c == ',' || c == ';' || c == '=';
    };
    auto const is_gen_delim = [](char c) {
      return c == ':' || c == '/' || c == '?' || c == '#' || c == '[' || c == ']' || c == '@';
    };
    auto const is_context_delimiter = [](char c) {
      return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '"' || c == '<' ||
             c == '>';
    };

    auto scheme_end = n;
    for (auto i = 1; i + 2 < n; ++i) {
      if (input.data()[i] == ':' && input.data()[i + 1] == '/' && input.data()[i + 2] == '/') {
        scheme_end = i;
        break;
      }
    }
    if (scheme_end == n) { return false; }

    auto url_begin = scheme_end;
    while (url_begin > 0 && is_scheme_char(input.data()[url_begin - 1])) { --url_begin; }
    if (url_begin == scheme_end || !is_alpha(input.data()[url_begin])) { return false; }

    auto url_end = n;
    for (auto i = scheme_end + 3; i < n; ++i) {
      if (is_context_delimiter(input.data()[i])) {
        url_end = i;
        break;
      }
    }
    for (auto i = url_begin; i < url_end; ++i) {
      auto const c = input.data()[i];
      if (c == '%') {
        if (i + 2 >= url_end || !is_hex(input.data()[i + 1]) || !is_hex(input.data()[i + 2])) {
          return false;
        }
        i += 2;
      } else if (!is_unreserved(c) && !is_sub_delim(c) && !is_gen_delim(c)) {
        return false;
      }
    }

    auto hash = url_end;
    for (auto i = scheme_end + 3; i < url_end; ++i) {
      if (input.data()[i] == '#') {
        hash = i;
        break;
      }
    }
    auto question = hash;
    for (auto i = scheme_end + 3; i < hash; ++i) {
      if (input.data()[i] == '?') {
        question = i;
        break;
      }
    }
    auto const base_end = question < hash ? question : hash;
    out->protocol       = {url_begin, scheme_end};
    if (question < hash) { out->query = {question + 1, hash}; }
    if (hash < url_end) { out->fragment = {hash + 1, url_end}; }

    auto const authority_begin = scheme_end + 3;
    auto authority_end         = base_end;
    for (auto i = authority_begin; i < base_end; ++i) {
      if (input.data()[i] == '/') {
        authority_end = i;
        break;
      }
    }
    out->path = {authority_end, base_end};

    auto host_begin = authority_begin;
    for (auto i = authority_begin; i < authority_end; ++i) {
      if (input.data()[i] == '@') { host_begin = i + 1; }
    }
    if (host_begin < authority_end && input.data()[host_begin] == '[') {
      auto close = authority_end;
      for (auto i = host_begin + 1; i < authority_end; ++i) {
        if (input.data()[i] == ']') {
          close = i;
          break;
        }
      }
      if (close == authority_end) { return false; }
      out->host = {host_begin, close + 1};
      if (close + 1 < authority_end) {
        if (input.data()[close + 1] != ':') { return false; }
        out->port = {close + 2, authority_end};
      }
    } else {
      auto colon = authority_end;
      for (auto i = host_begin; i < authority_end; ++i) {
        if (input.data()[i] == ':') { colon = i; }
      }
      out->host = {host_begin, colon};
      if (colon < authority_end) { out->port = {colon + 1, authority_end}; }
    }
    for (auto i = out->port.begin; i < out->port.end; ++i) {
      if (!is_digit(input.data()[i])) { return false; }
    }
    return true;
  };
)***";

// Builds the sizing UDF by inserting the shared parser into a self-contained device function.
std::string const url_component_sizes_udf = std::string{R"***(
// Computes exact output byte counts for the six URL component columns.
__device__ int compute_url_component_sizes(int32_t* protocol_size,
                                           int32_t* host_size,
                                           int32_t* port_size,
                                           int32_t* path_size,
                                           int32_t* query_size,
                                           int32_t* fragment_size,
                                           cudf::string_view input) {
  *protocol_size = *host_size = *port_size = 0;
  *path_size = *query_size = *fragment_size = 0;
)***"} + parse_url_udf + R"***(
  url_ranges ranges;
  if (!parse_url(&ranges)) { return 0; }
  *protocol_size = ranges.protocol.end - ranges.protocol.begin;
  *host_size     = ranges.host.end - ranges.host.begin;
  *port_size     = ranges.port.end - ranges.port.begin;
  *path_size     = ranges.path.end - ranges.path.begin;
  *query_size    = ranges.query.end - ranges.query.begin;
  *fragment_size = ranges.fragment.end - ranges.fragment.begin;
  return 0;
}
)***";

// Builds the output UDF from the same parser so both CUDA passes use identical ranges.
std::string const url_component_output_udf = std::string{R"***(
// Copies the six parsed URL components into their preallocated string buffers.
__device__ int write_url_components(cuda::std::span<char>* protocol,
                                    cuda::std::span<char>* host,
                                    cuda::std::span<char>* port,
                                    cuda::std::span<char>* path,
                                    cuda::std::span<char>* query,
                                    cuda::std::span<char>* fragment,
                                    cudf::string_view input) {
)***"} + parse_url_udf + R"***(
  url_ranges ranges;
  if (!parse_url(&ranges)) { return 0; }
  cuda::std::span<char>* outputs[] = {protocol, host, port, path, query, fragment};
  range32 components[]     = {
    ranges.protocol, ranges.host, ranges.port, ranges.path, ranges.query, ranges.fragment};
  for (auto component = 0; component < 6; ++component) {
    auto const range = components[component];
    auto const size  = range.end - range.begin;
    if (size > 0) { memcpy(outputs[component]->data(), input.data() + range.begin, size); }
  }
  return 0;
}
)***";

constexpr std::string_view usage =
  "usage: url_log_transforms INPUT.csv OUTPUT.csv <regex|precompiled|jit|lto> ROWS ITERATIONS\n"
  "       url_log_transforms <usage|--help>\n";

// Extracts RFC 3986-style hierarchical URI components from unstructured log lines.
[[nodiscard]] std::unique_ptr<cudf::table> run_regex(cudf::column_view input,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  // Derived from RFC 3986 Appendix B. The authority capture is expanded into optional
  // userinfo plus host and port, and Appendix C delimiters bound the URI within a log line.
  static auto program = cudf::strings::regex_program::create(
    R"((?:^|[^A-Za-z0-9+.-])([A-Za-z][A-Za-z0-9+.-]*):\/\/(?:[^@\/?# \t\n\r"<>]*@)?(\[[^\]\/?# \t\n\r"<>]*\]|[^\/:?# \t\n\r"<>]*)(?::([0-9]*))?([^?# \t\n\r"<>]*)(?:\?([^# \t\n\r"<>]*))?(?:#([^ \t\n\r"<>]*))?(?:$|[ \t\n\r"<>]))");
  auto extracted = cudf::strings::extract(cudf::strings_column_view{input}, *program, stream, mr);
  auto columns   = extracted->release();
  auto empty     = cudf::string_scalar{"", true, stream, mr};
  for (auto& column : columns) {
    column = cudf::replace_nulls(column->view(), empty, stream, mr);
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

// Decomposes key-value URL tokens using only precompiled libcudf string primitives.
[[nodiscard]] std::unique_ptr<cudf::table> run_precompiled(cudf::column_view input,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr)
{
  // Materialize the delimiters used by each partitioning stage.
  auto empty            = cudf::string_scalar{"", true, stream, mr};
  auto scheme_separator = cudf::string_scalar{"://", true, stream, mr};
  auto marker_separator = cudf::string_scalar{"=", true, stream, mr};
  auto token_separator  = cudf::string_scalar{" ", true, stream, mr};
  auto hash             = cudf::string_scalar{"#", true, stream, mr};
  auto question         = cudf::string_scalar{"?", true, stream, mr};
  auto slash            = cudf::string_scalar{"/", true, stream, mr};
  auto at               = cudf::string_scalar{"@", true, stream, mr};
  auto right_bracket    = cudf::string_scalar{"]", true, stream, mr};
  auto left_bracket     = cudf::string_scalar{"[", true, stream, mr};
  auto colon            = cudf::string_scalar{":", true, stream, mr};

  // Mark rows containing an authority-style URI and split at the first "://".
  auto has_url =
    cudf::strings::contains(cudf::strings_column_view{input}, scheme_separator, stream, mr);
  auto scheme_table =
    cudf::strings::partition(cudf::strings_column_view{input}, scheme_separator, stream, mr);
  auto scheme_columns = scheme_table->release();

  // Extract the scheme from the key-value token immediately preceding "://".
  auto marker_table = cudf::strings::rpartition(
    cudf::strings_column_view{scheme_columns[0]->view()}, marker_separator, stream, mr);
  auto marker_columns = marker_table->release();

  // Stop at the first space so later log fields are excluded from the URI.
  auto token_table = cudf::strings::partition(
    cudf::strings_column_view{scheme_columns[2]->view()}, token_separator, stream, mr);
  auto token_columns = token_table->release();

  // Split off the fragment; everything after the first '#' belongs to it.
  auto fragment_table =
    cudf::strings::partition(cudf::strings_column_view{token_columns[0]->view()}, hash, stream, mr);
  auto fragment_columns = fragment_table->release();

  // Split the pre-fragment portion at the first '?' to isolate the query.
  auto query_table = cudf::strings::partition(
    cudf::strings_column_view{fragment_columns[0]->view()}, question, stream, mr);
  auto query_columns = query_table->release();

  // Split the remaining hierarchical part at its first slash into authority and path.
  auto authority_path_table = cudf::strings::partition(
    cudf::strings_column_view{query_columns[0]->view()}, slash, stream, mr);
  auto authority_path_columns = authority_path_table->release();

  // Reattach the slash delimiter to produce the RFC path value.
  auto path = cudf::strings::concatenate(
    cudf::table_view{{authority_path_columns[1]->view(), authority_path_columns[2]->view()}},
    empty,
    cudf::string_scalar{"", false, stream, mr},
    cudf::strings::separator_on_nulls::YES,
    stream,
    mr);

  // Remove optional userinfo by retaining everything after the authority's last '@'.
  auto has_userinfo = cudf::strings::contains(
    cudf::strings_column_view{authority_path_columns[0]->view()}, at, stream, mr);
  auto userinfo_table = cudf::strings::rpartition(
    cudf::strings_column_view{authority_path_columns[0]->view()}, at, stream, mr);
  auto userinfo_columns = userinfo_table->release();
  auto host_port        = cudf::copy_if_else(userinfo_columns[2]->view(),
                                      authority_path_columns[0]->view(),
                                      has_userinfo->view(),
                                      stream,
                                      mr);

  // Bracketed IP literals and regular hosts require different port splitting rules.
  auto is_ip_literal = cudf::strings::starts_with(
    cudf::strings_column_view{host_port->view()}, left_bracket, stream, mr);
  auto bracket_table = cudf::strings::partition(
    cudf::strings_column_view{host_port->view()}, right_bracket, stream, mr);
  auto bracket_columns = bracket_table->release();

  // Preserve both brackets as part of an IP-literal host.
  auto bracket_host = cudf::strings::concatenate(
    cudf::table_view{{bracket_columns[0]->view(), bracket_columns[1]->view()}},
    empty,
    cudf::string_scalar{"", false, stream, mr},
    cudf::strings::separator_on_nulls::YES,
    stream,
    mr);

  // For an IP literal, parse an optional port only after the closing bracket.
  auto bracket_port_table = cudf::strings::partition(
    cudf::strings_column_view{bracket_columns[2]->view()}, colon, stream, mr);
  auto bracket_port_columns = bracket_port_table->release();

  // For a regular authority, treat the final colon as the port separator.
  auto has_regular_port =
    cudf::strings::contains(cudf::strings_column_view{host_port->view()}, colon, stream, mr);
  auto regular_table =
    cudf::strings::rpartition(cudf::strings_column_view{host_port->view()}, colon, stream, mr);
  auto regular_columns = regular_table->release();
  auto regular_host    = cudf::copy_if_else(
    regular_columns[0]->view(), host_port->view(), has_regular_port->view(), stream, mr);
  auto regular_port =
    cudf::copy_if_else(regular_columns[2]->view(), empty, has_regular_port->view(), stream, mr);

  // Select the bracketed or regular host/port result for each row.
  auto host = cudf::copy_if_else(
    bracket_host->view(), regular_host->view(), is_ip_literal->view(), stream, mr);
  auto port = cudf::copy_if_else(
    bracket_port_columns[2]->view(), regular_port->view(), is_ip_literal->view(), stream, mr);

  // Convert missing components to empty strings and blank rows without a URL.
  auto normalize = [&](cudf::column_view column) {
    auto no_nulls = cudf::replace_nulls(column, empty, stream, mr);
    return cudf::copy_if_else(no_nulls->view(), empty, has_url->view(), stream, mr);
  };

  // Return the six columns in the same order used by the regex and CUDA implementations.
  std::vector<std::unique_ptr<cudf::column>> result;
  result.reserve(output_count);
  result.push_back(normalize(marker_columns[2]->view()));
  result.push_back(normalize(host->view()));
  result.push_back(normalize(port->view()));
  result.push_back(normalize(path->view()));
  result.push_back(normalize(query_columns[2]->view()));
  result.push_back(normalize(fragment_columns[2]->view()));
  return std::make_unique<cudf::table>(std::move(result));
}

// Runs either the runtime-compiled CUDA-string UDFs or their AOT fatbin/LTO counterparts.
[[nodiscard]] std::unique_ptr<cudf::table> run_jit(cudf::column_view input,
                                                   bool use_lto,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  cudf::transform_output const size_spec{cudf::data_type{cudf::type_id::INT32},
                                         cudf::output_nullability::ALL_VALID};
  std::vector<cudf::transform_output> const size_outputs(output_count, size_spec);
  cudf::transform_input inputs[] = {input};
  std::unique_ptr<cudf::table> sizes;

  if (use_lto) {
    auto range    = url_log_fragments::file_ranges[url_log_fragments::url_component_sizes];
    auto fragment = url_log_fragments::files.subspan(range[0], range[1]);
    sizes         = cudf::transform_lto(fragment,
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
    sizes = cudf::multi_transform(url_component_sizes_udf,
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

  std::vector<std::unique_ptr<cudf::column>> offsets;
  offsets.reserve(output_count);
  for (auto& string_sizes : sizes->view()) {
    auto run_ends = cudf::scan(string_sizes,
                               *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                               cudf::scan_type::INCLUSIVE,
                               cudf::null_policy::EXCLUDE,
                               stream,
                               mr);
    auto zero     = cudf::numeric_scalar<int32_t>{0, true, stream, mr};
    auto first    = cudf::make_column_from_scalar(zero, 1, stream, mr);
    offsets.push_back(cudf::concatenate(
      std::vector<cudf::column_view>{first->view(), run_ends->view()}, stream, mr));
  }

  cudf::transform_output const output_spec{cudf::data_type{cudf::type_id::STRING},
                                           cudf::output_nullability::ALL_VALID};
  std::vector<cudf::transform_output> const outputs(output_count, output_spec);
  if (use_lto) {
    auto range    = url_log_fragments::file_ranges[url_log_fragments::url_component_output];
    auto fragment = url_log_fragments::files.subspan(range[0], range[1]);
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
  return cudf::multi_transform(url_component_output_udf,
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
      throw std::invalid_argument("invalid arguments; run url_log_transforms --help for usage");
    }

    auto input_path     = std::string{argv[1]};
    auto output_path    = std::string{argv[2]};
    auto implementation = std::string_view{argv[3]};
    if (implementation != "regex" && implementation != "precompiled" && implementation != "jit" &&
        implementation != "lto") {
      throw std::invalid_argument("variant must be regex, precompiled, jit, or lto");
    }
    auto requested_rows = std::stoll(argv[4]);
    auto iterations     = std::stoi(argv[5]);
    if (requested_rows < 0 || requested_rows > std::numeric_limits<cudf::size_type>::max()) {
      throw std::invalid_argument("ROWS is outside the cudf::size_type range");
    }
    if (iterations < 1) { throw std::invalid_argument("ITERATIONS must be positive"); }

    auto rows         = static_cast<cudf::size_type>(requested_rows);
    auto use_lto      = implementation == "lto";
    auto stream       = cudf::get_default_stream();
    auto mr           = cudf::get_current_device_resource_ref();
    auto read_options = cudf::io::csv_reader_options::builder(cudf::io::source_info{input_path})
                          .header(0)
                          .use_cols_names({"LogLine"})
                          .build();
    auto input = cudf::io::read_csv(read_options).tbl;
    if (rows != input->num_rows()) {
      input = cudf::sample(input->view(), rows, cudf::sample_with_replacement::TRUE);
    }

    auto input_bytes = input->get_column(0).alloc_size();
    auto input_view  = input->get_column(0).view();
    rmm::mr::statistics_resource_adaptor stats{mr};
    auto stats_mr      = rmm::device_async_resource_ref{stats};
    auto run_transform = [&]() {
      if (implementation == "regex") { return run_regex(input_view, stream, stats_mr); }
      if (implementation == "precompiled") { return run_precompiled(input_view, stream, stats_mr); }
      return run_jit(input_view, use_lto, stream, stats_mr);
    };

    stream.synchronize();
    auto cold_start = std::chrono::steady_clock::now();
    nvtxRangePush("url_log_cold");
    auto cold_result = run_transform();
    stream.synchronize();
    nvtxRangePop();
    auto cold_seconds =
      std::chrono::duration<double>{std::chrono::steady_clock::now() - cold_start}.count();
    cold_result.reset();

    std::unique_ptr<cudf::table> result;
    auto warm_start = std::chrono::steady_clock::now();
    nvtxRangePush("url_log_warm");
    for (auto i = 0; i < iterations; ++i) {
      result.reset();
      result = run_transform();
    }
    stream.synchronize();
    nvtxRangePop();
    auto warm_seconds =
      std::chrono::duration<double>{std::chrono::steady_clock::now() - warm_start}.count() /
      iterations;

    if (output_path != "-") {
      auto write_options =
        cudf::io::csv_writer_options::builder(cudf::io::sink_info{output_path}, result->view())
          .include_header(true)
          .names({"protocol", "host", "port", "path", "query", "fragment"})
          .build();
      cudf::io::write_csv(write_options);
    }

    auto bytes        = stats.get_bytes_counter();
    auto output_bytes = result->alloc_size();
    auto gib          = static_cast<double>(input_bytes + output_bytes) / (1ULL << 30);
    std::cout << std::format(
      "variant={}\nrows={}\ncold_seconds={}\nwarm_seconds={}\nrows_per_second={}\neffective_gib_"
      "per_second={}\ninput_bytes={}\noutput_bytes={}\npeak_memory_bytes={}\ntotal_allocated_bytes="
      "{}\nallocated_bytes_per_call={}\n",
      implementation,
      rows,
      cold_seconds,
      warm_seconds,
      static_cast<double>(rows) / warm_seconds,
      gib / warm_seconds,
      input_bytes,
      output_bytes,
      bytes.peak,
      bytes.total,
      bytes.total / static_cast<std::size_t>(iterations + 1));
    return EXIT_SUCCESS;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
