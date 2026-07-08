/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/strings/string_view.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/span>

namespace http_log_udf {

// A field is represented as a half-open byte range into the original log line. Keeping ranges
// avoids copying data while the parser discovers all output fields.
struct field_range {
  int32_t begin{};
  int32_t end{};

  [[nodiscard]] __device__ int32_t size() const { return end - begin; }
};

struct request_line_fields {
  field_range method;
  field_range path;
  field_range version;
};

struct combined_log_fields {
  field_range client_ip;
  field_range timestamp;
  field_range method;
  field_range path;
  field_range status;
  field_range referer;
  field_range user_agent;
};

[[nodiscard]] __device__ int32_t find_character(cudf::string_view const input,
                                                char const needle,
                                                int32_t begin)
{
  // These example formats are ASCII-delimited, so byte offsets are also valid character offsets.
  for (auto i = begin; i < input.size_bytes(); ++i) {
    if (input.data()[i] == needle) { return i; }
  }
  return input.size_bytes();
}

[[nodiscard]] __device__ request_line_fields parse_request_line(cudf::string_view const input)
{
  // Expected form: METHOD /path?optional-query HTTP/version
  auto const method_end              = find_character(input, ' ', 0);
  auto const target_end              = find_character(input, ' ', method_end + 1);
  auto const query_begin             = find_character(input, '?', method_end + 1);
  auto const path_end                = query_begin < target_end ? query_begin : target_end;
  constexpr int32_t http_prefix_size = 6;  // " HTTP/"

  return {{0, method_end},
          {method_end + 1, path_end},
          {target_end + http_prefix_size, input.size_bytes()}};
}

[[nodiscard]] __device__ combined_log_fields parse_combined_log(cudf::string_view const input)
{
  // Walk the delimiters once and retain only ranges for the seven fields emitted by the example.
  // The checked-in input is well-formed, so validation and malformed-row handling are intentionally
  // outside the scope of this transform demonstration.
  auto const ip_end           = find_character(input, ' ', 0);
  auto const timestamp_begin  = find_character(input, '[', ip_end) + 1;
  auto const timestamp_end    = find_character(input, ']', timestamp_begin);
  auto const request_begin    = find_character(input, '"', timestamp_end) + 1;
  auto const method_end       = find_character(input, ' ', request_begin);
  auto const target_end       = find_character(input, ' ', method_end + 1);
  auto const query_begin      = find_character(input, '?', method_end + 1);
  auto const path_end         = query_begin < target_end ? query_begin : target_end;
  auto const request_end      = find_character(input, '"', target_end);
  auto const status_begin     = request_end + 2;
  auto const status_end       = find_character(input, ' ', status_begin);
  auto const bytes_end        = find_character(input, ' ', status_end + 1);
  auto const referer_begin    = find_character(input, '"', bytes_end) + 1;
  auto const referer_end      = find_character(input, '"', referer_begin);
  auto const user_agent_begin = find_character(input, '"', referer_end + 1) + 1;
  auto const user_agent_end   = find_character(input, '"', user_agent_begin);

  return {{0, ip_end},
          {timestamp_begin, timestamp_end},
          {request_begin, method_end},
          {method_end + 1, path_end},
          {status_begin, status_end},
          {referer_begin, referer_end},
          {user_agent_begin, user_agent_end}};
}

__device__ void copy_field(cuda::std::span<char> output,
                           cudf::string_view const input,
                           field_range const field)
{
  // output is a view into the final chars child allocated from the sizing pass offsets.
  for (auto i = int32_t{0}; i < field.size(); ++i) {
    output[i] = input.data()[field.begin + i];
  }
}

}  // namespace http_log_udf
