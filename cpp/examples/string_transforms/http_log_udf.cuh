/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/strings/string_view.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/span>

namespace http_log_udf {

struct range {
  int32_t begin{};
  int32_t end{};

  [[nodiscard]] __device__ int32_t size() const { return end - begin; }
};

struct medium_fields {
  range method;
  range path;
  range version;
};

struct high_fields {
  range client_ip;
  range timestamp;
  range method;
  range path;
  range status;
  range referer;
  range user_agent;
};

[[nodiscard]] __device__ int32_t find(cudf::string_view const input,
                                      char const needle,
                                      int32_t begin)
{
  for (auto i = begin; i < input.size_bytes(); ++i) {
    if (input.data()[i] == needle) { return i; }
  }
  return input.size_bytes();
}

[[nodiscard]] __device__ medium_fields parse_medium(cudf::string_view const input)
{
  auto const method_end              = find(input, ' ', 0);
  auto const target_end              = find(input, ' ', method_end + 1);
  auto const query_begin             = find(input, '?', method_end + 1);
  auto const path_end                = query_begin < target_end ? query_begin : target_end;
  constexpr int32_t http_prefix_size = 6;  // " HTTP/"

  return {{0, method_end},
          {method_end + 1, path_end},
          {target_end + http_prefix_size, input.size_bytes()}};
}

[[nodiscard]] __device__ high_fields parse_high(cudf::string_view const input)
{
  auto const ip_end           = find(input, ' ', 0);
  auto const timestamp_begin  = find(input, '[', ip_end) + 1;
  auto const timestamp_end    = find(input, ']', timestamp_begin);
  auto const request_begin    = find(input, '"', timestamp_end) + 1;
  auto const method_end       = find(input, ' ', request_begin);
  auto const target_end       = find(input, ' ', method_end + 1);
  auto const query_begin      = find(input, '?', method_end + 1);
  auto const path_end         = query_begin < target_end ? query_begin : target_end;
  auto const request_end      = find(input, '"', target_end);
  auto const status_begin     = request_end + 2;
  auto const status_end       = find(input, ' ', status_begin);
  auto const bytes_end        = find(input, ' ', status_end + 1);
  auto const referer_begin    = find(input, '"', bytes_end) + 1;
  auto const referer_end      = find(input, '"', referer_begin);
  auto const user_agent_begin = find(input, '"', referer_end + 1) + 1;
  auto const user_agent_end   = find(input, '"', user_agent_begin);

  return {{0, ip_end},
          {timestamp_begin, timestamp_end},
          {request_begin, method_end},
          {method_end + 1, path_end},
          {status_begin, status_end},
          {referer_begin, referer_end},
          {user_agent_begin, user_agent_end}};
}

__device__ void copy_range(cuda::std::span<char> output,
                           cudf::string_view const input,
                           range const field)
{
  for (auto i = int32_t{0}; i < field.size(); ++i) {
    output[i] = input.data()[field.begin + i];
  }
}

}  // namespace http_log_udf
