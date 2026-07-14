/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/strings/string_view.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/span>

// Runtime JIT compilation consumes CUDA source strings. Each operation has one UDF that computes
// exact output sizes and another that writes into the resulting character buffers.
__device__ void compute_request_line_sizes(int32_t* method_size,
                                           int32_t* path_size,
                                           int32_t* version_size,
                                           cudf::string_view input)
{
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
  auto path_end = query_begin < target_end ? query_begin : target_end;

  *method_size  = method_end;
  *path_size    = path_end - method_end - 1;
  *version_size = input.size_bytes() - target_end - 6;
}

// Each span points at the final character buffer for one output string in this row. Its size came
// from compute_request_line_sizes, so this pass only copies bytes and performs no allocation.
__device__ void write_request_line(cuda::std::span<char>* method,
                                   cuda::std::span<char>* path,
                                   cuda::std::span<char>* version,
                                   cudf::string_view input)
{
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

#ifdef UDF_COMPUTE_SIZES
extern "C" __device__ auto* transform = &compute_request_line_sizes;
#else
#ifdef UDF_WRITE_OUTPUT
extern "C" __device__ auto* transform = &write_request_line;
#else
#error "Must define either UDF_COMPUTE_SIZES or UDF_WRITE_OUTPUT"
#endif
#endif
