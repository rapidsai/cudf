/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>

#include <cuda/std/cstdint>
#include <cuda/std/span>

__device__ int32_t find_character(cudf::string_view input, int32_t begin, char needle)
{
  return cuda::std::find(input.data() + begin, input.data() + input.size_bytes(), needle) -
         input.data();
}

__device__ int compute_request_line_sizes(int32_t* method_size,
                                          int32_t* path_size,
                                          int32_t* version_size,
                                          cudf::string_view input)
{
  // Initialize output sizes to zero in case of early return.
  *method_size  = 0;
  *path_size    = 0;
  *version_size = 0;

  auto n = input.size_bytes();

  auto method_end = find_character(input, 0, ' ');
  if (method_end == n) { return 0; }

  auto target_end = find_character(input, method_end + 1, ' ');
  if (target_end == n || n - target_end < 6) { return 0; }
  if (input.data()[target_end + 1] != 'H' || input.data()[target_end + 2] != 'T' ||
      input.data()[target_end + 3] != 'T' || input.data()[target_end + 4] != 'P' ||
      input.data()[target_end + 5] != '/') {
    return 0;
  }

  auto query_begin = find_character(input, method_end + 1, '?');

  // The path ends at the query or the target, whichever comes first.
  auto path_end = query_begin < target_end ? query_begin : target_end;

  *method_size  = method_end;
  *path_size    = path_end - method_end - 1;
  *version_size = n - target_end - 6;

  // return 0 to indicate success
  return 0;
}

// Each span points at the final character buffer for one output string in this row. Its size came
// from compute_request_line_sizes, so this pass only copies bytes and performs no allocation.
__device__ int write_request_line(cuda::std::span<char>* method,
                                  cuda::std::span<char>* path,
                                  cuda::std::span<char>* version,
                                  cudf::string_view input)
{
  auto n = input.size_bytes();

  auto method_end = find_character(input, 0, ' ');
  if (method_end == n) { return 0; }

  auto target_end = find_character(input, method_end + 1, ' ');
  if (target_end == n || n - target_end < 6) { return 0; }
  if (input.data()[target_end + 1] != 'H' || input.data()[target_end + 2] != 'T' ||
      input.data()[target_end + 3] != 'T' || input.data()[target_end + 4] != 'P' ||
      input.data()[target_end + 5] != '/') {
    return 0;
  }

  auto query_begin = find_character(input, method_end + 1, '?');

  // The path ends at the query or the target, whichever comes first.
  auto path_end = query_begin < target_end ? query_begin : target_end;

  memcpy(method->data(), input.data(), method_end);
  memcpy(path->data(), input.data() + method_end + 1, path_end - (method_end + 1));
  memcpy(version->data(), input.data() + target_end + 6, n - (target_end + 6));

  // return 0 to indicate success
  return 0;
}

// The symbol `transform` is the UDF entry point for `cudf::transform_lto`.
#ifdef UDF_COMPUTE_SIZES
extern "C" __device__ int transform(int32_t* method_size,
                                    int32_t* path_size,
                                    int32_t* version_size,
                                    cudf::string_view input)
{
  return compute_request_line_sizes(method_size, path_size, version_size, input);
}
#else
#ifdef UDF_WRITE_OUTPUT
extern "C" __device__ int transform(cuda::std::span<char>* method,
                                    cuda::std::span<char>* path,
                                    cuda::std::span<char>* version,
                                    cudf::string_view input)
{
  return write_request_line(method, path, version, input);
}
#else
#error "Must define either UDF_COMPUTE_SIZES or UDF_WRITE_OUTPUT"
#endif
#endif
