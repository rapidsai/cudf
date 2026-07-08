/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "udf.cuh"

// The host supplies spans backed by the final strings-column character buffers. Their lengths were
// computed by the matching sizing fragment.
extern "C" __device__ int transform(cuda::std::span<char>* method,
                                    cuda::std::span<char>* path,
                                    cuda::std::span<char>* version,
                                    cudf::string_view input)
{
  auto const fields = http_log_udf::parse_request_line(input);
  http_log_udf::copy_field(*method, input, fields.method);
  http_log_udf::copy_field(*path, input, fields.path);
  http_log_udf::copy_field(*version, input, fields.version);
  return 0;
}
