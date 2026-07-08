/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "udf.cuh"

// AOT fragments expose an unmangled `transform` symbol so transform_lto can link them with the
// precompiled libcudf kernel. Output pointers precede the per-row input in the transform ABI.
extern "C" __device__ int transform(int32_t* method_size,
                                    int32_t* path_size,
                                    int32_t* version_size,
                                    cudf::string_view input)
{
  auto const fields = http_log_udf::parse_request_line(input);
  *method_size      = fields.method.size();
  *path_size        = fields.path.size();
  *version_size     = fields.version.size();
  return 0;
}
