/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "udf.cuh"

extern "C" __device__ int transform(cuda::std::span<char>* client_ip,
                                    cuda::std::span<char>* timestamp,
                                    cuda::std::span<char>* method,
                                    cuda::std::span<char>* path,
                                    cuda::std::span<char>* status,
                                    cuda::std::span<char>* referer,
                                    cuda::std::span<char>* user_agent,
                                    cudf::string_view input)
{
  auto const fields = http_log_udf::parse_combined_log(input);
  http_log_udf::copy_field(*client_ip, input, fields.client_ip);
  http_log_udf::copy_field(*timestamp, input, fields.timestamp);
  http_log_udf::copy_field(*method, input, fields.method);
  http_log_udf::copy_field(*path, input, fields.path);
  http_log_udf::copy_field(*status, input, fields.status);
  http_log_udf::copy_field(*referer, input, fields.referer);
  http_log_udf::copy_field(*user_agent, input, fields.user_agent);
  return 0;
}
