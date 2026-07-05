/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "http_log_udf.cuh"

extern "C" __device__ int transform(int32_t* client_ip_size,
                                    int32_t* timestamp_size,
                                    int32_t* method_size,
                                    int32_t* path_size,
                                    int32_t* status_size,
                                    int32_t* referer_size,
                                    int32_t* user_agent_size,
                                    cudf::string_view input)
{
  auto const fields = http_log_udf::parse_combined_log(input);
  *client_ip_size   = fields.client_ip.size();
  *timestamp_size   = fields.timestamp.size();
  *method_size      = fields.method.size();
  *path_size        = fields.path.size();
  *status_size      = fields.status.size();
  *referer_size     = fields.referer.size();
  *user_agent_size  = fields.user_agent.size();
  return 0;
}
