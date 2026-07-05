/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" __device__ unsigned long long regex_ir_test_count(char const*, unsigned long long);

extern "C" __global__ void regex_ir_count_kernel(char const* data,
                                                 unsigned long long size,
                                                 unsigned long long* result)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) { *result = regex_ir_test_count(data, size); }
}
