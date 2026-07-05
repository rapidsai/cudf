/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" __device__ unsigned long long regex_ir_test_replace(char const*,
                                                               unsigned long long,
                                                               char*);

extern "C" __global__ void regex_ir_replace_kernel(char const* data,
                                                   unsigned long long size,
                                                   char* output,
                                                   unsigned long long* result_size)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *result_size = regex_ir_test_replace(data, size, output);
  }
}
