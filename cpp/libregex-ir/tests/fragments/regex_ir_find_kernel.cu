/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" __device__ bool regex_ir_test_find(char const*, unsigned long long, unsigned long long*);

extern "C" __global__ void regex_ir_find_kernel(char const* data,
                                                unsigned long long size,
                                                unsigned long long* span,
                                                int* result)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *result = regex_ir_test_find(data, size, span) ? 1 : 0;
  }
}
