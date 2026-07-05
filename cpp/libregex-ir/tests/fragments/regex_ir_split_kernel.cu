/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" __device__ unsigned long long regex_ir_test_split(char const*,
                                                             unsigned long long,
                                                             unsigned long long*);

extern "C" __global__ void regex_ir_split_kernel(char const* data,
                                                 unsigned long long size,
                                                 unsigned long long* spans,
                                                 unsigned long long* field_count)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *field_count = regex_ir_test_split(data, size, spans);
  }
}
