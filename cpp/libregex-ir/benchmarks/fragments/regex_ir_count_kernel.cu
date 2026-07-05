/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" __device__ unsigned long long regex_ir_benchmark_count(char const*, unsigned long long);

extern "C" __global__ void regex_ir_benchmark_count_kernel(char const* chars,
                                                           int const* offsets,
                                                           int rows,
                                                           int* counts)
{
  auto row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= rows) return;
  auto begin  = offsets[row];
  auto length = static_cast<unsigned long long>(offsets[row + 1] - begin);
  counts[row] = static_cast<int>(regex_ir_benchmark_count(chars + begin, length));
}
