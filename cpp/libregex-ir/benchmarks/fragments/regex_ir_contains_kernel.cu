/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" __device__ bool regex_ir_benchmark_execute(char const*, unsigned long long);

extern "C" __global__ void regex_ir_benchmark_contains(char const* chars,
                                                       int const* offsets,
                                                       int rows,
                                                       unsigned char* output)
{
  auto row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= rows) return;
  auto begin = offsets[row];
  auto end   = offsets[row + 1];
  output[row] =
    regex_ir_benchmark_execute(chars + begin, static_cast<unsigned long long>(end - begin));
}
