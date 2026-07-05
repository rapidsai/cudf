/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" __device__ unsigned long long regex_ir_benchmark_replace(char const*,
                                                                    unsigned long long,
                                                                    char*);

extern "C" __global__ void regex_ir_benchmark_replace_size(char const* chars,
                                                           int const* offsets,
                                                           int rows,
                                                           int* sizes)
{
  auto row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row > rows) return;
  if (row == rows) {
    sizes[row] = 0;
    return;
  }
  auto begin = offsets[row];
  auto size  = static_cast<unsigned long long>(offsets[row + 1] - begin);
  sizes[row] = static_cast<int>(regex_ir_benchmark_replace(chars + begin, size, nullptr));
}

extern "C" __global__ void regex_ir_benchmark_replace_emit(
  char const* chars, int const* offsets, int rows, int const* output_offsets, char* output)
{
  auto row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= rows) return;
  auto begin = offsets[row];
  auto size  = static_cast<unsigned long long>(offsets[row + 1] - begin);
  static_cast<void>(regex_ir_benchmark_replace(chars + begin, size, output + output_offsets[row]));
}
