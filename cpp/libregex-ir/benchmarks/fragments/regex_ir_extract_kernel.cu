/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" __device__ bool regex_ir_benchmark_extract(char const*,
                                                      unsigned long long,
                                                      unsigned long long,
                                                      unsigned long long*);

struct regex_ir_string_pair {
  char const* data;
  int size;
};

extern "C" __global__ void regex_ir_benchmark_extract_kernel(char const* chars,
                                                             int const* offsets,
                                                             int rows,
                                                             unsigned long long* first_captures,
                                                             regex_ir_string_pair* pairs,
                                                             unsigned int capture_count)
{
  auto row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= rows) return;

  auto begin        = offsets[row];
  auto size         = static_cast<unsigned long long>(offsets[row + 1] - begin);
  auto data         = chars + begin;
  auto slots        = static_cast<unsigned int>((capture_count + 1U) * 2U);
  auto row_captures = first_captures + static_cast<unsigned long long>(row) * slots;
  auto matched      = regex_ir_benchmark_extract(data, size, 0, row_captures);
  for (unsigned int capture = 0; capture < capture_count; ++capture) {
    auto slot  = static_cast<unsigned int>((capture + 1U) * 2U);
    auto begin = row_captures[slot];
    auto end   = row_captures[slot + 1U];
    auto index = static_cast<unsigned long long>(capture) * static_cast<unsigned long long>(rows) +
                 static_cast<unsigned long long>(row);
    if (matched && begin != ~0ULL && end != ~0ULL) {
      pairs[index] = {data + begin, static_cast<int>(end - begin)};
    } else {
      pairs[index] = {nullptr, 0};
    }
  }
}
