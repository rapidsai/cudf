/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" __device__ unsigned long long regex_ir_benchmark_split(char const*,
                                                                  unsigned long long,
                                                                  unsigned long long*);

struct regex_ir_string_pair {
  char const* data;
  int size;
};

extern "C" __global__ void regex_ir_benchmark_split_size(char const* chars,
                                                         int const* offsets,
                                                         int rows,
                                                         int* counts)
{
  auto row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row > rows) return;
  if (row == rows) {
    counts[row] = 0;
    return;
  }
  auto begin  = offsets[row];
  auto size   = static_cast<unsigned long long>(offsets[row + 1] - begin);
  counts[row] = static_cast<int>(regex_ir_benchmark_split(chars + begin, size, nullptr));
}

extern "C" __global__ void regex_ir_benchmark_split_emit(char const* chars,
                                                         int const* offsets,
                                                         int rows,
                                                         int const* field_offsets,
                                                         unsigned long long* spans,
                                                         regex_ir_string_pair* pairs)
{
  auto row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= rows) return;
  auto begin       = offsets[row];
  auto size        = static_cast<unsigned long long>(offsets[row + 1] - begin);
  auto first_field = field_offsets[row];
  auto row_spans   = spans + static_cast<unsigned long long>(first_field) * 2ULL;
  auto field_count = regex_ir_benchmark_split(chars + begin, size, row_spans);
  for (unsigned long long field = 0; field < field_count; ++field) {
    auto field_begin                                            = row_spans[field * 2ULL];
    auto field_end                                              = row_spans[field * 2ULL + 1ULL];
    pairs[static_cast<unsigned long long>(first_field) + field] = {
      chars + begin + field_begin, static_cast<int>(field_end - field_begin)};
  }
}
