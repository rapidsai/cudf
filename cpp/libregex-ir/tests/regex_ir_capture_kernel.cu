/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" __device__ bool regex_ir_test_find_execute(char const*,
                                                      unsigned long long,
                                                      unsigned long long,
                                                      unsigned long long*);

__device__ unsigned long long next_character(char const* data,
                                             unsigned long long size,
                                             unsigned long long position)
{
  if (position >= size) return size + 1;
  auto first = static_cast<unsigned char>(data[position]);
  auto width = first < 0x80U ? 1ULL : (first < 0xE0U ? 2ULL : (first < 0xF0U ? 3ULL : 4ULL));
  return position + width <= size ? position + width : position + 1;
}

extern "C" __global__ void regex_ir_capture_kernel(char const* data,
                                                   unsigned long long const* offsets,
                                                   unsigned long long row_count,
                                                   unsigned long long capture_slots,
                                                   unsigned long long max_matches,
                                                   unsigned long long* captures,
                                                   unsigned long long* counts,
                                                   int* overflows)
{
  auto row = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= row_count) return;

  auto begin                      = offsets[row];
  auto size                       = offsets[row + 1] - begin;
  auto row_data                   = data + begin;
  unsigned long long search_start = 0;
  unsigned long long count        = 0;
  while (search_start <= size) {
    if (count == max_matches) {
      overflows[row] = 1;
      break;
    }
    auto match_captures = captures + (row * max_matches + count) * capture_slots;
    if (!regex_ir_test_find_execute(row_data, size, search_start, match_captures)) break;
    auto match_begin = match_captures[0];
    auto match_end   = match_captures[1];
    ++count;
    search_start = match_end != match_begin ? match_end : next_character(row_data, size, match_end);
  }
  counts[row] = count;
}
