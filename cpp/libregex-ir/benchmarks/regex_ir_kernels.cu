/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(REGEX_IR_ENUMERATION_KERNEL)

extern "C" __device__ bool regex_ir_benchmark_find(char const*,
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

extern "C" __global__ void regex_ir_benchmark_enumerate(char const* chars,
                                                        int const* offsets,
                                                        int rows,
                                                        unsigned long long* counts,
                                                        unsigned long long* first_captures,
                                                        unsigned int capture_count,
                                                        unsigned int match_limit)
{
  auto row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= rows) return;

  auto begin = offsets[row];
  auto size  = static_cast<unsigned long long>(offsets[row + 1] - begin);
  auto data  = chars + begin;
  unsigned long long local[10];
  unsigned long long search = 0;
  unsigned long long count  = 0;
  auto slots                = static_cast<unsigned int>((capture_count + 1U) * 2U);
  while (search <= size) {
    for (unsigned int slot = 0; slot < slots; ++slot) {
      local[slot] = ~0ULL;
    }
    if (!regex_ir_benchmark_find(data, size, search, local)) break;
    if (count == 0 && first_captures != nullptr) {
      for (unsigned int slot = 0; slot < slots; ++slot) {
        first_captures[static_cast<unsigned long long>(row) * slots + slot] = local[slot];
      }
    }
    ++count;
    if (count == match_limit) break;
    search = local[1] != local[0] ? local[1] : next_character(data, size, local[1]);
  }
  counts[row] = count;
}

#else

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

#endif
