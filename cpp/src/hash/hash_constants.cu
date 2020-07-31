/* Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "hash_constants.hpp"

#include <strings/utilities.cuh>

namespace cudf {
namespace detail {

const md5_shift_constants_type g_md5_shift_constants[] = {
  7,
  12,
  17,
  22,
  5,
  9,
  14,
  20,
  4,
  11,
  16,
  23,
  6,
  10,
  15,
  21,
};

const md5_hash_constants_type g_md5_hash_constants[] = {
  0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
  0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
  0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
  0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
  0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
  0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
  0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
  0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
};

namespace {
__device__ md5_hash_constants_type md5_hash_constants[sizeof(g_md5_hash_constants)];
__device__ md5_shift_constants_type md5_shift_constants[sizeof(g_md5_shift_constants)];

strings::detail::thread_safe_per_context_cache<md5_hash_constants_type> d_md5_hash_constants;
strings::detail::thread_safe_per_context_cache<md5_shift_constants_type> d_md5_shift_constants;
}  // namespace

/**
 * @copydoc cudf::detail::get_md5_hash_constants
 */
const md5_hash_constants_type* get_md5_hash_constants()
{
  return d_md5_hash_constants.find_or_initialize([&](void) {
    md5_hash_constants_type* table = nullptr;
    CUDA_TRY(
      cudaMemcpyToSymbol(md5_hash_constants, g_md5_hash_constants, sizeof(g_md5_hash_constants)));
    CUDA_TRY(cudaGetSymbolAddress((void**)&table, md5_hash_constants));
    return table;
  });
}

/**
 * @copydoc cudf::detail::get_md5_shift_constants
 */
const md5_shift_constants_type* get_md5_shift_constants()
{
  return d_md5_shift_constants.find_or_initialize([&](void) {
    md5_shift_constants_type* table = nullptr;
    CUDA_TRY(cudaMemcpyToSymbol(
      md5_shift_constants, g_md5_shift_constants, sizeof(g_md5_shift_constants)));
    CUDA_TRY(cudaGetSymbolAddress((void**)&table, md5_shift_constants));
    return table;
  });
}

}  // namespace detail
}  // namespace cudf
