/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.
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

#ifndef CONVERSION_FUNCTIONS_CUH
#define CONVERSION_FUNCTIONS_CUH

#include "datetime_parser.cuh"
#include "utilities/wrapper_types.hpp"
#include <cuda_runtime_api.h>

#include "utilities/trie.cuh"

/**---------------------------------------------------------------------------*
 * @brief Checks whether the given character is a whitespace character.
 * 
 * @param[in] ch The character to check
 * 
 * @return True if the input is whitespace, False otherwise
 *---------------------------------------------------------------------------**/
__inline__ __device__ bool isWhitespace(char ch) {
  return ch == '\t' || ch == ' ';
}

/**---------------------------------------------------------------------------*
 * @brief Scans a character stream within a range, and adjusts the start and end
 * indices of the range to ignore whitespace and quotation characters.
 * 
 * @param[in] data The character stream to scan
 * @param[in,out] start The start index to adjust
 * @param[in,out] end The end index to adjust
 * @param[in] quotechar The character used to denote quotes
 * 
 * @return Adjusted or unchanged start_idx and end_idx
 *---------------------------------------------------------------------------**/
__device__ void adjustForWhitespaceAndQuotes(const char* data, long* start,
                                             long* end, char quotechar = '\0') {
  while ((*start < *end) && isWhitespace(data[*start])) {
    (*start)++;
  }
  if ((*start < *end) && data[*start] == quotechar) {
    (*start)++;
  }
  while ((*start <= *end) && isWhitespace(data[*end])) {
    (*end)--;
  }
  if ((*start <= *end) && data[*end] == quotechar) {
    (*end)--;
  }
}

/**---------------------------------------------------------------------------*
 * @brief Computes a 32-bit hash when given a byte stream and range.
 * 
 * MurmurHash3_32 implementation from
 * https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 *
 * MurmurHash3 was written by Austin Appleby, and is placed in the public
 * domain. The author hereby disclaims copyright to this source code.
 * Note - The x86 and x64 versions do _not_ produce the same results, as the
 * algorithms are optimized for their respective platforms. You can still
 * compile and run any of them on any platform, but your performance with the
 * non-native version will be less than optimal.
 * 
 * This is a modified version of what is used for hash-join. The change is at
 * accept a char * key and range (start and end) so that the large raw CSV data
 * pointer could be used
 * 
 * @param[in] key The input data to hash
 * @param[in] start The start index of the input data
 * @param[in] end The end index of the input data
 * @param[in] seed An initialization value
 * 
 * @return The hash value
 *---------------------------------------------------------------------------**/
__device__ int32_t convertStrToHash(const char* key, long start, long end,
                                    uint32_t seed) {
  auto getblock32 = [] __device__(const uint32_t* p, int i) -> uint32_t {
    // Individual byte reads for possible unaligned accesses
    auto q = (const uint8_t*)(p + i);
    return q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
  };

  auto rotl32 = [] __device__(uint32_t x, int8_t r) -> uint32_t {
    return (x << r) | (x >> (32 - r));
  };

  auto fmix32 = [] __device__(uint32_t h) -> uint32_t {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  };

  const int len = (end - start);
  const uint8_t* const data = (const uint8_t*)(key + start);
  const int nblocks = len / 4;
  uint32_t h1 = seed;
  constexpr uint32_t c1 = 0xcc9e2d51;
  constexpr uint32_t c2 = 0x1b873593;
  //----------
  // body
  const uint32_t* const blocks = (const uint32_t*)(data + nblocks * 4);
  for (int i = -nblocks; i; i++) {
    uint32_t k1 = getblock32(blocks, i);
    k1 *= c1;
    k1 = rotl32(k1, 15);
    k1 *= c2;
    h1 ^= k1;
    h1 = rotl32(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;
  }
  //----------
  // tail
  const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
  uint32_t k1 = 0;
  switch (len & 3) {
    case 3:
      k1 ^= tail[2] << 16;
    case 2:
      k1 ^= tail[1] << 8;
    case 1:
      k1 ^= tail[0];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
  };
  //----------
  // finalization
  h1 ^= len;
  h1 = fmix32(h1);
  return h1;
}

/**---------------------------------------------------------------------------*
 * @brief Structure for holding various options used when parsing and
 * converting CSV data to cuDF data type values.
 *---------------------------------------------------------------------------**/
struct ParseOptions {
  char delimiter;
  char terminator;
  char quotechar;
  char decimal;
  char thousands;
  char comment;
  bool keepquotes;
  bool doublequote;
  bool dayfirst;
  bool skipblanklines;
  SerialTrieNode* trueValuesTrie;
  SerialTrieNode* falseValuesTrie;
  SerialTrieNode* naValuesTrie;
  bool multi_delimiter;
};

/**
* @brief Specialization of determineBase for integral types. Checks if the
* string represents a hex value and updates the starting position if it does.
*/
template <typename T,
          typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
__device__ __forceinline__ int determineBase(const char* data, long* start,
                                             long end) {
  // check if this is a hex number
  if (end - *start >= 2 && data[*start] == '0' && data[*start + 1] == 'x') {
    *start += 2;
    return 16;
  }
  return 10;
}

/**
 * @brief Specialization of determineBase for non-integral numeric types.
 * Always returns 10, only decimal floating-point numbers are supported.
 */
template <typename T,
          typename std::enable_if_t<!std::is_integral<T>::value>* = nullptr>
__device__ __forceinline__ int determineBase(const char* data, long* start,
                                             long end) {
  return 10;
}

/**
 * @brief Specialization of decodeAsciiDigit for integral types.
 * Handles hexadecimal digits, both uppercase and lowercase.
 */
template <typename T,
          typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
__device__ __forceinline__ char decodeAsciiDigit(char d, int base) {
  if (base == 16) {
    if (d >= 'a' && d <= 'f') return d - 'a' + 10;
    if (d >= 'A' && d <= 'F') return d - 'A' + 10;
  }
  return d - '0';
}

/**
 * @brief Specialization of decodeAsciiDigit for non-integral numeric types.
 * Only handles decimal digits.
 */
template <typename T,
          typename std::enable_if_t<!std::is_integral<T>::value>* = nullptr>
__device__ __forceinline__ char decodeAsciiDigit(char d, int base) {
  return d - '0';
}

/**---------------------------------------------------------------------------*
 * @brief Default function for extracting a data value from a character string.
 * Handles all arithmetic data types; other data types are handled in
 * specialized template functions.
 *
 * @param[in] data The character string for parse
 * @param[in] start The index within data to start parsing from
 * @param[in] end The end index within data to end parsing
 * @param[in] opts The various parsing behavior options and settings
 *
 * @return The parsed and converted value
 *---------------------------------------------------------------------------**/
template <typename T>
__device__ T convertStrToValue(const char* data, long start, long end,
                               const ParseOptions& opts) {
  T value = 0;

  // Handle negative values if necessary
  int32_t sign = 1;
  if (data[start] == '-') {
    sign = -1;
    start++;
  }

  const int base = determineBase<T>(data, &start, end);

  // Handle the whole part of the number
  long index = start;
  while (index <= end) {
    if (data[index] == opts.decimal) {
      ++index;
      break;
    } else if (base == 10 && 
        (data[index] == 'e' || data[index] == 'E')) {
      break;
    } else if (data[index] != opts.thousands) {
      value = (value * base) + decodeAsciiDigit<T>(data[index], base);
    }
    ++index;
  }

  if (std::is_floating_point<T>::value) {
    // Handle fractional part of the number if necessary
    double divisor = 1;
    while (index <= end) {
      if (data[index] == 'e' || data[index] == 'E') {
        ++index;
        break;
      } else if (data[index] != opts.thousands) {
        divisor /= base;
        value += decodeAsciiDigit<T>(data[index], base) * divisor;
      }
      ++index;
    }

    // Handle exponential part of the number if necessary
    int32_t exponent = 0;
    int32_t exponentsign = 1;
    while (index <= end) {
      if (data[index] == '-') {
        exponentsign = -1;
      } else if (data[index] == '+') {
        exponentsign = 1;
      } else {
        exponent = (exponent * 10) + (data[index] - '0');
      }
      ++index;
    }
    if (exponent != 0) {
      value *= exp10(double(exponent * exponentsign));
    }
  }

  return value * sign;
}

template <>
__device__ cudf::date32 convertStrToValue<cudf::date32>(
    const char* data, long start, long end, const ParseOptions& opts) {
  return cudf::date32{parseDateFormat(data, start, end, opts.dayfirst)};
}

template <>
__device__ cudf::date64 convertStrToValue<cudf::date64>(
    const char* data, long start, long end, const ParseOptions& opts) {
  return cudf::date64{parseDateTimeFormat(data, start, end, opts.dayfirst)};
}

template <>
__device__ cudf::category convertStrToValue<cudf::category>(
    const char* data, long start, long end, const ParseOptions& opts) {
  constexpr int32_t HASH_SEED = 33;
  return cudf::category{convertStrToHash(data, start, end + 1, HASH_SEED)};
}

template <>
__device__ cudf::timestamp convertStrToValue<cudf::timestamp>(
    const char* data, long start, long end, const ParseOptions& opts) {
  return cudf::timestamp{convertStrToValue<int64_t>(data, start, end, opts)};
}

//The purpose of this is merely to allow compilation
//It should NOT be used
template <>
__device__ cudf::nvstring_category convertStrToValue<cudf::nvstring_category>(
    const char* data, long start, long end, const ParseOptions& opts) {
  assert(false);
  return cudf::nvstring_category{0};
}


template <>
__device__ cudf::bool8 convertStrToValue<cudf::bool8>(
    const char* data, long start, long end, const ParseOptions& opts) {
  cudf::bool8 return_value{cudf::false_v};

  // Check for user-specified true/false values first
  if (serializedTrieContains(opts.trueValuesTrie, data + start,
                             end - start + 1)) {
    return_value = cudf::true_v;
  } else if (serializedTrieContains(opts.falseValuesTrie, data + start,
                                    end - start + 1)) {
    return_value = cudf::false_v;
  } else {
    // Expect 'false_v' or 'true_v' in data, but clamp any non-zero value to 1
    // in case
    if (convertStrToValue<typename cudf::bool8::value_type>(
            data, start, end, opts) != cudf::detail::unwrap(cudf::false_v)) {
      return_value = cudf::true_v;
    } else {
      return_value = cudf::false_v;
    }
  }
  return return_value;
}

#endif
