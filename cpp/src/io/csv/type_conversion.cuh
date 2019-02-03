/* Copyright 2017 NVIDIA Corporation.  All rights reserved. */

#ifndef CONVERSION_FUNCTIONS_CUH
#define CONVERSION_FUNCTIONS_CUH

#include "datetime_parser.cuh"
#include "utilities/wrapper_types.hpp"
#include <cuda_runtime_api.h>

//---------------------------------------------------------------------------
//				Helper functions
//---------------------------------------------------------------------------


__inline__ __device__ 
bool isWhitespace(char ch) {
	return ch == '\t' || ch == ' ';
}


__device__
void adjustForWhitespaceAndQuotes(const char *data, long* start_idx, long* end_idx, char quotechar='\0') {
  while ((*start_idx < *end_idx) && (isWhitespace(data[*start_idx]) || data[*start_idx] == quotechar)) {
    (*start_idx)++;
  }
  while ((*start_idx < *end_idx) && (isWhitespace(data[*end_idx]) || data[*end_idx] == quotechar)) {
    (*end_idx)--;
  }
}

template<typename T>
__host__ __device__
bool isBooleanValue(T value, int32_t* boolValues, int32_t count) {
	for (int i = 0; i < count; ++i) {
		if (static_cast<int32_t>(value) == boolValues[i]) {
			return true;
		}
	}
	return false;
}

//---------------------------------------------------------------------------

__forceinline__
__host__ __device__ uint32_t rotl32( uint32_t x, int8_t r )
{
  return (x << r) | (x >> (32 - r));
}

__forceinline__
__host__ __device__ uint32_t fmix32( uint32_t h )
{
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

//MurmurHash3_32 implementation from https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.//MurmurHash3_32 implementation from https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.
//
// This is a modified version of what is used for hash-join.  The change is at accept
// a char * key and range (start and end) so that the large raw CSV data pointer
// could be used
__host__ __device__
int32_t convertStrtoHash(const char * key, long start_idx, long end_idx, uint32_t m_seed)
{

    int len = (end_idx - start_idx) + 1;				// +1 since it is an inclusive range

    const uint8_t * const data = (const uint8_t*)key;
    int nblocks = len / 4;
    uint32_t h1 = m_seed;
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;
    //----------
    // body
    const uint32_t * const blocks = (const uint32_t *)(data + nblocks*4);
    for(int i = -nblocks; i; i++)
    {
        uint32_t k1 = blocks[i];//getblock32(blocks,i);
        k1 *= c1;
        k1 = rotl32(k1,15);
        k1 *= c2;
        h1 ^= k1;
        h1 = rotl32(h1,13);
        h1 = h1*5+0xe6546b64;
    }
    //----------
    // tail
    int processed_len = nblocks * 4;
    int left		= len - processed_len;
    long pad_idx	= end_idx - left + 1;
	char tail[4];

	for ( int idx = 0; idx < 4; idx ++) {
		if ( pad_idx + idx > end_idx)
			tail[idx] = 0;
		else
			tail[idx] = data[pad_idx + idx];
	}

    //const uint8_t * tail = (const uint8_t*)(data + nblocks*4);
    uint32_t k1 = 0;
    switch(len & 3)
    {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
                k1 *= c1; k1 = rotl32(k1,15); k1 *= c2; h1 ^= k1;
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
  bool keepquotes;
  bool doublequote;
  bool dayfirst;
  int32_t* trueValues;
  int32_t* falseValues;
  int32_t trueValuesCount;
  int32_t falseValuesCount;
};

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
__host__ __device__ T convertStrToValue(const char* data, long start, long end,
                                        const ParseOptions& opts) {
  T value = 0;

  // Handle negative values if necessary
  int32_t sign = 1;
  if (data[start] == '-') {
    sign = -1;
    start++;
  }

  // Handle the whole part of the number
  long index = start;
  while (index <= end) {
    if (data[index] == opts.decimal) {
      ++index;
      break;
    } else if (data[index] == 'e' || data[index] == 'E') {
      break;
    } else if (data[index] != opts.thousands) {
      value *= 10;
      value += data[index] - '0';
    }
    ++index;
  }

  if (std::is_floating_point<T>::value) {
    // Handle fractional part of the number if necessary
    int32_t divisor = 1;
    while (index <= end) {
      if (data[index] == 'e' || data[index] == 'E') {
        ++index;
        break;
      } else if (data[index] != opts.thousands) {
        value *= 10;
        value += data[index] - '0';
        divisor *= 10;
      }
      ++index;
    }

    // Handle exponential part of the number if necessary
    int32_t exponent = 0;
    while (index <= end) {
      if (data[index] == '-') {
        ++index;
        exponent = (data[index] - '0') * -1;
      } else {
        exponent *= 10;
        exponent += data[index] - '0';
      }
      ++index;
    }

    if (divisor > 1) {
      value /= divisor;
    }
    if (exponent != 0) {
      value *= exp10f(exponent);
    }
  }

  return value * sign;
}

template <>
__host__ __device__ cudf::date32 convertStrToValue<cudf::date32>(
    const char* data, long start, long end, const ParseOptions& opts) {
  return cudf::date32{parseDateFormat(data, start, end, opts.dayfirst)};
}

template <>
__host__ __device__ cudf::date64 convertStrToValue<cudf::date64>(
    const char* data, long start, long end, const ParseOptions& opts) {
  return cudf::date64{parseDateTimeFormat(data, start, end, opts.dayfirst)};
}

template <>
__host__ __device__ cudf::category convertStrToValue<cudf::category>(
    const char* data, long start, long end, const ParseOptions& opts) {
  constexpr int32_t HASH_SEED = 33;
  return cudf::category{convertStrtoHash(data, start, end + 1, HASH_SEED)};
}

template <>
__host__ __device__ cudf::timestamp convertStrToValue<cudf::timestamp>(
    const char* data, long start, long end, const ParseOptions& opts) {
  return cudf::timestamp{convertStrToValue<int64_t>(data, start, end, opts)};
}

#endif
