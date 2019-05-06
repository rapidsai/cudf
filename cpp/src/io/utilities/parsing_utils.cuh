/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

/**
 * @file parsing_utils.cuh Declarations of utility functions for parsing plain-text files
 *
 */


#pragma once

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cudf.h"

#include <thrust/pair.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "rmm/rmm.h"
#include "rmm/thrust_rmm_allocator.h"

#include "io/utilities/wrapper_utils.hpp"
#include "io/csv/type_conversion.cuh"

gdf_size_type countAllFromSet(const char *h_data, size_t h_size, const std::vector<char>& keys);

template<class T>
gdf_size_type findAllFromSet(const char *h_data, size_t h_size, const std::vector<char>& keys, uint64_t result_offset,
	T *positions);

device_buffer<int16_t> getBracketLevels(
	thrust::pair<uint64_t,char>* brackets, int count, 
	const std::string& open_chars, const std::string& close_chars);

#ifdef __CUDACC__
/**
 * @brief Sets the specified bit in a device memory bitmap.
 * Uses atomics for synchronization.
 */
__device__ __inline__ void setBitmapBit(gdf_valid_type *bitmap, long bit_idx) {
  constexpr int32_t bit_mask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  const auto address = bitmap + bit_idx / 8;

  int32_t *const base_address = (int32_t *)((gdf_valid_type *)address - ((size_t)address & 3));
  const int32_t mask = bit_mask[bit_idx % 8] << (((size_t)address & 3) * 8);

  atomicOr(base_address, mask);
}

/**
 * @brief Returns true is the input character is a valid digit.
 * Supports both decimal and hexadecimal digits (uppercase and lowercase).
 */
__device__ __inline__ bool isDigit(char c, bool is_hex) {
  if (c >= '0' && c <= '9')
    return true;
  if (is_hex) {
    if (c >= 'A' && c <= 'F')
      return true;
    if (c >= 'a' && c <= 'f')
      return true;
  }
  return false;
}

/**
 * @brief Returns true if the counters indicate a potentially valid float.
 * False positives are possible because positions are not taken into account.
 * For example, field "e.123-" would match the pattern.
 */
__device__ __inline__ bool isLikeFloat(long len, long digit_cnt, long decimal_cnt, long dash_cnt, long exponent_cnt) {
  // Can't have more than one exponent and one decimal point
  if (decimal_cnt > 1)
    return false;
  if (exponent_cnt > 1)
    return false;
  // Without the exponent or a decimal point, this is an integer, not a float
  if (decimal_cnt == 0 && exponent_cnt == 0)
    return false;

  // Can only have one '-' per component
  if (dash_cnt > 1 + exponent_cnt)
    return false;

  // If anything other than these characters is present, it's not a float
  if (digit_cnt + decimal_cnt + dash_cnt + exponent_cnt != len)
    return false;

  // Needs at least 1 digit, 2 if exponent is present
  if (digit_cnt < 1 + exponent_cnt)
    return false;

  return true;
}

/**---------------------------------------------------------------------------*
 * @brief CUDA kernel iterates over the data until the end of the current field
 *
 * Also iterates over (one or more) delimiter characters after the field.
 * Function applies to formats with field delimiters and line terminators.
 *
 * @param[in] data The entire plain text data to read
 * @param[in] opts A set of parsing options
 * @param[in] pos Offset to start the seeking from
 * @param[in] stop Offset of the end of the row
 *
 * @return long position of the last character in the field, including the
 *  delimiter(s) following the field data
 *---------------------------------------------------------------------------**/
__inline__ __device__ long seekFieldEnd(const char *data, const ParseOptions opts, long pos, long stop) {
  bool quotation = false;
  while (true) {
    // Use simple logic to ignore control chars between any quote seq
    // Handles nominal cases including doublequotes within quotes, but
    // may not output exact failures as PANDAS for malformed fields
    if (data[pos] == opts.quotechar) {
      quotation = !quotation;
    } else if (quotation == false) {
      if (data[pos] == opts.delimiter) {
        while (opts.multi_delimiter && pos < stop && data[pos + 1] == opts.delimiter) {
          ++pos;
        }
        break;
      } else if (data[pos] == opts.terminator) {
        break;
      } else if (data[pos] == '\r' && (pos + 1 < stop && data[pos + 1] == '\n')) {
        stop--;
        break;
      }
    }
    if (pos >= stop)
      break;
    pos++;
  }
  return pos;
}
#endif
