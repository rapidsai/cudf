/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/detail/utilities/trie.cuh>
#include <cudf/io/types.hpp>

namespace cudf {
namespace io {
/**
 * @brief Structure for holding various options used when parsing and
 * converting CSV/json data to cuDF data type values.
 */
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

namespace gpu {
/**
 * @brief CUDA kernel iterates over the data until the end of the current field
 *
 * Also iterates over (one or more) delimiter characters after the field.
 * Function applies to formats with field delimiters and line terminators.
 *
 * @param begin Pointer to the first character in the parsing range
 * @param end pointer to the first character after the parsing range
 * @param opts A set of parsing options
 *
 * @return Pointer to the last character in the field, including the
 *  delimiter(s) following the field data
 */
__device__ __inline__ char const* seek_field_end(char const* begin,
                                                 char const* end,
                                                 ParseOptions const& opts)
{
  bool quotation = false;
  auto current   = begin;
  while (true) {
    // Use simple logic to ignore control chars between any quote seq
    // Handles nominal cases including doublequotes within quotes, but
    // may not output exact failures as PANDAS for malformed fields
    if (*current == opts.quotechar) {
      quotation = !quotation;
    } else if (!quotation) {
      if (*current == opts.delimiter) {
        while (opts.multi_delimiter && current < end && *(current + 1) == opts.delimiter) {
          ++current;
        }
        break;
      } else if (*current == opts.terminator) {
        break;
      } else if (*current == '\r' && (current + 1 < end && *(current + 1) == '\n')) {
        --end;
        break;
      }
    }
    if (current >= end) break;
    current++;
  }
  return current;
}

/**
 * @brief Returns the numeric value of an ASCII/UTF-8 character. Specialization
 * for integral types. Handles hexadecimal digits, both uppercase and lowercase.
 * If the character is not a valid numeric digit then `0` is returned and
 * valid_flag is set to false.
 *
 * @param c ASCII or UTF-8 character
 * @param valid_flag Set to false if input is not valid. Unchanged otherwise.
 *
 * @return uint8_t Numeric value of the character, or `0`
 */
template <typename T, typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
__device__ __forceinline__ uint8_t decode_digit(char c, bool* valid_flag)
{
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;

  *valid_flag = false;
  return 0;
}

/**
 * @brief Returns the numeric value of an ASCII/UTF-8 character. Specialization
 * for non-integral types. Handles only decimal digits. If the character is not
 * a valid numeric digit then `0` is returned and valid_flag is set to false.
 *
 * @param c ASCII or UTF-8 character
 * @param valid_flag Set to false if input is not valid. Unchanged otherwise.
 *
 * @return uint8_t Numeric value of the character, or `0`
 */
template <typename T, typename std::enable_if_t<!std::is_integral<T>::value>* = nullptr>
__device__ __forceinline__ uint8_t decode_digit(char c, bool* valid_flag)
{
  if (c >= '0' && c <= '9') return c - '0';

  *valid_flag = false;
  return 0;
}

/**
 * @brief Parses a character string and returns its numeric value.
 *
 * @param data The character string for parse
 * @param start The index within data to start parsing from
 * @param end The end index within data to end parsing
 * @param opts The global parsing behavior options
 * @param base Base (radix) to use for conversion
 *
 * @return The parsed and converted value
 */
template <typename T, int base = 10>
__inline__ __device__ T
parse_numeric(const char* data, long start, long end, ParseOptions const& opts)
{
  T value{};
  bool all_digits_valid = true;

  // Handle negative values if necessary
  int32_t sign = 1;
  if (data[start] == '-') {
    sign = -1;
    start++;
  }

  // Skip over the "0x" prefix for hex notation
  if (base == 16 && start + 2 <= end && data[start] == '0' && data[start + 1] == 'x') {
    start += 2;
  }

  // Handle the whole part of the number
  long index = start;
  while (index <= end) {
    if (data[index] == opts.decimal) {
      ++index;
      break;
    } else if (base == 10 && (data[index] == 'e' || data[index] == 'E')) {
      break;
    } else if (data[index] != opts.thousands && data[index] != '+') {
      value = (value * base) + decode_digit<T>(data[index], &all_digits_valid);
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
      } else if (data[index] != opts.thousands && data[index] != '+') {
        divisor /= base;
        value += decode_digit<T>(data[index], &all_digits_valid) * divisor;
      }
      ++index;
    }

    // Handle exponential part of the number if necessary
    if (index <= end) {
      const int32_t exponent_sign = data[index] == '-' ? -1 : 1;
      if (data[index] == '-' || data[index] == '+') { ++index; }
      int32_t exponent = 0;
      while (index <= end) {
        exponent = (exponent * 10) + decode_digit<T>(data[index++], &all_digits_valid);
      }
      if (exponent != 0) { value *= exp10(double(exponent * exponent_sign)); }
    }
  }
  if (!all_digits_valid) { return std::numeric_limits<T>::quiet_NaN(); }

  return value * sign;
}

}  // namespace gpu

/**
 * @brief Searches the input character array for each of characters in a set.
 * Sums up the number of occurrences. If the 'positions' parameter is not void*,
 * positions of all occurrences are stored in the output device array.
 *
 * @param[in] d_data Input character array in device memory
 * @param[in] keys Vector containing the keys to count in the buffer
 * @param[in] result_offset Offset to add to the output positions
 * @param[out] positions Array containing the output positions
 *
 * @return cudf::size_type total number of occurrences
 **/
template <class T>
cudf::size_type find_all_from_set(const rmm::device_buffer& d_data,
                                  const std::vector<char>& keys,
                                  uint64_t result_offset,
                                  T* positions);

/**
 * @brief Searches the input character array for each of characters in a set.
 * Sums up the number of occurrences. If the 'positions' parameter is not void*,
 * positions of all occurrences are stored in the output device array.
 *
 * Does not load the entire file into the GPU memory at any time, so it can
 * be used to parse large files. Output array needs to be preallocated.
 *
 * @param[in] h_data Pointer to the input character array
 * @param[in] h_size Number of bytes in the input array
 * @param[in] keys Vector containing the keys to count in the buffer
 * @param[in] result_offset Offset to add to the output positions
 * @param[out] positions Array containing the output positions
 *
 * @return cudf::size_type total number of occurrences
 **/
template <class T>
cudf::size_type find_all_from_set(const char* h_data,
                                  size_t h_size,
                                  const std::vector<char>& keys,
                                  uint64_t result_offset,
                                  T* positions);

/**
 * @brief Searches the input character array for each of characters in a set
 * and sums up the number of occurrences.
 *
 * @param[in] d_data Input data buffer in device memory
 * @param[in] keys Vector containing the keys to count in the buffer
 *
 * @return cudf::size_type total number of occurrences
 **/
cudf::size_type count_all_from_set(const rmm::device_buffer& d_data, const std::vector<char>& keys);

/**
 * @brief Searches the input character array for each of characters in a set
 * and sums up the number of occurrences.
 *
 * Does not load the entire buffer into the GPU memory at any time, so it can
 * be used with buffers of any size.
 *
 * @param[in] h_data Pointer to the data in host memory
 * @param[in] h_size Size of the input data, in bytes
 * @param[in] keys Vector containing the keys to count in the buffer
 *
 * @return cudf::size_type total number of occurrences
 **/
cudf::size_type count_all_from_set(const char* h_data,
                                   size_t h_size,
                                   const std::vector<char>& keys);

/**
 * @brief Infer file compression type based on user supplied arguments.
 *
 * If the user specifies a valid compression_type for compression arg,
 * compression type will be computed based on that.  Otherwise the filename
 * and ext_to_comp_map will be used.
 *
 * @param[in] compression_arg User specified compression type (if any)
 * @param[in] filename Filename to base compression type (by extension) on
 * @param[in] ext_to_comp_map User supplied mapping of file extension to compression type
 *
 * @return string representing compression type ("gzip, "bz2", etc)
 **/
std::string infer_compression_type(
  const compression_type& compression_arg,
  const std::string& filename,
  const std::vector<std::pair<std::string, std::string>>& ext_to_comp_map);

}  // namespace io
}  // namespace cudf
