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
#include <cudf/utilities/span.hpp>

#include <io/utilities/column_type_histogram.hpp>

#include <rmm/device_vector.hpp>

using cudf::detail::device_span;

namespace cudf {
namespace io {
/**
 * @brief Structure for holding various options used when parsing and
 * converting CSV/json data to cuDF data type values.
 */
struct parse_options_view {
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
  device_span<SerialTrieNode const> trie_true;
  device_span<SerialTrieNode const> trie_false;
  device_span<SerialTrieNode const> trie_na;
  bool multi_delimiter;
};

struct parse_options {
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
  rmm::device_vector<SerialTrieNode> trie_true;
  rmm::device_vector<SerialTrieNode> trie_false;
  rmm::device_vector<SerialTrieNode> trie_na;
  bool multi_delimiter;

  parse_options_view view()
  {
    return {delimiter,
            terminator,
            quotechar,
            decimal,
            thousands,
            comment,
            keepquotes,
            doublequote,
            dayfirst,
            skipblanklines,
            trie_true,
            trie_false,
            trie_na,
            multi_delimiter};
  }
};

namespace gpu {
/**
 * @brief CUDA kernel iterates over the data until the end of the current field
 *
 * Also iterates over (one or more) delimiter characters after the field.
 * Function applies to formats with field delimiters and line terminators.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param opts A set of parsing options
 * @param escape_char A boolean value to signify whether to consider `\` as escape character or
 * just a character.
 *
 * @return Pointer to the last character in the field, including the
 *  delimiter(s) following the field data
 */
__device__ __inline__ char const* seek_field_end(char const* begin,
                                                 char const* end,
                                                 parse_options_view const& opts,
                                                 bool escape_char = false)
{
  bool quotation   = false;
  auto current     = begin;
  bool escape_next = false;
  while (true) {
    // Use simple logic to ignore control chars between any quote seq
    // Handles nominal cases including doublequotes within quotes, but
    // may not output exact failures as PANDAS for malformed fields.
    // Check for instances such as "a2\"bc" and "\\" if `escape_char` is true.

    if (*current == opts.quotechar and not escape_next) {
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

    if (escape_char == true) {
      // If a escape character is encountered, escape next character in next loop.
      if (escape_next == false and *current == '\\') {
        escape_next = true;
      } else {
        escape_next = false;
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

// Converts character to lowercase.
__inline__ __device__ char to_lower(char const c)
{
  return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c;
}

/**
 * @brief Checks if string is infinity, case insensitive with/without sign
 * Valid infinity strings are inf, +inf, -inf, infinity, +infinity, -infinity
 * String comparison is case insensitive.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return true if string is valid infinity, else false.
 */
__inline__ __device__ bool is_infinity(char const* begin, char const* end)
{
  if (*begin == '-' || *begin == '+') begin++;
  char const* cinf = "infinity";
  auto index       = begin;
  while (index < end) {
    if (*cinf != to_lower(*index)) break;
    index++;
    cinf++;
  }
  return ((index == begin + 3 || index == begin + 8) && index >= end);
}

/**
 * @brief Parses a character string and returns its numeric value.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param opts The global parsing behavior options
 * @tparam base Base (radix) to use for conversion
 *
 * @return The parsed and converted value
 */
template <typename T, int base = 10>
__inline__ __device__ T parse_numeric(const char* begin,
                                      const char* end,
                                      parse_options_view const& opts)
{
  T value{};
  bool all_digits_valid = true;

  // Handle negative values if necessary
  int32_t sign = (*begin == '-') ? -1 : 1;

  // Handle infinity
  if (std::is_floating_point<T>::value && is_infinity(begin, end)) {
    return sign * std::numeric_limits<T>::infinity();
  }
  if (*begin == '-' || *begin == '+') begin++;

  // Skip over the "0x" prefix for hex notation
  if (base == 16 && begin + 2 < end && *begin == '0' && *(begin + 1) == 'x') { begin += 2; }

  // Handle the whole part of the number
  // auto index = begin;
  while (begin < end) {
    if (*begin == opts.decimal) {
      ++begin;
      break;
    } else if (base == 10 && (*begin == 'e' || *begin == 'E')) {
      break;
    } else if (*begin != opts.thousands && *begin != '+') {
      value = (value * base) + decode_digit<T>(*begin, &all_digits_valid);
    }
    ++begin;
  }

  if (std::is_floating_point<T>::value) {
    // Handle fractional part of the number if necessary
    double divisor = 1;
    while (begin < end) {
      if (*begin == 'e' || *begin == 'E') {
        ++begin;
        break;
      } else if (*begin != opts.thousands && *begin != '+') {
        divisor /= base;
        value += decode_digit<T>(*begin, &all_digits_valid) * divisor;
      }
      ++begin;
    }

    // Handle exponential part of the number if necessary
    if (begin < end) {
      const int32_t exponent_sign = *begin == '-' ? -1 : 1;
      if (*begin == '-' || *begin == '+') { ++begin; }
      int32_t exponent = 0;
      while (begin < end) {
        exponent = (exponent * 10) + decode_digit<T>(*(begin++), &all_digits_valid);
      }
      if (exponent != 0) { value *= exp10(double(exponent * exponent_sign)); }
    }
  }
  if (!all_digits_valid) { return std::numeric_limits<T>::quiet_NaN(); }

  return value * sign;
}

/**
 * @brief Lexicographically compare digits in input against string
 * representing an integer
 *
 * @param raw_data The pointer to beginning of character string
 * @param golden The pointer to beginning of character string representing
 * the value to be compared against
 * @return bool True if integer represented by character string is less
 * than or equal to golden data
 */
template <int N>
__device__ __inline__ bool less_equal_than(const char* data, const char (&golden)[N])
{
  auto mismatch_pair = thrust::mismatch(thrust::seq, data, data + N - 1, golden);
  if (mismatch_pair.first != data + N - 1) {
    return *mismatch_pair.first <= *mismatch_pair.second;
  } else {
    // Exact match
    return true;
  }
}

/**
 * @brief Determine which counter to increment when a sequence of digits
 * and a parity sign is encountered.
 *
 * @param raw_data The pointer to beginning of character string
 * @param digit_count Total number of digits
 * @param stats Reference to structure with counters
 * @return Pointer to appropriate counter that belong to
 * the interpreted data type
 */
__device__ __inline__ cudf::size_type* infer_integral_field_counter(char const* data_begin,
                                                                    char const* data_end,
                                                                    bool is_negative,
                                                                    column_type_histogram& stats)
{
  static constexpr char uint64_max_abs[] = "18446744073709551615";
  static constexpr char int64_min_abs[]  = "9223372036854775808";
  static constexpr char int64_max_abs[]  = "9223372036854775807";

  auto digit_count = data_end - data_begin;

  // Remove preceding zeros
  if (digit_count >= (sizeof(int64_max_abs) - 1)) {
    // Trim zeros at the beginning of raw_data
    while (*data_begin == '0' && (data_begin < data_end)) { data_begin++; }
  }
  digit_count = data_end - data_begin;

  // After trimming the number of digits could be less than maximum
  // int64 digit count
  if (digit_count < (sizeof(int64_max_abs) - 1)) {  // CASE 0 : Accept validity
    // If the length of the string representing the integer is smaller
    // than string length of Int64Max then count this as an integer
    // representable by int64
    // If digit_count is 0 then ignore - sign, i.e. -000..00 should
    // be treated as a positive small integer
    return is_negative && (digit_count != 0) ? &stats.negative_small_int_count
                                             : &stats.positive_small_int_count;
  } else if (digit_count > (sizeof(uint64_max_abs) - 1)) {  // CASE 1 : Reject validity
    // If the length of the string representing the integer is greater
    // than string length of UInt64Max then count this as a string
    // since it cannot be represented as an int64 or uint64
    return &stats.string_count;
  } else if (digit_count == (sizeof(uint64_max_abs) - 1) && is_negative) {
    // A negative integer of length UInt64Max digit count cannot be represented
    // as a 64 bit integer
    return &stats.string_count;
  }

  if (digit_count == (sizeof(int64_max_abs) - 1) && is_negative) {
    return less_equal_than(data_begin, int64_min_abs) ? &stats.negative_small_int_count
                                                      : &stats.string_count;
  } else if (digit_count == (sizeof(int64_max_abs) - 1) && !is_negative) {
    return less_equal_than(data_begin, int64_max_abs) ? &stats.positive_small_int_count
                                                      : &stats.big_int_count;
  } else if (digit_count == (sizeof(uint64_max_abs) - 1)) {
    return less_equal_than(data_begin, uint64_max_abs) ? &stats.big_int_count : &stats.string_count;
  }

  return &stats.string_count;
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
 */
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
 */
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
 */
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
 */
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
 */
std::string infer_compression_type(
  const compression_type& compression_arg,
  const std::string& filename,
  const std::vector<std::pair<std::string, std::string>>& ext_to_comp_map);

/**
 * @brief Checks whether the given character is a whitespace character.
 *
 * @param[in] ch The character to check
 *
 * @return True if the input is whitespace, False otherwise
 */
__inline__ __device__ bool is_whitespace(char ch) { return ch == '\t' || ch == ' '; }

/**
 * @brief Skips past the current character if it matches the given value.
 */
template <typename It>
__inline__ __device__ It skip_character(It const& it, char ch)
{
  return it + (*it == ch);
}

/**
 * @brief Adjusts the range to ignore starting/trailing whitespace and quotation characters.
 *
 * @param[in] begin Pointer to the first character in the parsing range
 * @param[in] end pointer to the first character after the parsing range
 * @param[in] quotechar The character used to denote quotes; '\0' if none
 *
 * @return Trimmed range
 */
__inline__ __device__ std::pair<char const*, char const*> trim_whitespaces_quotes(
  char const* begin, char const* end, char quotechar = '\0')
{
  auto not_whitespace = [] __device__(auto c) { return !is_whitespace(c); };

  auto const trim_begin = thrust::find_if(thrust::seq, begin, end, not_whitespace);
  auto const trim_end   = thrust::find_if(thrust::seq,
                                        thrust::make_reverse_iterator(end),
                                        thrust::make_reverse_iterator(trim_begin),
                                        not_whitespace);

  return {skip_character(trim_begin, quotechar), skip_character(trim_end, quotechar).base()};
}

/**
 * @brief Excludes the prefix from the input range if the string starts with the prefix.
 *
 * @tparam N length on the prefix, plus one
 * @param begin[in, out] Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param prefix String we're searching for at the start of the input range
 */
template <int N>
__inline__ __device__ auto skip_if_starts_with(char const* begin,
                                               char const* end,
                                               const char (&prefix)[N])
{
  static constexpr size_t prefix_len = N - 1;
  if (end - begin < prefix_len) return begin;
  return thrust::equal(thrust::seq, begin, begin + prefix_len, prefix) ? begin + prefix_len : begin;
}

/**
 * @brief Finds the first element after the leading space characters.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 */
__inline__ __device__ auto skip_spaces(char const* begin, char const* end)
{
  return thrust::find_if(thrust::seq, begin, end, [](auto elem) { return elem != ' '; });
}

}  // namespace io
}  // namespace cudf
