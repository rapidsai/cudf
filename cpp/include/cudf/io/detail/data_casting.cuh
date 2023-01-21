/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <io/utilities/parsing_utils.cuh>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>

#include <memory>

namespace cudf::io::json::experimental::detail {

// Unicode code point escape sequence
static constexpr char UNICODE_SEQ = 0x7F;

// Invalid escape sequence
static constexpr char NON_ESCAPE_CHAR = 0x7E;

// Unicode code point escape sequence prefix comprises '\' and 'u' characters
static constexpr size_type UNICODE_ESC_PREFIX = 2;

// Unicode code point escape sequence comprises four hex characters
static constexpr size_type UNICODE_HEX_DIGIT_COUNT = 4;

// A unicode code point escape sequence is \uXXXX
static auto constexpr NUM_UNICODE_ESC_SEQ_CHARS = UNICODE_ESC_PREFIX + UNICODE_HEX_DIGIT_COUNT;

static constexpr auto UTF16_HIGH_SURROGATE_BEGIN = 0xD800;
static constexpr auto UTF16_HIGH_SURROGATE_END   = 0xDC00;
static constexpr auto UTF16_LOW_SURROGATE_BEGIN  = 0xDC00;
static constexpr auto UTF16_LOW_SURROGATE_END    = 0xE000;

/**
 * @brief Describing whether data casting of a certain item succeed, the item was parsed to null, or
 * whether type casting failed.
 */
enum class data_casting_result { PARSING_SUCCESS, PARSED_TO_NULL, PARSING_FAILURE };

/**
 * @brief Providing additional information about the type casting result.
 */
struct data_casting_result_info {
  // Number of bytes written to output
  size_type bytes;
  // Whether parsing succeeded, item was parsed to null, or failed
  data_casting_result result;
};

/**
 * @brief Returns the character to output for a given escaped character that's following a
 * backslash.
 *
 * @param escaped_char The character following the backslash.
 * @return The character to output for a given character that's following a backslash
 */
__device__ __forceinline__ char get_escape_char(char escaped_char)
{
  switch (escaped_char) {
    case '"': return '"';
    case '\\': return '\\';
    case '/': return '/';
    case 'b': return '\b';
    case 'f': return '\f';
    case 'n': return '\n';
    case 'r': return '\r';
    case 't': return '\t';
    case 'u': return UNICODE_SEQ;
    default: return NON_ESCAPE_CHAR;
  }
}

/**
 * @brief Parses the hex value from the four hex digits of a unicode code point escape sequence
 * \uXXXX.
 *
 * @param str Pointer to the first (most-significant) hex digit
 * @return The parsed hex value if successful, -1 otherwise.
 */
__device__ __forceinline__ int32_t parse_unicode_hex(char const* str)
{
  // Prepare result
  int32_t result = 0, base = 1;
  constexpr int32_t hex_radix = 16;

  // Iterate over hex digits right-to-left
  size_type index = UNICODE_HEX_DIGIT_COUNT;
  while (index-- > 0) {
    char const ch = str[index];
    if (ch >= '0' && ch <= '9') {
      result += static_cast<int32_t>((ch - '0') + 0) * base;
      base *= hex_radix;
    } else if (ch >= 'A' && ch <= 'F') {
      result += static_cast<int32_t>((ch - 'A') + 10) * base;
      base *= hex_radix;
    } else if (ch >= 'a' && ch <= 'f') {
      result += static_cast<int32_t>((ch - 'a') + 10) * base;
      base *= hex_radix;
    } else {
      return -1;
    }
  }
  return result;
}

/**
 * @brief Writes the UTF-8 byte sequence to \p out_it and returns the number of bytes written to
 * \p out_it
 */
constexpr size_type write_utf8_char(char_utf8 character, char*& out_it)
{
  auto const bytes = (out_it == nullptr) ? strings::detail::bytes_in_char_utf8(character)
                                         : strings::detail::from_char_utf8(character, out_it);
  if (out_it) out_it += bytes;
  return bytes;
}

/**
 * @brief Processes a string, replaces escape sequences and optionally strips off the quote
 * characters.
 *
 * @tparam in_iterator_t A bidirectional input iterator type whose value_type is convertible to
 * char
 * @param in_begin Iterator to the first item to process
 * @param in_end Iterator to one past the last item to process
 * @param d_buffer Output character buffer to the first item to write
 * @param options Settings for controlling string processing behavior
 * @return A struct of (num_bytes_written, parsing_success_result), where num_bytes_written is
 * the number of bytes written to d_buffer, parsing_success_result is enum value indicating whether
 * parsing succeeded, item was parsed to null, or failed.
 */
template <typename in_iterator_t>
__device__ __forceinline__ data_casting_result_info
process_string(in_iterator_t in_begin,
               in_iterator_t in_end,
               char* d_buffer,
               cudf::io::parse_options_view const& options)
{
  int32_t bytes           = 0;
  const auto num_in_chars = thrust::distance(in_begin, in_end);
  // String values are indicated by keeping the quote character
  bool const is_string_value = num_in_chars >= 2LL && (*in_begin == options.quotechar) &&
                               (*thrust::prev(in_end) == options.quotechar);

  // Copy literal/numeric value
  if (not is_string_value) {
    while (in_begin != in_end) {
      if (d_buffer) *d_buffer++ = *in_begin;
      ++in_begin;
      ++bytes;
    }
    return {bytes, data_casting_result::PARSING_SUCCESS};
  }
  // Whether in the original JSON this was a string value enclosed in quotes
  // ({"a":"foo"} vs. {"a":1.23})
  char const backslash_char = '\\';

  // Escape-flag, set after encountering a backslash character
  bool escape = false;

  // Exclude beginning and ending quote chars from string range
  if (!options.keepquotes) {
    ++in_begin;
    --in_end;
  }

  // Iterate over the input
  while (in_begin != in_end) {
    // Copy single character to output
    if (!escape) {
      escape = (*in_begin == backslash_char);
      if (!escape) {
        if (d_buffer) *d_buffer++ = *in_begin;
        ++bytes;
      }
      ++in_begin;
      continue;
    }

    // Previous char indicated beginning of escape sequence
    // Reset escape flag for next loop iteration
    escape = false;

    // Check the character that is supposed to be escaped
    auto escaped_char = get_escape_char(*in_begin);

    // We escaped an invalid escape character -> "fail"/null for this item
    if (escaped_char == NON_ESCAPE_CHAR) { return {bytes, data_casting_result::PARSING_FAILURE}; }

    // Regular, single-character escape
    if (escaped_char != UNICODE_SEQ) {
      if (d_buffer) *d_buffer++ = escaped_char;
      ++bytes;
      ++in_begin;
      continue;
    }

    // This is an escape sequence of a unicode code point: \uXXXX,
    // where each X in XXXX represents a hex digit
    // Skip over the 'u' char from \uXXXX to the first hex digit
    ++in_begin;

    // Make sure that there's at least 4 characters left from the
    // input, which are expected to be hex digits
    if (thrust::distance(in_begin, in_end) < UNICODE_HEX_DIGIT_COUNT) {
      return {bytes, data_casting_result::PARSING_FAILURE};
    }

    auto hex_val = parse_unicode_hex(in_begin);

    // Couldn't parse hex values from the four-character sequence -> "fail"/null for this item
    if (hex_val < 0) { return {bytes, data_casting_result::PARSING_FAILURE}; }

    // Skip over the four hex digits
    thrust::advance(in_begin, UNICODE_HEX_DIGIT_COUNT);

    // If this may be a UTF-16 encoded surrogate pair:
    // we expect another \uXXXX sequence
    int32_t hex_low_val = 0;
    if (thrust::distance(in_begin, in_end) >= NUM_UNICODE_ESC_SEQ_CHARS &&
        *in_begin == backslash_char && *thrust::next(in_begin) == 'u') {
      // Try to parse hex value following the '\' and 'u' characters from what may be a UTF16 low
      // surrogate
      hex_low_val = parse_unicode_hex(thrust::next(in_begin, 2));
    }

    // This is indeed a UTF16 surrogate pair
    if (hex_val >= UTF16_HIGH_SURROGATE_BEGIN && hex_val < UTF16_HIGH_SURROGATE_END &&
        hex_low_val >= UTF16_LOW_SURROGATE_BEGIN && hex_low_val < UTF16_LOW_SURROGATE_END) {
      // Skip over the second \uXXXX sequence
      thrust::advance(in_begin, NUM_UNICODE_ESC_SEQ_CHARS);

      // Compute UTF16-encoded code point
      uint32_t unicode_code_point = 0x10000 + ((hex_val - UTF16_HIGH_SURROGATE_BEGIN) << 10) +
                                    (hex_low_val - UTF16_LOW_SURROGATE_BEGIN);
      auto utf8_chars = strings::detail::codepoint_to_utf8(unicode_code_point);
      bytes += write_utf8_char(utf8_chars, d_buffer);
    }

    // Just a single \uXXXX sequence
    else {
      auto utf8_chars = strings::detail::codepoint_to_utf8(hex_val);
      bytes += write_utf8_char(utf8_chars, d_buffer);
    }
  }

  // The last character of the input is a backslash -> "fail"/null for this item
  if (escape) { return {bytes, data_casting_result::PARSING_FAILURE}; }
  return {bytes, data_casting_result::PARSING_SUCCESS};
}

template <typename str_tuple_it>
struct string_parse {
  str_tuple_it str_tuples;
  bitmask_type* null_mask;
  cudf::io::parse_options_view const options;
  size_type* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (not bit_is_set(null_mask, idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const in_begin     = str_tuples[idx].first;
    auto const in_end       = in_begin + str_tuples[idx].second;
    auto const num_in_chars = str_tuples[idx].second;

    // Check if the value corresponds to the null literal
    auto const is_null_literal =
      (!d_chars) &&
      serialized_trie_contains(options.trie_na, {in_begin, static_cast<std::size_t>(num_in_chars)});
    if (is_null_literal) {
      clear_bit(null_mask, idx);
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }

    char* d_buffer        = d_chars ? d_chars + d_offsets[idx] : nullptr;
    auto str_process_info = process_string(in_begin, in_end, d_buffer, options);
    if (str_process_info.result != data_casting_result::PARSING_SUCCESS) {
      clear_bit(null_mask, idx);
      if (!d_chars) d_offsets[idx] = 0;
    } else {
      if (!d_chars) d_offsets[idx] = str_process_info.bytes;
    }
  }
};
/**
 * @brief Parses the data from an iterator of string views, casting it to the given target data type
 *
 * @param str_tuples Iterator returning a string view, i.e., a (ptr, length) pair
 * @param col_size The total number of items of this column
 * @param col_type The column's target data type
 * @param null_mask A null mask that renders certain items from the input invalid
 * @param options Settings for controlling the processing behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr The resource to be used for device memory allocation
 * @return The column that contains the parsed data
 */
template <typename str_tuple_it, typename B>
std::unique_ptr<column> parse_data(str_tuple_it str_tuples,
                                   size_type col_size,
                                   data_type col_type,
                                   B&& null_mask,
                                   cudf::io::parse_options_view const& options,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  if (col_type == cudf::data_type{cudf::type_id::STRING}) {
    // this utility calls the functor to build the offsets and chars columns
    auto [offsets, chars] = cudf::strings::detail::make_strings_children(
      string_parse<decltype(str_tuples)>{
        str_tuples, static_cast<bitmask_type*>(null_mask.data()), options},
      col_size,
      stream,
      mr);

    auto null_count =
      cudf::detail::null_count(static_cast<bitmask_type*>(null_mask.data()), 0, col_size, stream);
    return make_strings_column(
      col_size, std::move(offsets), std::move(chars), null_count, std::move(null_mask));
  }

  auto out_col = make_fixed_width_column(
    col_type, col_size, std::move(null_mask), cudf::UNKNOWN_NULL_COUNT, stream, mr);
  auto output_dv_ptr = mutable_column_device_view::create(*out_col, stream);

  // use existing code (`ConvertFunctor`) to convert values
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    col_size,
    [str_tuples, col = *output_dv_ptr, options, col_type] __device__(size_type row) {
      if (col.is_null(row)) { return; }
      auto const in = str_tuples[row];

      auto const is_null_literal =
        serialized_trie_contains(options.trie_na, {in.first, static_cast<size_t>(in.second)});

      if (is_null_literal) {
        col.set_null(row);
        return;
      }

      // If this is a string value, remove quotes
      auto [in_begin, in_end] = trim_quotes(in.first, in.first + in.second, options.quotechar);

      auto const is_parsed = cudf::type_dispatcher(col_type,
                                                   ConvertFunctor{},
                                                   in_begin,
                                                   in_end,
                                                   col.data<char>(),
                                                   row,
                                                   col_type,
                                                   options,
                                                   false);
      if (not is_parsed) { col.set_null(row); }
    });

  return out_col;
}

}  // namespace cudf::io::json::experimental::detail
