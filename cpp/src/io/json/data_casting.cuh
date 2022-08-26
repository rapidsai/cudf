/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <memory>

namespace cudf::io::json::experimental {

constexpr char UNICODE_SEQ     = 0x7F;
constexpr char NON_ESCAPE_CHAR = 0x7E;
__device__ __forceinline__ char get_escape_char(char escaped_char)
{
  switch (escaped_char) {
    case '"': return 0x22;
    case '\\': return 0x5C;
    case '/': return 0x2F;
    case 'b': return 0x08;
    case 'f': return 0x0C;
    case 'n': return 0x0A;
    case 'r': return 0x0D;
    case 't': return 0x09;
    case 'u': return UNICODE_SEQ;
    default: return NON_ESCAPE_CHAR;
  }
}

__device__ __forceinline__ int64_t string_to_hex(char const* str)
{
  // Unicode code point escape sequence comprises four hex characters
  constexpr size_type unicode_hex_digits = 4;

  // Prepare result
  int64_t result = 0, base = 1;

  // Iterate over hex digits right-to-left
  size_type index = unicode_hex_digits;
  while (index-- > 0) {
    char const ch = str[index];
    if (ch >= '0' && ch <= '9') {
      result += static_cast<int64_t>((ch - '0') + 0) * base;
      base *= 16;
    } else if (ch >= 'A' && ch <= 'F') {
      result += static_cast<int64_t>((ch - 'A') + 10) * base;
      base *= 16;
    } else if (ch >= 'a' && ch <= 'f') {
      result += static_cast<int64_t>((ch - 'a') + 10) * base;
      base *= 16;
    } else {
      return -1;
    }
  }
  return result;
}

template <typename str_tuple_it, typename B>
std::unique_ptr<column> parse_data(str_tuple_it str_tuples,
                                   size_type col_size,
                                   data_type col_type,
                                   B&& null_mask,
                                   cudf::io::parse_options_view const& options,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  if (col_type == cudf::data_type{cudf::type_id::STRING}) {
    rmm::device_uvector<size_type> offsets(col_size + 1, stream);
    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<size_type>(0),
      col_size,
      [str_tuples,
       sizes     = device_span<size_type>{offsets},
       null_mask = static_cast<bitmask_type*>(null_mask.data()),
       options] __device__(size_type row) {
        if (not bit_is_set(null_mask, row)) {
          sizes[row] = 0;
          return;
        }
        auto const in = str_tuples[row];

        auto const is_null_literal =
          serialized_trie_contains(options.trie_na, {in.first, static_cast<size_t>(in.second)});
        if (is_null_literal) {
          sizes[row] = 0;
          clear_bit(null_mask, row);
          return;
        }

        // Whether in the original JSON this was a string value enclosed in quotes
        // ({"a":"foo"} vs. {"a":1.23})
        char const quote_char = options.quotechar;
        bool const is_string_value =
          in.second >= 2 && (*in.first == quote_char) && (in.first[in.second - 1] == quote_char);

        // Handling non-string values
        if (not is_string_value) { sizes[row] = in.second; }

        // Strip off quote chars
        decltype(in.second) out_size = 0;

        // Escape-flag, set after encountering an escape character
        bool escape = false;

        // Exclude beginning and ending quote chars from string range
        auto start_index = options.keepquotes ? 0 : 1;
        auto end_index   = in.second - (options.keepquotes ? 0 : 1);
        for (decltype(in.second) i = start_index; i < end_index; ++i) {
          // Previous char was an escape char
          if (escape) {
            // A unicode code point escape sequence is \uXXXX
            auto constexpr NUM_UNICODE_ESC_SEQ_CHARS = 6;
            // The escape sequence comprises four hex digits
            auto constexpr NUM_UNICODE_ESC_HEX_DIGITS = 4;
            // A name for the char following the current one
            auto constexpr NEXT_CHAR = 1;
            // A name for the char after the next char
            auto constexpr NEXT_NEXT_CHAR = 2;
            // Reset escape flag for next loop iteration
            escape = false;

            // Check the character that is supposed to be escaped
            auto escaped_char = get_escape_char(in.first[i]);

            // This is an escape sequence of a unicode code point: \uXXXX,
            // where each X in XXXX represents a hex digit
            if (escaped_char == UNICODE_SEQ) {
              // Make sure that there's at least 4 characters left from the
              // input, which are expected to be hex digits
              if (i + NUM_UNICODE_ESC_HEX_DIGITS < end_index) {
                auto hex_val = string_to_hex(&in.first[i + NEXT_CHAR]);
                if (hex_val < 0) {
                  // TODO signal parsing error: not all 4 hex digits
                  continue;
                }
                // Skip over the four hex digits
                i += NUM_UNICODE_ESC_HEX_DIGITS;

                // If this may be a UTF-16 encoded surrogate pair:
                // we expect another \uXXXX sequence
                if (hex_val >= 0xD800 && i + NUM_UNICODE_ESC_SEQ_CHARS < end_index &&
                    in.first[i + NEXT_CHAR] == '\\' && in.first[i + NEXT_NEXT_CHAR] == 'u') {
                  auto hex_low_val = string_to_hex(&in.first[i + 3]);
                  if (hex_val < 0xD800 || hex_low_val < 0xDC00) {
                    // TODO signal parsing error: not all 4 hex digits
                    continue;
                  }
                  // Skip over the second \uXXXX sequence
                  i += NUM_UNICODE_ESC_SEQ_CHARS;
                  uint32_t unicode_code_point =
                    0x10000 + (hex_val - 0xD800) + (hex_low_val - 0xDC00);
                  auto utf8_chars = strings::detail::codepoint_to_utf8(unicode_code_point);
                  out_size += strings::detail::bytes_in_char_utf8(utf8_chars);
                }
                // Just a single \uXXXX sequence
                else {
                  auto utf8_chars = strings::detail::codepoint_to_utf8(hex_val);
                  out_size += strings::detail::bytes_in_char_utf8(utf8_chars);
                }
              } else {
                // TODO signal parsing error: expected 4 hex digits
              }
            } else if (escaped_char == NON_ESCAPE_CHAR) {
              // TODO signal parsing error: this char does not need to be escape
            } else {
              out_size++;
            }
          } else {
            escape = in.first[i] == '\\';
            out_size += escape ? 0 : 1;
          }
        }
        if (escape) {
          // TODO signal parsing error: last char was escape, not followed by
          // anything to escape
        }
        sizes[row] = out_size;
      });

    thrust::exclusive_scan(
      rmm::exec_policy(stream), offsets.begin(), offsets.end(), offsets.begin());

    rmm::device_uvector<char> chars(offsets.back_element(stream), stream);
    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<size_type>(0),
      col_size,
      [str_tuples,
       chars     = device_span<char>{chars},
       offsets   = device_span<size_type>{offsets},
       null_mask = static_cast<bitmask_type*>(null_mask.data()),
       options] __device__(size_type row) {
        if (not bit_is_set(null_mask, row)) { return; }
        auto const in = str_tuples[row];

        // Whether in the original JSON this was a string value enclosed in quotes
        // ({"a":"foo"} vs. {"a":1.23})
        char const quote_char = options.quotechar;
        bool const is_string_value =
          in.second >= 2 && (*in.first == quote_char) && (in.first[in.second - 1] == quote_char);

        // Copy literal/numeric value
        if (!is_string_value) {
          for (int i = 0, j = 0; i < in.second; ++i) {
            chars[offsets[row] + j] = *(in.first + i);
            j++;
          }
        }

        // Escape-flag, set after encountering an escape character
        bool escape = false;

        // Exclude beginning and ending quote chars from string range
        auto start_index = options.keepquotes ? 0 : 1;
        auto end_index   = in.second - (options.keepquotes ? 0 : 1);

        for (int i = start_index, j = 0; i < end_index; ++i) {
          // Previous char was escape char
          if (escape) {
            // A unicode code point escape sequence is \uXXXX
            auto constexpr NUM_UNICODE_ESC_SEQ_CHARS = 6;
            // The escape sequence comprises four hex digits
            auto constexpr NUM_UNICODE_ESC_HEX_DIGITS = 4;
            // A name for the char following the current one
            auto constexpr NEXT_CHAR = 1;
            // A name for the char after the next char
            auto constexpr NEXT_NEXT_CHAR = 2;
            // Reset escape flag for next loop iteration
            escape = false;

            // Check the character that is supposed to be escaped
            auto escaped_char = get_escape_char(in.first[i]);

            // This is an escape sequence of a unicode code point: \uXXXX,
            // where each X in XXXX represents a hex digit
            if (escaped_char == UNICODE_SEQ) {
              //  Make sure that there's at least 4 characters left from the
              //  input, which are expected to be hex digits
              if (i + NUM_UNICODE_ESC_HEX_DIGITS < end_index) {
                auto hex_val = string_to_hex(&in.first[i + NEXT_CHAR]);
                if (hex_val < 0) {
                  // TODO signal parsing error: not all 4 hex digits
                  continue;
                }
                // Skip over the four hex digits
                i += NUM_UNICODE_ESC_HEX_DIGITS;

                // If this may be a UTF-16 encoded surrogate pair:
                // we expect another \uXXXX sequence
                if (hex_val >= 0xD800 && i + NUM_UNICODE_ESC_SEQ_CHARS < end_index &&
                    in.first[i + NEXT_CHAR] == '\\' && in.first[i + NEXT_NEXT_CHAR] == 'u') {
                  auto hex_low_val = string_to_hex(&in.first[i + 3]);
                  if (hex_val < 0xD800 || hex_low_val < 0xDC00) {
                    // TODO signal parsing error: not all 4 hex digits
                    continue;
                  }
                  // Skip over the second \uXXXX sequence
                  i += NUM_UNICODE_ESC_SEQ_CHARS;
                  uint32_t unicode_code_point =
                    0x10000 + ((hex_val - 0xD800) << 10) + (hex_low_val - 0xDC00);
                  auto utf8_chars = strings::detail::codepoint_to_utf8(unicode_code_point);
                  j += strings::detail::from_char_utf8(utf8_chars, &chars[offsets[row] + j]);
                }
                // Just a single \uXXXX sequence
                else {
                  auto utf8_chars = strings::detail::codepoint_to_utf8(hex_val);
                  j += strings::detail::from_char_utf8(utf8_chars, &chars[offsets[row] + j]);
                }
              } else {
                // TODO signal parsing error: expected 4 hex digits
              }
            } else if (escaped_char == NON_ESCAPE_CHAR) {
              // TODO signal parsing error: this char does not need to be escape
            } else {
              chars[offsets[row] + j] = escaped_char;
              j++;
            }
          } else {
            escape = in.first[i] == '\\';
            if (!escape) {
              chars[offsets[row] + j] = *(in.first + i);
              j++;
            }
          }
        }
        if (escape) {
          // TODO signal parsing error: last char was escape, not followed by
          // anything to escape
        }
      });

    return make_strings_column(
      col_size, std::move(offsets), std::move(chars), std::move(null_mask));
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

      auto const is_parsed = cudf::type_dispatcher(col_type,
                                                   ConvertFunctor{},
                                                   in.first,
                                                   in.first + in.second,
                                                   col.data<char>(),
                                                   row,
                                                   col_type,
                                                   options,
                                                   false);
      if (not is_parsed) { col.set_null(row); }
    });

  return out_col;
}

}  // namespace cudf::io::json::experimental
