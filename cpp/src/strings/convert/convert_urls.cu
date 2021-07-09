/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/convert/convert_urls.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>

using cudf::device_span;

namespace cudf {
namespace strings {
namespace detail {
namespace {
//
// This is the functor for the url_encode() method below.
// Specific requirements are documented in custrings issue #321.
// In summary it converts mostly non-ascii characters and control characters into UTF-8 hex
// characters prefixed with '%'. For example, the space character must be converted to characters
// '%20' where the '20' indicates the hex value for space in UTF-8. Likewise, multi-byte characters
// are converted to multiple hex characters. For example, the é character is converted to characters
// '%C3%A9' where 'C3A9' is the UTF-8 bytes xc3a9 for this character.
//
struct url_encoder_fn {
  column_device_view const d_strings;
  int32_t const* d_offsets{};
  char* d_chars{};

  // utility to create 2-byte hex characters from single binary byte
  __device__ void byte_to_hex(uint8_t byte, char* hex)
  {
    hex[0] = '0';
    if (byte >= 16) {
      uint8_t hibyte = byte / 16;
      hex[0]         = hibyte < 10 ? '0' + hibyte : 'A' + (hibyte - 10);
      byte           = byte - (hibyte * 16);
    }
    hex[1] = byte < 10 ? '0' + byte : 'A' + (byte - 10);
  }

  __device__ bool should_not_url_encode(char ch)
  {
    return (
      (ch >= '0' && ch <= '9') ||  // these are the characters
      (ch >= 'A' && ch <= 'Z') ||  // that are not to be url encoded
      (ch >= 'a' &&
       ch <= 'z') ||  // reference: docs.python.org/3/library/urllib.parse.html#urllib.parse.quote
      (ch == '.') ||
      (ch == '_') || (ch == '~') || (ch == '-'));
  }

  // main part of the functor the performs the url-encoding
  __device__ size_type operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    string_view d_str = d_strings.element<string_view>(idx);
    //
    char* out_ptr    = d_chars ? d_chars + d_offsets[idx] : nullptr;
    size_type nbytes = 0;
    char hex[2];  // two-byte hex max
    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      auto ch = *itr;
      if (ch < 128) {
        if (should_not_url_encode(static_cast<char>(ch))) {
          nbytes++;
          if (out_ptr) out_ptr = copy_and_increment(out_ptr, d_str.data() + itr.byte_offset(), 1);
        } else  // url-encode everything else
        {
          nbytes += 3;
          if (out_ptr) {
            out_ptr = copy_and_increment(out_ptr, "%", 1);  // add the '%' prefix
            byte_to_hex(static_cast<uint8_t>(ch), hex);     // convert to 2 hex chars
            out_ptr = copy_and_increment(out_ptr, hex, 2);  // add them to the output
          }
        }
      } else  // these are to be utf-8 url-encoded
      {
        uint8_t char_bytes[4];  // holds utf-8 bytes for one character
        size_type char_width = from_char_utf8(ch, reinterpret_cast<char*>(char_bytes));
        nbytes += char_width * 3;  // '%' plus 2 hex chars per byte (example: é is %C3%A9)
        // process each byte in this current character
        for (size_type chidx = 0; out_ptr && (chidx < char_width); ++chidx) {
          out_ptr = copy_and_increment(out_ptr, "%", 1);  // add '%' prefix
          byte_to_hex(char_bytes[chidx], hex);            // convert to 2 hex chars
          out_ptr = copy_and_increment(out_ptr, hex, 2);  // add them to the output
        }
      }
    }
    return nbytes;
  }
};

}  // namespace

//
std::unique_ptr<column> url_encode(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(data_type{type_id::STRING});

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  // copy null mask
  rmm::device_buffer null_mask = cudf::detail::copy_bitmask(strings.parent(), stream, mr);
  // build offsets column
  auto offsets_transformer_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_type>(0), url_encoder_fn{d_strings});
  auto offsets_column = make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);
  auto d_offsets = offsets_column->view().data<int32_t>();
  auto const bytes =
    cudf::detail::get_value<int32_t>(offsets_column->view(), strings_count, stream);
  // build chars column
  auto chars_column = create_chars_child_column(bytes, stream, mr);
  auto d_chars      = chars_column->mutable_view().data<char>();
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     strings_count,
                     url_encoder_fn{d_strings, d_offsets, d_chars});

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             strings.null_count(),
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail

// external API
std::unique_ptr<column> url_encode(strings_column_view const& strings,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::url_encode(strings, rmm::cuda_stream_default, mr);
}

namespace detail {
namespace {

// utility to convert a hex char into a single byte
constexpr uint8_t hex_char_to_byte(char ch)
{
  if (ch >= '0' && ch <= '9') return (ch - '0');
  if (ch >= 'A' && ch <= 'F') return (ch - 'A' + 10);  // in hex A=10,B=11,...,F=15
  if (ch >= 'a' && ch <= 'f') return (ch - 'a' + 10);  // same for lower case
  return 0;
}

constexpr bool is_hex_digit(char ch)
{
  return (ch >= '0' && ch <= '9') || (ch >= 'A' && ch <= 'F') || (ch >= 'a' && ch <= 'f');
}

/**
 * @brief Functor for detecting character escape sequences in URL-encoded strings.
 */
struct url_decode_escape_detector {
  device_span<char const> const d_chars{};

  /**
   * @brief Detects if an escape sequence is at the specified character position.
   *
   * Returns true for a character index corresponding to the start of an escape
   * sequence, i.e.: '%' followed by two hexadecimal digits. This does not check
   * against string boundaries and therefore can produce false-positives.
   *
   * @param char_idx Character position to examine
   * @return true An escape sequence was detected at the character position
   * @return false No escape sequence at the character position
   */
  __device__ bool operator()(size_type char_idx) const
  {
    return (char_idx + 2 < d_chars.size()) && d_chars[char_idx] == '%' &&
           is_hex_digit(d_chars[char_idx + 1]) && is_hex_digit(d_chars[char_idx + 2]);
  }
};

/**
 * @brief Functor for detecting escape sequence positions that cross a string boundary.
 */
struct url_decode_esc_position_filter {
  device_span<int32_t const> const d_offsets{};

  /**
   * @brief Detects if an escape sequence crosses a string boundary
   *
   * Returns true for an escape sequence that straddles a boundary between two strings.
   *
   * @param esc_pos_idx Character position corresponding to the start of an escape sequence
   * @return true The escape sequence crosses a string boundary
   * @return false The escape sequence does not cross a string boundary
   */
  __device__ bool operator()(size_type esc_pos_idx) const
  {
    // find the end offset of the current string
    size_type const* offset_ptr =
      thrust::upper_bound(thrust::seq, d_offsets.begin(), d_offsets.end(), esc_pos_idx);
    return esc_pos_idx + 2 >= *offset_ptr;
  }
};

/**
 * @brief Functor for replacing character escape sequences in URL-encoded strings.
 *
 * Each escape sequence interprets the following 2 characters as hex values to create the output
 * byte. For example, the sequence '%20' is converted into byte (0x20) which is a single space
 * character. Another example converts '%C3%A9' into 2 sequential bytes (0xc3 and 0xa9
 * respectively). Overall, 3 characters are converted into one byte whenever a '%' character
 * is encountered in the string.
 */
struct url_decode_char_replacer {
  device_span<size_type const> const d_esc_positions{};
  char const* const d_in_chars{};
  char* const d_out_chars{};
  int32_t const first_input_char_offset = 0;

  /**
   * @brief Copy an input character to the output, decoding escape sequences
   *
   * Each character position is examined against the list of known escape sequence positions.
   * If the position is not within an escape sequence then the input character is copied to the
   * output. If the position is the start of an escape sequence then the sequence is decoded to
   * produce the character copied to the output. Any characters after the start of an escape
   * sequence but still within the escape sequence are discarded.
   *
   * @param input_idx The input character position to process
   */
  __device__ void operator()(size_type input_idx) const
  {
    char ch = d_in_chars[input_idx];

    // determine the number of escape sequences at or before this character position
    size_type const* next_esc_pos_ptr =
      thrust::upper_bound(thrust::seq, d_esc_positions.begin(), d_esc_positions.end(), input_idx);
    size_type num_prev_esc = next_esc_pos_ptr - d_esc_positions.data();

    // every escape that occurs before this one replaces 3 characters with 1
    size_type output_idx = input_idx - (num_prev_esc * 2) - first_input_char_offset;
    if (num_prev_esc > 0) {
      size_type prev_esc_pos = *(next_esc_pos_ptr - 1);
      // find the previous escape to see if this character is within the escape sequence
      if (input_idx - prev_esc_pos < 3) {
        if (input_idx == prev_esc_pos) {  // at a position that needs to be replaced
          ch = (hex_char_to_byte(d_in_chars[input_idx + 1]) << 4) |
               hex_char_to_byte(d_in_chars[input_idx + 2]);
          // previous escape sequence is this position, so the original calculation over-adjusted
          output_idx += 2;
        } else {
          // one of the escape hex digits that has no corresponding character in the output
          return;
        }
      }
    }

    d_out_chars[output_idx] = ch;
  }
};

/**
 * @brief Functor to update the string column offsets.
 */
struct url_decode_offsets_updater {
  device_span<size_type const> const d_esc_positions{};
  int32_t const first_input_offset = 0;

  /**
   * @brief Convert input offsets into output offsets
   *
   * Each offset is reduced by 2 for every escape sequence that occurs in the entire string column
   * character data before the offset, as 3 characters are replaced with 1 for each escape.
   *
   * @param offset An original offset value from the input string column
   * @return Adjusted offset value
   */
  __device__ int32_t operator()(int32_t offset) const
  {
    // determine the number of escape sequences occurring before this offset
    size_type const* next_esc_pos_ptr =
      thrust::lower_bound(thrust::seq, d_esc_positions.begin(), d_esc_positions.end(), offset);
    size_type num_prev_esc = next_esc_pos_ptr - d_esc_positions.data();
    // every escape that occurs before this one replaces 3 characters with 1
    return offset - first_input_offset - (num_prev_esc * 2);
  }
};

}  // namespace

//
std::unique_ptr<column> url_decode(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(data_type{type_id::STRING});

  auto offset_count = strings_count + 1;
  auto d_offsets    = strings.offsets().data<int32_t>() + strings.offset();
  auto d_in_chars   = strings.chars().data<char>();
  // determine index of first character in base column
  size_type chars_start = (strings.offset() == 0) ? 0
                                                  : cudf::detail::get_value<int32_t>(
                                                      strings.offsets(), strings.offset(), stream);
  size_type chars_end   = (offset_count == strings.offsets().size())
                            ? strings.chars_size()
                            : cudf::detail::get_value<int32_t>(
                              strings.offsets(), strings.offset() + strings_count, stream);
  size_type chars_bytes = chars_end - chars_start;

  url_decode_escape_detector esc_detector{device_span<char const>(d_in_chars, chars_end)};

  // Count the number of URL escape sequences across all strings, ignoring string boundaries.
  // This may count more sequences than actually are there since string boundaries are ignored.
  size_type esc_count = thrust::count_if(rmm::exec_policy(stream),
                                         thrust::make_counting_iterator<size_type>(chars_start),
                                         thrust::make_counting_iterator<size_type>(chars_end),
                                         esc_detector);

  if (esc_count == 0) {
    // nothing to replace, so just copy the input column
    return std::make_unique<cudf::column>(strings.parent(), stream, mr);
  }

  // create a vector of the potential escape sequence positions
  rmm::device_uvector<size_type> esc_positions(esc_count, stream);
  auto d_esc_positions = esc_positions.data();
  thrust::copy_if(rmm::exec_policy(stream),
                  thrust::make_counting_iterator<size_t>(chars_start),
                  thrust::make_counting_iterator<size_t>(chars_end),
                  d_esc_positions,
                  esc_detector);

  // In-place remove any escape positions that crossed string boundaries.
  device_span<int32_t const> d_offsets_span(d_offsets, offset_count);
  auto esc_pos_end = thrust::remove_if(rmm::exec_policy(stream),
                                       d_esc_positions,
                                       d_esc_positions + esc_count,
                                       url_decode_esc_position_filter{d_offsets_span});

  // update count in case any were filtered
  esc_count = esc_pos_end - d_esc_positions;
  if (esc_count == 0) {
    // nothing to replace, so just copy the input column
    return std::make_unique<cudf::column>(strings.parent(), stream, mr);
  }

  device_span<size_type const> d_esc_positions_span(d_esc_positions, esc_count);

  // build offsets column
  auto offsets_column = make_numeric_column(
    data_type{type_id::INT32}, offset_count, mask_state::UNALLOCATED, stream, mr);
  auto offsets_view = offsets_column->mutable_view();
  thrust::transform(rmm::exec_policy(stream),
                    d_offsets_span.begin(),
                    d_offsets_span.end(),
                    offsets_view.begin<int32_t>(),
                    url_decode_offsets_updater{d_esc_positions_span, chars_start});

  // create the chars column
  auto chars_column =
    create_chars_child_column(chars_bytes - (esc_count * 2),  // replacing 3 bytes with 1
                              stream,
                              mr);
  auto d_out_chars = chars_column->mutable_view().data<char>();
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(chars_start),
    chars_bytes,
    url_decode_char_replacer{d_esc_positions_span, d_in_chars, d_out_chars, chars_start});

  // free the escape positions buffer as it is no longer needed
  (void)esc_positions.release();

  // copy null mask
  rmm::device_buffer null_mask = cudf::detail::copy_bitmask(strings.parent(), stream, mr);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             strings.null_count(),
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> url_decode(strings_column_view const& strings,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::url_decode(strings, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
