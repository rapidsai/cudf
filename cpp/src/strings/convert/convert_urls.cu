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
#include <cudf/detail/utilities/cuda.cuh>
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

#include <algorithm>

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
  if (strings_count == 0) return make_empty_strings_column(stream, mr);

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
  auto chars_column = create_chars_child_column(strings_count, bytes, stream, mr);
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

__forceinline__ __device__ bool is_escape_char(char const* const ptr)
{
  return (ptr[0] == '%' && is_hex_digit(ptr[1]) && is_hex_digit(ptr[2]));
}

template <int warps_per_threadblock, int block_size>
__global__ void url_decode_char_replacer(char const* const in_chars,
                                         int32_t const* const in_offsets,
                                         char* const out_chars,
                                         int32_t const* const out_offsets,
                                         int32_t const num_rows)
{
  __shared__ char temporary_buffer[warps_per_threadblock][block_size + 4];
  __shared__ typename cub::WarpScan<int8_t>::TempStorage cub_storage[warps_per_threadblock];
  __shared__ int out_idx[warps_per_threadblock];

  int global_thread_id  = blockIdx.x * blockDim.x + threadIdx.x;
  int global_warp_id    = global_thread_id / cudf::detail::warp_size;
  int local_warp_id     = threadIdx.x / cudf::detail::warp_size;
  int warp_lane         = threadIdx.x % cudf::detail::warp_size;
  int nwarps            = gridDim.x * blockDim.x / cudf::detail::warp_size;
  char* in_chars_shared = temporary_buffer[local_warp_id];

  for (size_type row_idx = global_warp_id; row_idx < num_rows; row_idx += nwarps) {
    auto in_chars_string  = in_chars + in_offsets[row_idx];
    auto out_chars_string = out_chars + out_offsets[row_idx];
    auto string_length    = in_offsets[row_idx + 1] - in_offsets[row_idx];
    int nblocks           = (string_length + block_size - 1) / block_size;

    for (int iblock = 0; iblock < nblocks; iblock++) {
      int string_length_iblock = std::min(block_size, string_length - block_size * iblock);

      // Each warp collectively loads input characters of the current block to shared memory
      for (int ichar = warp_lane; ichar < string_length_iblock + 4;
           ichar += cudf::detail::warp_size) {
        char ch    = 0;
        int in_idx = iblock * block_size + ichar - 2;
        if (in_idx >= 0 && in_idx < string_length) ch = in_chars_string[in_idx];
        in_chars_shared[ichar] = ch;
      }

      if (warp_lane == cudf::detail::warp_size - 1) { out_idx[local_warp_id] = 0; }

      __syncwarp();

      for (int ichar_start = 0; ichar_start < string_length_iblock;
           ichar_start += cudf::detail::warp_size) {
        int ichar = ichar_start + warp_lane;
        int8_t out_size =
          (ichar >= string_length_iblock || is_escape_char(in_chars_shared + ichar) ||
           is_escape_char(in_chars_shared + ichar + 1))
            ? 0
            : 1;
        int8_t out_offset;

        cub::WarpScan<int8_t>(cub_storage[local_warp_id]).ExclusiveSum(out_size, out_offset);

        if (out_size == 1) {
          char ch;
          if (is_escape_char(in_chars_shared + ichar + 2)) {
            ch = (hex_char_to_byte(in_chars_shared[ichar + 3]) << 4) |
                 hex_char_to_byte(in_chars_shared[ichar + 4]);
          } else {
            ch = in_chars_shared[ichar + 2];
          }
          out_chars_string[out_idx[local_warp_id] + out_offset] = ch;
        }

        if (warp_lane == cudf::detail::warp_size - 1) {
          out_idx[local_warp_id] += (out_offset + out_size);
        };

        __syncwarp();
      }
    }
  }
}

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
  if (strings_count == 0) return make_empty_strings_column(stream, mr);

  auto offset_count = strings_count + 1;
  auto d_offsets    = strings.offsets().data<int32_t>() + strings.offset();
  auto d_in_chars   = strings.chars().data<char>();
  // determine index of first character in base column
  size_type chars_start = (strings.offset() == 0) ? 0
                                                  : cudf::detail::get_value<int32_t>(
                                                      strings.offsets(), strings.offset(), stream);
  size_type chars_end = (offset_count == strings.offsets().size())
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
    create_chars_child_column(strings_count,
                              chars_bytes - (esc_count * 2),  // replacing 3 bytes with 1
                              stream,
                              mr);
  auto d_out_chars = chars_column->mutable_view().data<char>();

  url_decode_char_replacer<4, 284><<<65536, 128>>>(
    d_in_chars, d_offsets, d_out_chars, offsets_column->view().begin<int32_t>(), strings_count);

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
