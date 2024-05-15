/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "text/subword/detail/data_normalizer.hpp"
#include "text/subword/detail/tokenizer_utils.cuh"
#include "text/utilities/tokenize_ops.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtext/normalize.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <limits>

namespace nvtext {
namespace detail {
namespace {
/**
 * @brief Normalize spaces in a strings column.
 *
 * Repeated whitespace (code-point <= ' ') is replaced with a single space.
 * Also, whitespace is trimmed from the beginning and end of each string.
 *
 * This functor can be called to compute the output size in bytes
 * of each string and then called again to fill in the allocated buffer.
 */
struct normalize_spaces_fn {
  cudf::column_device_view const d_strings;  // strings to normalize
  cudf::size_type* d_sizes{};                // size of each output row
  char* d_chars{};                           // output buffer for characters
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    cudf::string_view const single_space(" ", 1);
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    char* buffer     = d_chars ? d_chars + d_offsets[idx] : nullptr;
    char* optr       = buffer;  // running output pointer

    cudf::size_type nbytes = 0;  // holds the number of bytes per output string

    // create a tokenizer for this string with whitespace delimiter (default)
    characters_tokenizer tokenizer(d_str);

    // this will retrieve tokens automatically skipping runs of whitespace
    while (tokenizer.next_token()) {
      auto const token_pos = tokenizer.token_byte_positions();
      auto const token =
        cudf::string_view(d_str.data() + token_pos.first, token_pos.second - token_pos.first);
      if (optr) {
        // prepend space unless we are at the beginning
        if (optr != buffer) { optr = cudf::strings::detail::copy_string(optr, single_space); }
        // write token to output buffer
        thrust::copy_n(thrust::seq, token.data(), token.size_bytes(), optr);
        optr += token.size_bytes();
      }
      nbytes += token.size_bytes() + 1;  // token size plus a single space
    }
    // remove trailing space
    if (!d_chars) { d_sizes[idx] = (nbytes > 0) ? nbytes - 1 : 0; }
  }
};

// code-point to multi-byte range limits
constexpr uint32_t UTF8_1BYTE = 0x0080;
constexpr uint32_t UTF8_2BYTE = 0x0800;
constexpr uint32_t UTF8_3BYTE = 0x01'0000;

/**
 * @brief Convert code-point arrays into UTF-8 bytes for each string.
 */
struct codepoint_to_utf8_fn {
  cudf::column_device_view const d_strings;  // input strings
  uint32_t const* cp_data;                   // full code-point array
  int64_t const* d_cp_offsets{};             // offsets to each string's code-point array
  cudf::size_type* d_sizes{};                // size of output string
  char* d_chars{};                           // buffer for the output strings column
  cudf::detail::input_offsetalator d_offsets;

  /**
   * @brief Return the number of bytes for the output string given its code-point array.
   *
   * @param str_cps code-points for the string
   * @param count number of code-points in `str_cps`
   * @return Number of bytes required for the output
   */
  __device__ cudf::size_type compute_output_size(uint32_t const* str_cps, uint32_t count)
  {
    return thrust::transform_reduce(
      thrust::seq,
      str_cps,
      str_cps + count,
      [](auto cp) { return 1 + (cp >= UTF8_1BYTE) + (cp >= UTF8_2BYTE) + (cp >= UTF8_3BYTE); },
      0,
      thrust::plus());
  }

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const offset = d_cp_offsets[idx];
    auto const count  = d_cp_offsets[idx + 1] - offset;  // number of code-points
    auto str_cps      = cp_data + offset;                // code-points for this string
    if (!d_chars) {
      d_sizes[idx] = compute_output_size(str_cps, count);
      return;
    }
    // convert each code-point to 1-4 UTF-8 encoded bytes
    char* out_ptr = d_chars + d_offsets[idx];
    for (uint32_t jdx = 0; jdx < count; ++jdx) {
      uint32_t code_point = *str_cps++;
      if (code_point < UTF8_1BYTE)  // ASCII range
        *out_ptr++ = static_cast<char>(code_point);
      else if (code_point < UTF8_2BYTE) {  // create two-byte UTF-8
        // b00001xxx:byyyyyyyy => b110xxxyy:b10yyyyyy
        *out_ptr++ = static_cast<char>((((code_point << 2) & 0x00'1F00) | 0x00'C000) >> 8);
        *out_ptr++ = static_cast<char>((code_point & 0x3F) | 0x0080);
      } else if (code_point < UTF8_3BYTE) {  // create three-byte UTF-8
        // bxxxxxxxx:byyyyyyyy => b1110xxxx:b10xxxxyy:b10yyyyyy
        *out_ptr++ = static_cast<char>((((code_point << 4) & 0x0F'0000) | 0x00E0'0000) >> 16);
        *out_ptr++ = static_cast<char>((((code_point << 2) & 0x00'3F00) | 0x00'8000) >> 8);
        *out_ptr++ = static_cast<char>((code_point & 0x3F) | 0x0080);
      } else {  // create four-byte UTF-8
        // maximum code-point value is 0x0011'0000
        // b000xxxxx:byyyyyyyy:bzzzzzzzz => b11110xxx:b10xxyyyy:b10yyyyzz:b10zzzzzz
        *out_ptr++ = static_cast<char>((((code_point << 6) & 0x0700'0000u) | 0xF000'0000u) >> 24);
        *out_ptr++ = static_cast<char>((((code_point << 4) & 0x003F'0000u) | 0x0080'0000u) >> 16);
        *out_ptr++ = static_cast<char>((((code_point << 2) & 0x00'3F00u) | 0x00'8000u) >> 8);
        *out_ptr++ = static_cast<char>((code_point & 0x3F) | 0x0080);
      }
    }
  }
};

}  // namespace

// detail API
std::unique_ptr<cudf::column> normalize_spaces(cudf::strings_column_view const& strings,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  if (strings.is_empty()) return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  // create device column
  auto d_strings = cudf::column_device_view::create(strings.parent(), stream);

  // build offsets and children using the normalize_space_fn
  auto [offsets_column, chars] = cudf::strings::detail::make_strings_children(
    normalize_spaces_fn{*d_strings}, strings.size(), stream, mr);

  return cudf::make_strings_column(strings.size(),
                                   std::move(offsets_column),
                                   chars.release(),
                                   strings.null_count(),
                                   cudf::detail::copy_bitmask(strings.parent(), stream, mr));
}

/**
 * @copydoc nvtext::normalize_characters
 */
std::unique_ptr<cudf::column> normalize_characters(cudf::strings_column_view const& strings,
                                                   bool do_lower_case,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  if (strings.is_empty()) return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  // create the normalizer and call it
  auto result = [&] {
    auto const cp_metadata = get_codepoint_metadata(stream);
    auto const aux_table   = get_aux_codepoint_data(stream);
    auto const normalizer  = data_normalizer(cp_metadata.data(), aux_table.data(), do_lower_case);
    return normalizer.normalize(strings, stream);
  }();

  CUDF_EXPECTS(
    result.first->size() < static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "output exceeds the column size limit",
    std::overflow_error);

  // convert the result into a strings column
  // - the cp_chars are the new 4-byte code-point values for all the characters in the output
  // - the cp_offsets identify which code-points go with which strings
  auto const cp_chars   = result.first->data();
  auto const cp_offsets = result.second->data();

  auto d_strings = cudf::column_device_view::create(strings.parent(), stream);

  // build offsets and children using the codepoint_to_utf8_fn
  auto [offsets_column, chars] = cudf::strings::detail::make_strings_children(
    codepoint_to_utf8_fn{*d_strings, cp_chars, cp_offsets}, strings.size(), stream, mr);

  return cudf::make_strings_column(strings.size(),
                                   std::move(offsets_column),
                                   chars.release(),
                                   strings.null_count(),
                                   cudf::detail::copy_bitmask(strings.parent(), stream, mr));
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> normalize_spaces(cudf::strings_column_view const& input,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_spaces(input, stream, mr);
}

/**
 * @copydoc nvtext::normalize_characters
 */
std::unique_ptr<cudf::column> normalize_characters(cudf::strings_column_view const& input,
                                                   bool do_lower_case,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_characters(input, do_lower_case, stream, mr);
}

}  // namespace nvtext
