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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <strings/utilities.cuh>

#include <nvtext/normalize.hpp>
#include <text/subword/detail/data_normalizer.hpp>
#include <text/utilities/tokenize_ops.cuh>

#include <thrust/for_each.h>
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
  int32_t const* d_offsets{};                // offsets into d_buffer
  char* d_buffer{};                          // output buffer for characters

  __device__ int32_t operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    cudf::string_view single_space(" ", 1);
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    char* buffer     = d_offsets ? d_buffer + d_offsets[idx] : nullptr;
    char* optr       = buffer;  // running output pointer
    int32_t nbytes   = 0;       // holds the number of bytes per output string
    // create tokenizer for this string with whitespace delimiter (default)
    characters_tokenizer tokenizer(d_str);
    // this will retrieve tokens automatically skipping runs of whitespace
    while (tokenizer.next_token()) {
      auto token_pos = tokenizer.token_byte_positions();
      nbytes += token_pos.second - token_pos.first + 1;  // token size plus a single space
      if (optr) {
        cudf::string_view token(d_str.data() + token_pos.first, token_pos.second - token_pos.first);
        if (optr != buffer)  // prepend space unless we are at the beginning
          optr = cudf::strings::detail::copy_string(optr, single_space);
        // write token to output buffer
        optr = cudf::strings::detail::copy_string(optr, token);  // copy token to output
      }
    }
    return (nbytes > 0) ? nbytes - 1 : 0;  // remove trailing space
  }
};

// code-point to multi-byte range limits
constexpr uint32_t UTF8_1BYTE = 0x0080;
constexpr uint32_t UTF8_2BYTE = 0x0800;
constexpr uint32_t UTF8_3BYTE = 0x010000;

/**
 * @brief Convert code-point arrays into UTF-8 bytes for each string.
 */
struct codepoint_to_utf8_fn {
  cudf::column_device_view const d_strings;  // input strings
  uint32_t const* cp_data;                   // full code-point array
  int32_t const* d_cp_offsets{};             // offsets to each string's code-point array
  int32_t const* d_offsets{};                // offsets for the output strings
  char* d_chars{};                           // buffer for the output strings column

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
      thrust::plus<cudf::size_type>());
  }

  __device__ cudf::size_type operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    auto const d_str  = d_strings.element<cudf::string_view>(idx);
    auto const offset = d_cp_offsets[idx];
    auto const count  = d_cp_offsets[idx + 1] - offset;  // number of code-points
    auto str_cps      = cp_data + offset;                // code-points for this string
    if (!d_chars) return compute_output_size(str_cps, count);
    // convert each code-point to 1-4 UTF-8 encoded bytes
    char* out_ptr = d_chars + d_offsets[idx];
    for (uint32_t jdx = 0; jdx < count; ++jdx) {
      uint32_t code_point = *str_cps++;
      if (code_point < UTF8_1BYTE)  // ASCII range
        *out_ptr++ = static_cast<char>(code_point);
      else if (code_point < UTF8_2BYTE) {  // create two-byte UTF-8
        // b00001xxx:byyyyyyyy => b110xxxyy:b10yyyyyy
        *out_ptr++ = static_cast<char>((((code_point << 2) & 0x001F00) | 0x00C000) >> 8);
        *out_ptr++ = static_cast<char>((code_point & 0x3F) | 0x0080);
      } else if (code_point < UTF8_3BYTE) {  // create three-byte UTF-8
        // bxxxxxxxx:byyyyyyyy => b1110xxxx:b10xxxxyy:b10yyyyyy
        *out_ptr++ = static_cast<char>((((code_point << 4) & 0x0F0000) | 0x00E00000) >> 16);
        *out_ptr++ = static_cast<char>((((code_point << 2) & 0x003F00) | 0x008000) >> 8);
        *out_ptr++ = static_cast<char>((code_point & 0x3F) | 0x0080);
      } else {  // create four-byte UTF-8
        // maximum code-point value is 0x00110000
        // b000xxxxx:byyyyyyyy:bzzzzzzzz => b11110xxx:b10xxyyyy:b10yyyyzz:b10zzzzzz
        *out_ptr++ =
          static_cast<char>((((code_point << 6) & 0x07000000) | unsigned{0xF0000000}) >> 24);
        *out_ptr++ = static_cast<char>((((code_point << 4) & 0x003F0000) | 0x00800000) >> 16);
        *out_ptr++ = static_cast<char>((((code_point << 2) & 0x003F00) | 0x008000) >> 8);
        *out_ptr++ = static_cast<char>((code_point & 0x3F) | 0x0080);
      }
    }
    return 0;
  }
};

}  // namespace

// details API
std::unique_ptr<cudf::column> normalize_spaces(
  cudf::strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  cudf::size_type strings_count = strings.size();
  if (strings_count == 0) return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  // create device column
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // copy bitmask
  rmm::device_buffer null_mask = copy_bitmask(strings.parent(), stream, mr);

  // create offsets by calculating size of each string for output
  auto offsets_transformer_itr =
    thrust::make_transform_iterator(thrust::make_counting_iterator<int32_t>(0),
                                    normalize_spaces_fn{d_strings});  // this does size-only calc
  auto offsets_column = cudf::strings::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
  auto d_offsets = offsets_column->view().data<int32_t>();

  // build the chars column
  cudf::size_type bytes = thrust::device_pointer_cast(d_offsets)[strings_count];
  auto chars_column     = cudf::strings::detail::create_chars_child_column(
    strings_count, strings.null_count(), bytes, mr, stream);
  auto d_chars = chars_column->mutable_view().data<char>();

  // copy tokens to the chars buffer
  thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator<cudf::size_type>(0),
                     strings_count,
                     normalize_spaces_fn{d_strings, d_offsets, d_chars});
  chars_column->set_null_count(0);  // reset null count for child column
  //
  return cudf::make_strings_column(strings_count,
                                   std::move(offsets_column),
                                   std::move(chars_column),
                                   strings.null_count(),
                                   std::move(null_mask),
                                   stream,
                                   mr);
}

/**
 * @copydoc nvtext::normalize_characters
 */
std::unique_ptr<cudf::column> normalize_characters(cudf::strings_column_view const& strings,
                                                   bool do_lower_case,
                                                   cudaStream_t stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  auto const strings_count = strings.size();
  if (strings_count == 0 || strings.chars_size() == 0)
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  // create the normalizer and call it
  data_normalizer normalizer(strings_count, strings.chars_size(), stream, do_lower_case);
  auto result = [&strings, &normalizer, stream] {
    auto const offsets   = strings.offsets();
    auto const d_offsets = offsets.data<uint32_t>() + strings.offset();
    auto const offset    = cudf::detail::get_value<int32_t>(offsets, strings.offset(), stream);
    auto const d_chars   = strings.chars().data<char>() + offset;
    return normalizer.normalize(d_chars, d_offsets, strings.size(), stream);
  }();

  CUDF_EXPECTS(result.first.length <= std::numeric_limits<cudf::size_type>::max(),
               "output too large for strings column");

  // convert the result into a strings column
  // - the cp_chars are the new 4-byte code-point values for all the characters in the output
  // - the cp_offsets identify which code-points go with which strings
  uint32_t const* cp_chars  = result.first.gpu_ptr;
  int32_t const* cp_offsets = reinterpret_cast<int32_t const*>(result.second.gpu_ptr);
  auto strings_column       = cudf::column_device_view::create(strings.parent(), stream);

  // build the output offsets column: compute the output size of each string
  auto offsets_transformer_itr =
    thrust::make_transform_iterator(thrust::make_counting_iterator<int32_t>(0),
                                    codepoint_to_utf8_fn{*strings_column, cp_chars, cp_offsets});
  auto offsets_column = cudf::strings::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
  auto d_offsets = offsets_column->view().data<int32_t>();

  // create the output chars column
  cudf::size_type output_bytes =
    cudf::detail::get_value<int32_t>(offsets_column->view(), strings_count, stream);
  auto chars_column = cudf::strings::detail::create_chars_child_column(
    strings_count, strings.null_count(), output_bytes, mr, stream);
  auto d_chars = chars_column->mutable_view().data<char>();

  // build the chars output data: convert the 4-byte code-point values into UTF-8 chars
  thrust::for_each_n(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    strings_count,
    codepoint_to_utf8_fn{*strings_column, cp_chars, cp_offsets, d_offsets, d_chars});
  chars_column->set_null_count(0);  // reset null count for child column

  return cudf::make_strings_column(strings_count,
                                   std::move(offsets_column),
                                   std::move(chars_column),
                                   strings.null_count(),
                                   copy_bitmask(strings.parent(), stream, mr),
                                   stream,
                                   mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> normalize_spaces(cudf::strings_column_view const& strings,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_spaces(strings, mr);
}

/**
 * @copydoc nvtext::normalize_characters
 */
std::unique_ptr<cudf::column> normalize_characters(cudf::strings_column_view const& strings,
                                                   bool do_lower_case,
                                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_characters(strings, do_lower_case, 0, mr);
}

}  // namespace nvtext
