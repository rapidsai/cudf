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

struct codepoint_to_utf8_fn {
  cudf::column_device_view const d_strings;
  uint32_t const* cp_data;
  int32_t const* d_offsets{};
  char* d_chars{};

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return;
    auto const d_str  = d_strings.element<cudf::string_view>(idx);
    auto const offset = d_offsets[idx];
    auto str_cps      = cp_data + offset;
    char* out_ptr     = d_chars + offset;
    auto const count  = d_offsets[idx + 1] - offset;
    for (int32_t jdx = 0; jdx < count; ++jdx) {
      uint32_t code_point = *str_cps++;
      printf("%d:0x%04x\n", jdx, code_point);
      if (code_point < 0x0080)  // ASCII range
        *out_ptr++ = static_cast<char>(code_point);
      else if (code_point < 0x0800) {  // create two-byte UTF-8
        // b00001xxx:byyyyyyyy => b110xxxyy:b10yyyyyy
        *out_ptr++ = static_cast<char>((((code_point << 2) & 0x001F00) | 0x00C000) >> 8);
        *out_ptr++ = static_cast<char>((code_point & 0x3F) | 0x0080);
        printf("   %d:0x%02X:%02X\n", jdx, out_ptr[-2], out_ptr[-1]);
      } else if (code_point < 0x010000) {  // create three-byte UTF-8
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

std::unique_ptr<cudf::column> normalize_characters(cudf::strings_column_view const& strings,
                                                   bool do_lower_case,
                                                   cudaStream_t stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  auto const strings_count = strings.size();
  if (strings_count == 0) return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

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
  // convert result into strings column
  uint32_t const* cp_chars   = result.first.gpu_ptr;
  cudf::size_type chars_size = static_cast<cudf::size_type>(result.first.length);
  int32_t const* cp_offsets  = reinterpret_cast<int32_t const*>(result.second.gpu_ptr);

  auto offsets_column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                  strings_count + 1,
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
  auto d_offsets      = offsets_column->mutable_view().data<int32_t>();
  auto const execpol  = rmm::exec_policy(stream);
  thrust::copy(execpol->on(stream), cp_offsets, cp_offsets + strings_count + 1, d_offsets);

  auto chars_column = cudf::strings::detail::create_chars_child_column(
    strings_count, strings.null_count(), chars_size, mr, stream);
  auto d_chars = chars_column->mutable_view().data<char>();

  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);

  // copy tokens to the chars buffer
  thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator<cudf::size_type>(0),
                     strings_count,
                     codepoint_to_utf8_fn{*strings_column, cp_chars, d_offsets, d_chars});
  chars_column->set_null_count(0);  // reset null count for child column
  //
  return cudf::make_strings_column(strings_count,
                                   std::move(offsets_column),
                                   std::move(chars_column),
                                   strings.null_count(),
                                   copy_bitmask(strings.parent(), stream, mr),
                                   stream,
                                   mr);

  return nullptr;
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> normalize_spaces(cudf::strings_column_view const& strings,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_spaces(strings, mr);
}

std::unique_ptr<cudf::column> normalize_characters(cudf::strings_column_view const& strings,
                                                   bool do_lower_case,
                                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_characters(strings, do_lower_case, 0, mr);
}

}  // namespace nvtext
