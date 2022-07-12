/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Per string logic for case conversion functions.
 *
 */
struct upper_lower_fn {
  const column_device_view d_column;
  character_flags_table_type case_flag;  // flag to check with on each character
  const character_flags_table_type* d_flags;
  const character_cases_table_type* d_case_table;
  const special_case_mapping* d_special_case_mapping;
  int32_t* d_offsets{};
  char* d_chars{};

  __device__ special_case_mapping get_special_case_mapping(uint32_t code_point)
  {
    return d_special_case_mapping[get_special_case_hash_index(code_point)];
  }

  // compute-size / copy the bytes representing the special case mapping for this codepoint
  __device__ int32_t handle_special_case_bytes(uint32_t code_point,
                                               char* d_buffer,
                                               detail::character_flags_table_type flag)
  {
    special_case_mapping m = get_special_case_mapping(code_point);
    size_type bytes        = 0;

    auto const count  = IS_LOWER(flag) ? m.num_upper_chars : m.num_lower_chars;
    auto const* chars = IS_LOWER(flag) ? m.upper : m.lower;
    for (uint16_t idx = 0; idx < count; idx++) {
      bytes += d_buffer
                 ? detail::from_char_utf8(detail::codepoint_to_utf8(chars[idx]), d_buffer + bytes)
                 : detail::bytes_in_char_utf8(detail::codepoint_to_utf8(chars[idx]));
    }
    return bytes;
  }

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const d_str = d_column.template element<string_view>(idx);
    int32_t bytes    = 0;
    char* d_buffer   = d_chars ? d_chars + d_offsets[idx] : nullptr;
    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      uint32_t code_point = detail::utf8_to_codepoint(*itr);

      detail::character_flags_table_type flag = code_point <= 0x00FFFF ? d_flags[code_point] : 0;

      // we apply special mapping in two cases:
      // - uncased characters with the special mapping flag, always
      // - cased characters with the special mapping flag, when matching the input case_flag
      //
      if (IS_SPECIAL(flag) && ((flag & case_flag) || !IS_UPPER_OR_LOWER(flag))) {
        auto const new_bytes = handle_special_case_bytes(code_point, d_buffer, case_flag);
        bytes += new_bytes;
        if (d_buffer) d_buffer += new_bytes;
      } else {
        char_utf8 new_char =
          (flag & case_flag) ? detail::codepoint_to_utf8(d_case_table[code_point]) : *itr;
        if (!d_buffer)
          bytes += detail::bytes_in_char_utf8(new_char);
        else
          d_buffer += detail::from_char_utf8(new_char, d_buffer);
      }
    }
    if (!d_buffer) d_offsets[idx] = bytes;
  }
};

/**
 * @brief Utility method for converting upper and lower case characters
 * in a strings column.
 *
 * @param strings Strings to convert.
 * @param case_flag The character type to convert (upper, lower, or both)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column with characters converted.
 */
std::unique_ptr<column> convert_case(strings_column_view const& strings,
                                     character_flags_table_type case_flag,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  if (strings.is_empty()) return make_empty_column(type_id::STRING);

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;

  // build functor with lookup tables used for case conversion
  upper_lower_fn functor{d_column,
                         case_flag,
                         get_character_flags_table(),
                         get_character_cases_table(),
                         get_special_case_mapping_table()};

  // this utility calls the functor to build the offsets and chars columns
  auto children = cudf::strings::detail::make_strings_children(functor, strings.size(), stream, mr);

  return make_strings_column(strings.size(),
                             std::move(children.first),
                             std::move(children.second),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr));
}

}  // namespace

std::unique_ptr<column> to_lower(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  character_flags_table_type case_flag = IS_UPPER(0xFF);  // convert only upper case characters
  return convert_case(strings, case_flag, stream, mr);
}

//
std::unique_ptr<column> to_upper(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  character_flags_table_type case_flag = IS_LOWER(0xFF);  // convert only lower case characters
  return convert_case(strings, case_flag, stream, mr);
}

//
std::unique_ptr<column> swapcase(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  // convert only upper or lower case characters
  character_flags_table_type case_flag = IS_LOWER(0xFF) | IS_UPPER(0xFF);
  return convert_case(strings, case_flag, stream, mr);
}

}  // namespace detail

// APIs

std::unique_ptr<column> to_lower(strings_column_view const& strings,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_lower(strings, cudf::default_stream_value, mr);
}

std::unique_ptr<column> to_upper(strings_column_view const& strings,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_upper(strings, cudf::default_stream_value, mr);
}

std::unique_ptr<column> swapcase(strings_column_view const& strings,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::swapcase(strings, cudf::default_stream_value, mr);
}

}  // namespace strings
}  // namespace cudf
