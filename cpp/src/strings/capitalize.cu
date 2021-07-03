/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <strings/char_types/is_flags.h>
#include <strings/utf8.cuh>
#include <strings/utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/capitalize.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Base class for capitalize and title functors.
 *
 * Utility functions here manage access to the character case and flags tables.
 * Any derived class must supply a `capitalize_next` member function.
 *
 * @tparam Derived class uses the CRTP pattern to reuse code logic.
 */
template <typename Derived>
struct base_fn {
  character_flags_table_type const* d_flags;
  character_cases_table_type const* d_case_table;
  column_device_view const d_column;
  offset_type* d_offsets{};
  char* d_chars{};

  base_fn(column_device_view const& d_column)
    : d_flags(get_character_flags_table()),
      d_case_table(get_character_cases_table()),
      d_column(d_column)
  {
  }

  using char_info = thrust::pair<uint32_t, detail::character_flags_table_type>;

  __device__ char_info get_char_info(char_utf8 chr) const
  {
    auto const code_point = detail::utf8_to_codepoint(chr);
    auto const flag = code_point <= 0x00FFFF ? d_flags[code_point] : character_flags_table_type{0};
    return char_info{code_point, flag};
  }

  __device__ char_utf8 convert_char(char_info const& info) const
  {
    return codepoint_to_utf8(d_case_table[info.first]);
  }

  /**
   * @brief Operator called for each row in `d_column`.
   *
   * This logic is shared by capitalize() and title() functions.
   * The derived class must supply a `capitalize_next` member function.
   */
  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
    }

    Derived& derived  = static_cast<Derived&>(*this);
    auto const d_str  = d_column.element<string_view>(idx);
    offset_type bytes = 0;
    auto d_buffer     = d_chars ? d_chars + d_offsets[idx] : nullptr;
    bool capitalize   = true;
    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      auto const info        = get_char_info(*itr);
      auto const flag        = info.second;
      auto const change_case = capitalize ? IS_LOWER(flag) : IS_UPPER(flag);
      auto const new_char    = change_case ? convert_char(info) : *itr;
      // capitalize the next char if this one is a delimiter
      capitalize = derived.capitalize_next(*itr, flag);

      if (d_buffer)
        d_buffer += detail::from_char_utf8(new_char, d_buffer);
      else
        bytes += detail::bytes_in_char_utf8(new_char);
    }
    if (!d_chars) d_offsets[idx] = bytes;
  }
};

/**
 * @brief Capitalize functor.
 *
 * This capitalizes the first character of the string and lower-cases
 * the remaining characters.
 * If a delimiter is specified, capitalization continues within the string
 * on the first eligible character after any delimiter.
 */
struct capitalize_fn : base_fn<capitalize_fn> {
  string_view const d_delimiters;

  capitalize_fn(column_device_view const& d_column, string_view const& d_delimiters)
    : base_fn(d_column), d_delimiters(d_delimiters)
  {
  }

  __device__ bool capitalize_next(char_utf8 const chr, character_flags_table_type const)
  {
    return !d_delimiters.empty() && (d_delimiters.find(chr) >= 0);
  }
};

/**
 * @brief Title functor.
 *
 * This capitalizes the first letter of each word.
 * The beginning of a word is identified as the first sequence_type
 * character after a non-sequence_type character.
 * Also, lower-case all other alphabetic characters.
 */
struct title_fn : base_fn<title_fn> {
  string_character_types sequence_type;

  title_fn(column_device_view const& d_column, string_character_types sequence_type)
    : base_fn(d_column), sequence_type(sequence_type)
  {
  }

  __device__ bool capitalize_next(char_utf8 const, character_flags_table_type const flag)
  {
    return (flag & sequence_type) == 0;
  };
};

/**
 * @brief Common utility function for title() and capitalize().
 *
 * @tparam CapitalFn The specific functor.
 * @param cfn The functor instance.
 * @param input The input strings column.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used for allocating the new device_buffer
 */
template <typename CapitalFn>
std::unique_ptr<column> capitalizer(CapitalFn cfn,
                                    strings_column_view const& input,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  auto children = cudf::strings::detail::make_strings_children(cfn, input.size(), stream, mr);

  return make_strings_column(input.size(),
                             std::move(children.first),
                             std::move(children.second),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr),
                             stream,
                             mr);
}

}  // namespace

std::unique_ptr<column> capitalize(strings_column_view const& input,
                                   string_scalar const& delimiters,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(delimiters.is_valid(stream), "Delimiter must be a valid string");
  if (input.is_empty()) return make_empty_column(data_type{type_id::STRING});
  auto const d_column     = column_device_view::create(input.parent(), stream);
  auto const d_delimiters = delimiters.value(stream);
  return capitalizer(capitalize_fn{*d_column, d_delimiters}, input, stream, mr);
}

std::unique_ptr<column> title(strings_column_view const& input,
                              string_character_types sequence_type,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) return make_empty_column(data_type{type_id::STRING});
  auto d_column = column_device_view::create(input.parent(), stream);
  return capitalizer(title_fn{*d_column, sequence_type}, input, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> capitalize(strings_column_view const& input,
                                   string_scalar const& delimiter,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::capitalize(input, delimiter, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> title(strings_column_view const& input,
                              string_character_types sequence_type,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::title(input, sequence_type, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
