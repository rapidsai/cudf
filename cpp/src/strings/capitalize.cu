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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/capitalize.hpp>
#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

using char_info = thrust::pair<uint32_t, detail::character_flags_table_type>;

/**
 * @brief Returns the given character's info flags.
 */
__device__ char_info get_char_info(character_flags_table_type const* d_flags, char_utf8 chr)
{
  auto const code_point = detail::utf8_to_codepoint(chr);
  auto const flag = code_point <= 0x00'FFFF ? d_flags[code_point] : character_flags_table_type{0};
  return char_info{code_point, flag};
}

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
  special_case_mapping const* d_special_case_mapping;
  column_device_view const d_column;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  base_fn(column_device_view const& d_column)
    : d_flags(get_character_flags_table()),
      d_case_table(get_character_cases_table()),
      d_special_case_mapping(get_special_case_mapping_table()),
      d_column(d_column)
  {
  }

  __device__ int32_t convert_char(char_info const& info, char* d_buffer) const
  {
    auto const code_point = info.first;
    auto const flag       = info.second;

    if (!IS_SPECIAL(flag)) {
      auto const new_char = codepoint_to_utf8(d_case_table[code_point]);
      return d_buffer ? detail::from_char_utf8(new_char, d_buffer)
                      : detail::bytes_in_char_utf8(new_char);
    }

    special_case_mapping m = d_special_case_mapping[get_special_case_hash_index(code_point)];

    auto const count  = IS_LOWER(flag) ? m.num_upper_chars : m.num_lower_chars;
    auto const* chars = IS_LOWER(flag) ? m.upper : m.lower;
    size_type bytes   = 0;
    for (uint16_t idx = 0; idx < count; idx++) {
      bytes += d_buffer
                 ? detail::from_char_utf8(detail::codepoint_to_utf8(chars[idx]), d_buffer + bytes)
                 : detail::bytes_in_char_utf8(detail::codepoint_to_utf8(chars[idx]));
    }
    return bytes;
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
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }

    auto& derived    = static_cast<Derived&>(*this);
    auto const d_str = d_column.element<string_view>(idx);
    size_type bytes  = 0;
    auto d_buffer    = d_chars ? d_chars + d_offsets[idx] : nullptr;
    bool capitalize  = true;
    for (auto const chr : d_str) {
      auto const info        = get_char_info(d_flags, chr);
      auto const flag        = info.second;
      auto const change_case = capitalize ? IS_LOWER(flag) : IS_UPPER(flag);

      if (change_case) {
        auto const char_bytes = convert_char(info, d_buffer);
        bytes += char_bytes;
        d_buffer += d_buffer ? char_bytes : 0;
      } else {
        if (d_buffer) {
          d_buffer += detail::from_char_utf8(chr, d_buffer);
        } else {
          bytes += detail::bytes_in_char_utf8(chr);
        }
      }

      // capitalize the next char if this one is a delimiter
      capitalize = derived.capitalize_next(chr, flag);
    }
    if (!d_chars) { d_sizes[idx] = bytes; }
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
    return !d_delimiters.empty() && (d_delimiters.find(chr) != string_view::npos);
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
 * @brief Functor for determining title format for each string in a column.
 *
 * The first letter of each word should be upper-case (IS_UPPER).
 * All other characters should be lower-case (IS_LOWER).
 * Non-upper/lower-case (IS_UPPER_OR_LOWER) characters delimit words.
 */
struct is_title_fn {
  character_flags_table_type const* d_flags;
  column_device_view const d_column;

  __device__ bool operator()(size_type idx)
  {
    if (d_column.is_null(idx)) { return false; }
    auto const d_str = d_column.element<string_view>(idx);

    bool at_least_one_valid    = false;  // requires one or more cased characters
    bool should_be_capitalized = true;   // current character should be upper-case
    for (auto const chr : d_str) {
      auto const flag = get_char_info(d_flags, chr).second;
      if (IS_UPPER_OR_LOWER(flag)) {
        if (should_be_capitalized == !IS_UPPER(flag)) return false;
        at_least_one_valid = true;
      }
      should_be_capitalized = !IS_UPPER_OR_LOWER(flag);
    }
    return at_least_one_valid;
  }
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
                                    rmm::device_async_resource_ref mr)
{
  auto [offsets_column, chars] = make_strings_children(cfn, input.size(), stream, mr);

  return make_strings_column(input.size(),
                             std::move(offsets_column),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace

std::unique_ptr<column> capitalize(strings_column_view const& input,
                                   string_scalar const& delimiters,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiters.is_valid(stream), "Delimiter must be a valid string");
  if (input.is_empty()) return make_empty_column(type_id::STRING);
  auto const d_column     = column_device_view::create(input.parent(), stream);
  auto const d_delimiters = delimiters.value(stream);
  return capitalizer(capitalize_fn{*d_column, d_delimiters}, input, stream, mr);
}

std::unique_ptr<column> title(strings_column_view const& input,
                              string_character_types sequence_type,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return make_empty_column(type_id::STRING);
  auto d_column = column_device_view::create(input.parent(), stream);
  return capitalizer(title_fn{*d_column, sequence_type}, input, stream, mr);
}

std::unique_ptr<column> is_title(strings_column_view const& input,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return make_empty_column(type_id::BOOL8);
  auto results  = make_numeric_column(data_type{type_id::BOOL8},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  auto d_column = column_device_view::create(input.parent(), stream);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(input.size()),
                    results->mutable_view().data<bool>(),
                    is_title_fn{get_character_flags_table(), *d_column});
  results->set_null_count(input.null_count());
  return results;
}

}  // namespace detail

std::unique_ptr<column> capitalize(strings_column_view const& input,
                                   string_scalar const& delimiter,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::capitalize(input, delimiter, stream, mr);
}

std::unique_ptr<column> title(strings_column_view const& input,
                              string_character_types sequence_type,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::title(input, sequence_type, stream, mr);
}

std::unique_ptr<column> is_title(strings_column_view const& input,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_title(input, stream, mr);
}

}  // namespace strings
}  // namespace cudf
