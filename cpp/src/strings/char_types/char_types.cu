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
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
//
std::unique_ptr<column> all_characters_of_type(
  strings_column_view const& strings,
  string_character_types types,
  string_character_types verify_types,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto strings_count  = strings.size();
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;

  // create output column
  auto results      = make_numeric_column(data_type{type_id::BOOL8},
                                     strings_count,
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<bool>();
  // get the static character types table
  auto d_flags = detail::get_character_flags_table();
  // set the output values by checking the character types for each string
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(strings_count),
                    d_results,
                    [d_column, d_flags, types, verify_types, d_results] __device__(size_type idx) {
                      if (d_column.is_null(idx)) return false;
                      auto d_str            = d_column.element<string_view>(idx);
                      bool check            = !d_str.empty();  // require at least one character
                      size_type check_count = 0;
                      for (auto itr = d_str.begin(); check && (itr != d_str.end()); ++itr) {
                        auto code_point = detail::utf8_to_codepoint(*itr);
                        // lookup flags in table by code-point
                        auto flag = code_point <= 0x00'FFFF ? d_flags[code_point] : 0;
                        if ((verify_types & flag) ||                   // should flag be verified
                            (flag == 0 && verify_types == ALL_TYPES))  // special edge case
                        {
                          check = (types & flag) > 0;
                          ++check_count;
                        }
                      }
                      return check && (check_count > 0);
                    });
  //
  results->set_null_count(strings.null_count());
  return results;
}

namespace {

/**
 * @brief Removes individual characters from a strings column based on character type.
 *
 * Types to remove are specified by `types_to_remove` OR
 * types to not remove are specified by `types_to_keep`.
 *
 * This is called twice. The first pass calculates the size of each output string.
 * The final pass copies the results to the output strings column memory.
 */
struct filter_chars_fn {
  column_device_view const d_column;
  character_flags_table_type const* d_flags;
  string_character_types const types_to_remove;
  string_character_types const types_to_keep;
  string_view const d_replacement;  ///< optional replacement for removed characters
  int32_t* d_offsets{};             ///< size of the output string stored here during first pass
  char* d_chars{};                  ///< this is null only during the first pass

  /**
   * @brief Returns true if the given character should be replaced.
   */
  __device__ bool replace_char(char_utf8 ch)
  {
    auto const code_point = detail::utf8_to_codepoint(ch);
    auto const flag       = code_point <= 0x00'FFFF ? d_flags[code_point] : 0;
    if (flag == 0)  // all types pass unless specifically identified
      return (types_to_remove == ALL_TYPES);
    if (types_to_keep == ALL_TYPES)  // filter case
      return (types_to_remove & flag) != 0;
    return (types_to_keep & flag) == 0;  // keep case
  }

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const d_str  = d_column.element<string_view>(idx);
    auto const in_ptr = d_str.data();
    auto out_ptr      = d_chars ? d_chars + d_offsets[idx] : nullptr;
    auto nbytes       = d_str.size_bytes();

    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      auto const char_size = bytes_in_char_utf8(*itr);
      string_view const d_newchar =
        replace_char(*itr) ? d_replacement : string_view(in_ptr + itr.byte_offset(), char_size);
      nbytes += d_newchar.size_bytes() - char_size;
      if (out_ptr) out_ptr = cudf::strings::detail::copy_string(out_ptr, d_newchar);
    }
    if (!out_ptr) d_offsets[idx] = nbytes;
  }
};

}  // namespace

std::unique_ptr<column> filter_characters_of_type(strings_column_view const& strings,
                                                  string_character_types types_to_remove,
                                                  string_scalar const& replacement,
                                                  string_character_types types_to_keep,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(replacement.is_valid(stream), "Parameter replacement must be valid");
  if (types_to_remove == ALL_TYPES)
    CUDF_EXPECTS(types_to_keep != ALL_TYPES,
                 "Parameters types_to_remove and types_to_keep must not be both ALL_TYPES");
  else
    CUDF_EXPECTS(types_to_keep == ALL_TYPES,
                 "One of parameter types_to_remove and types_to_keep must be set to ALL_TYPES");

  auto const strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(cudf::data_type{cudf::type_id::STRING});

  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  cudf::string_view d_replacement(replacement.data(), replacement.size());
  filter_chars_fn filterer{*strings_column,
                           detail::get_character_flags_table(),
                           types_to_remove,
                           types_to_keep,
                           d_replacement};

  // copy null mask from input column
  rmm::device_buffer null_mask = cudf::detail::copy_bitmask(strings.parent(), stream, mr);

  // this utility calls filterer to build the offsets and chars columns
  auto children = cudf::strings::detail::make_strings_children(filterer, strings_count, stream, mr);

  // return new strings column
  return make_strings_column(strings_count,
                             std::move(children.first),
                             std::move(children.second),
                             strings.null_count(),
                             std::move(null_mask));
}

}  // namespace detail

// external API

std::unique_ptr<column> all_characters_of_type(strings_column_view const& strings,
                                               string_character_types types,
                                               string_character_types verify_types,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::all_characters_of_type(
    strings, types, verify_types, cudf::default_stream_value, mr);
}

std::unique_ptr<column> filter_characters_of_type(strings_column_view const& strings,
                                                  string_character_types types_to_remove,
                                                  string_scalar const& replacement,
                                                  string_character_types types_to_keep,
                                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_characters_of_type(
    strings, types_to_remove, replacement, types_to_keep, cudf::default_stream_value, mr);
}

}  // namespace strings
}  // namespace cudf
