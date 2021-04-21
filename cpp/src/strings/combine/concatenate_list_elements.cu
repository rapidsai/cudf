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

#include <strings/utilities.cuh>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf {
namespace strings {
namespace detail {

namespace {
/**
 * @brief Concatenate strings functor.
 *
 * This will concatenate the strings within each list element of the given string lists column
 * and apply the separator. The null-replacement string scalar `string_narep_dv` (if valid) is
 * used in place of any null string.
 */
struct concat_strings_fn {
  column_device_view const lists_dv;
  offset_type const* const list_offsets;
  column_device_view const strings_dv;
  string_scalar_device_view const string_narep_dv;
  string_view const separator;

  offset_type* d_offsets{nullptr};

  // If d_chars == nullptr: only compute sizes of the output strings.
  // If d_chars != nullptr: only concatenate strings.
  char* d_chars{nullptr};

  // This array is initialized to contain all `1` values, thus we only need to set `0` for the rows
  // corresponding to null string elements.
  int8_t* d_validities{nullptr};

  __device__ void operator()(size_type const idx)
  {
    // If this is the second pass, and the row `idx` is known to be a null string
    if (d_chars and not d_validities[idx]) { return; }

    if (not d_chars and lists_dv.is_null(idx)) {
      d_offsets[idx]    = 0;
      d_validities[idx] = 0;  // null output string
      return;
    }

    auto const separator_size = separator.size_bytes();
    auto size_bytes           = size_type{0};
    bool written              = false;
    char* output_ptr          = d_chars ? d_chars + d_offsets[idx] : nullptr;

    for (size_type str_idx = list_offsets[idx], idx_end = list_offsets[idx + 1]; str_idx < idx_end;
         ++str_idx) {
      if (not d_chars and strings_dv.is_null(str_idx) and not string_narep_dv.is_valid()) {
        d_offsets[idx]    = 0;
        d_validities[idx] = 0;  // null output string
        return;  // early termination: the entire list of strings will result in a null string
      }
      auto const d_str = strings_dv.is_null(str_idx) ? string_narep_dv.value()
                                                     : strings_dv.element<string_view>(str_idx);
      size_bytes += separator_size + d_str.size_bytes();
      if (output_ptr) {
        // Separator is inserted only in between strings
        if (written) { output_ptr = detail::copy_string(output_ptr, separator); }
        output_ptr = detail::copy_string(output_ptr, d_str);
        written    = true;
      }
    }

    // Separator is inserted only in between strings
    if (not d_chars) { d_offsets[idx] = static_cast<size_type>(size_bytes - separator_size); }
  }
};

}  // namespace

std::unique_ptr<column> concatenate_list_elements(lists_column_view const& lists_strings_column,
                                                  string_scalar const& separator,
                                                  string_scalar const& narep,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(lists_strings_column.child().type().id() == type_id::STRING,
               "The input column must be a column of lists of strings");
  CUDF_EXPECTS(separator.is_valid(), "Parameter separator must be a valid string_scalar");

  auto const num_rows = lists_strings_column.size();
  if (num_rows == 0) { return detail::make_empty_strings_column(stream, mr); }

  // Accessing the child strings column of the lists column must be done by calling `child()` on the
  // lists column, not `get_sliced_child()`. This is because calling to `offsets_begin()` on the
  // lists column returns a pointer to the offsets of the original lists column, which may not start
  // from `0`.
  auto const strings_col     = strings_column_view(lists_strings_column.child());
  auto const lists_dv_ptr    = column_device_view::create(lists_strings_column.parent(), stream);
  auto const strings_dv_ptr  = column_device_view::create(strings_col.parent(), stream);
  auto const string_narep_dv = get_scalar_device_view(const_cast<string_scalar&>(narep));
  auto const fn              = concat_strings_fn{*lists_dv_ptr,
                                    lists_strings_column.offsets_begin(),
                                    *strings_dv_ptr,
                                    string_narep_dv,
                                    string_view{separator.data(), separator.size()}};
  auto [offsets_column, chars_column, null_mask, null_count] =
    make_strings_children_with_null_mask(fn, num_rows, stream, mr);

  return make_strings_column(num_rows,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
}

namespace {
/**
 * @brief Concatenate strings functor using multiple separators.
 *
 * This will concatenate the strings within each list element of the given lists column
 * and apply the separators. A unique separator is provided for each list row. The null-replacement
 * string scalar `sep_narep_dv` (if valid) can be used in place of any null separator. The
 * null-replacement string scalar `string_narep_dv` (if valid) is used in place of any string in a
 * list that contains a null entry.
 */
struct concat_strings_multi_separators_fn {
  column_device_view const lists_dv;
  offset_type const* const list_offsets;
  column_device_view const strings_dv;
  column_device_view const separators_dv;
  string_scalar_device_view const string_narep_dv;
  string_scalar_device_view const sep_narep_dv;

  offset_type* d_offsets{nullptr};

  // If d_chars == nullptr: only compute sizes of the output strings.
  // If d_chars != nullptr: only concatenate strings.
  char* d_chars{nullptr};

  // This array is initialized to contain all `1` values, thus we only need to set `0` for the rows
  // corresponding to null string elements.
  int8_t* d_validities{nullptr};

  __device__ void operator()(size_type const idx)
  {
    // If the row `idx` is known to be a null string
    if (d_chars and not d_validities[idx]) { return; }

    if (not d_chars and
        (lists_dv.is_null(idx) or (separators_dv.is_null(idx) and not sep_narep_dv.is_valid()))) {
      d_offsets[idx]    = 0;
      d_validities[idx] = 0;  // null output string
      return;
    }

    auto const separator =
      separators_dv.is_valid(idx) ? separators_dv.element<string_view>(idx) : sep_narep_dv.value();
    auto const separator_size = separator.size_bytes();
    auto size_bytes           = size_type{0};
    bool written              = false;
    char* output_ptr          = d_chars ? d_chars + d_offsets[idx] : nullptr;

    for (size_type str_idx = list_offsets[idx], idx_end = list_offsets[idx + 1]; str_idx < idx_end;
         ++str_idx) {
      if (not d_chars and strings_dv.is_null(str_idx) and not string_narep_dv.is_valid()) {
        d_offsets[idx]    = 0;
        d_validities[idx] = 0;  // null output string
        return;  // early termination: the entire list of strings will result in a null string
      }
      auto const d_str = strings_dv.is_null(str_idx) ? string_narep_dv.value()
                                                     : strings_dv.element<string_view>(str_idx);
      size_bytes += separator_size + d_str.size_bytes();
      if (output_ptr) {
        // Separator is inserted only in between strings
        if (written) { output_ptr = detail::copy_string(output_ptr, separator); }
        output_ptr = detail::copy_string(output_ptr, d_str);
        written    = true;
      }
    }

    // Separator is inserted only in between strings
    if (not d_chars) { d_offsets[idx] = static_cast<size_type>(size_bytes - separator_size); }
  }
};

}  // namespace

std::unique_ptr<column> concatenate_list_elements(lists_column_view const& lists_strings_column,
                                                  strings_column_view const& separators,
                                                  string_scalar const& separator_narep,
                                                  string_scalar const& string_narep,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(lists_strings_column.child().type().id() == type_id::STRING,
               "The input column must be a column of lists of strings");
  CUDF_EXPECTS(lists_strings_column.size() == separators.size(),
               "Separators column should be the same size as the lists columns");

  auto const num_rows = lists_strings_column.size();
  if (num_rows == 0) { return detail::make_empty_strings_column(stream, mr); }

  // Accessing the child strings column of the lists column must be done by calling `child()` on the
  // lists column, not `get_sliced_child()`. This is because calling to `offsets_begin()` on the
  // lists column returns a pointer to the offsets of the original lists column, which may not start
  // from `0`.
  auto const strings_col     = strings_column_view(lists_strings_column.child());
  auto const lists_dv_ptr    = column_device_view::create(lists_strings_column.parent(), stream);
  auto const strings_dv_ptr  = column_device_view::create(strings_col.parent(), stream);
  auto const string_narep_dv = get_scalar_device_view(const_cast<string_scalar&>(string_narep));
  auto const sep_dv_ptr      = column_device_view::create(separators.parent(), stream);
  auto const sep_narep_dv    = get_scalar_device_view(const_cast<string_scalar&>(separator_narep));
  auto const fn              = concat_strings_multi_separators_fn{*lists_dv_ptr,
                                                     lists_strings_column.offsets_begin(),
                                                     *strings_dv_ptr,
                                                     *sep_dv_ptr,
                                                     string_narep_dv,
                                                     sep_narep_dv};
  auto [offsets_column, chars_column, null_mask, null_count] =
    make_strings_children_with_null_mask(fn, num_rows, stream, mr);

  return make_strings_column(num_rows,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail

std::unique_ptr<column> concatenate_list_elements(lists_column_view const& lists_strings_column,
                                                  string_scalar const& separator,
                                                  string_scalar const& narep,
                                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate_list_elements(
    lists_strings_column, separator, narep, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> concatenate_list_elements(lists_column_view const& lists_strings_column,
                                                  strings_column_view const& separators,
                                                  string_scalar const& separator_narep,
                                                  string_scalar const& string_narep,
                                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate_list_elements(
    lists_strings_column, separators, separator_narep, string_narep, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
