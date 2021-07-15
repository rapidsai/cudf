/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf {
namespace strings {
namespace detail {

namespace {
/**
 * @brief Compute string sizes, string validities, and concatenate strings functor.
 *
 * This functor is executed twice. In the first pass, the sizes and validities of the output strings
 * will be computed. In the second pass, this will concatenate the strings within each list element
 * of the given lists column and apply the separator. The null-replacement string scalar
 * `string_narep_dv` (if valid) is used in place of any null string.
 *
 * @tparam Functor The functor which can check for validity of the input list at a given list index
 * as well as access to the separator corresponding to the list index.
 */
template <class Functor>
struct compute_size_and_concatenate_fn {
  Functor const func;
  column_device_view const lists_dv;
  offset_type const* const list_offsets;
  column_device_view const strings_dv;
  string_scalar_device_view const string_narep_dv;
  separator_on_nulls const separate_nulls;
  output_if_empty_list const empty_list_policy;

  offset_type* d_offsets{nullptr};

  // If d_chars == nullptr: only compute sizes and validities of the output strings.
  // If d_chars != nullptr: only concatenate strings.
  char* d_chars{nullptr};

  // We need to set `1` or `0` for the validities of the output strings.
  int8_t* d_validities{nullptr};

  __device__ bool output_is_null(size_type const idx,
                                 size_type const start_idx,
                                 size_type const end_idx) const noexcept
  {
    if (func.is_null_list(lists_dv, idx)) { return true; }
    return empty_list_policy == output_if_empty_list::NULL_ELEMENT && start_idx == end_idx;
  }

  __device__ void operator()(size_type const idx) const noexcept
  {
    // If this is the second pass, and the row `idx` is known to be a null string
    if (d_chars && !d_validities[idx]) { return; }

    // Indices of the strings within the list row
    auto const start_idx = list_offsets[idx];
    auto const end_idx   = list_offsets[idx + 1];

    if (!d_chars && output_is_null(idx, start_idx, end_idx)) {
      d_offsets[idx]    = 0;
      d_validities[idx] = false;
      return;
    }

    auto const separator   = func.separator(idx);
    auto size_bytes        = size_type{0};
    char* output_ptr       = d_chars ? d_chars + d_offsets[idx] : nullptr;
    bool has_valid_element = false;
    bool write_separator   = false;

    for (size_type str_idx = start_idx; str_idx < end_idx; ++str_idx) {
      bool null_element = strings_dv.is_null(str_idx);
      has_valid_element = has_valid_element || !null_element;

      if (!d_chars && (null_element && !string_narep_dv.is_valid())) {
        d_offsets[idx]    = 0;
        d_validities[idx] = false;
        return;  // early termination: the entire list of strings will result in a null string
      }

      if (write_separator && (separate_nulls == separator_on_nulls::YES || !null_element)) {
        if (output_ptr) output_ptr = detail::copy_string(output_ptr, separator);
        size_bytes += separator.size_bytes();
        write_separator = false;
      }

      auto const d_str =
        null_element ? string_narep_dv.value() : strings_dv.element<string_view>(str_idx);
      if (output_ptr) output_ptr = detail::copy_string(output_ptr, d_str);
      size_bytes += d_str.size_bytes();

      write_separator =
        write_separator || (separate_nulls == separator_on_nulls::YES) || !null_element;
    }

    // If there are all null elements, the output should be the same as having an empty list input:
    // a null or an empty string
    if (!d_chars) {
      d_offsets[idx] = has_valid_element ? size_bytes : 0;
      d_validities[idx] =
        has_valid_element || empty_list_policy == output_if_empty_list::EMPTY_STRING;
    }
  }
};

/**
 * @brief Functor accompanying with `compute_size_and_concatenate_fn` for computing output string
 * sizes, output string validities, and concatenating strings within list elements; used when the
 * separator is a string scalar.
 */
struct scalar_separator_fn {
  string_scalar_device_view const d_separator;

  __device__ bool is_null_list(column_device_view const& lists_dv,
                               size_type const idx) const noexcept
  {
    return lists_dv.is_null(idx);
  }

  __device__ string_view separator(size_type const) const noexcept { return d_separator.value(); }
};

}  // namespace

std::unique_ptr<column> join_list_elements(lists_column_view const& lists_strings_column,
                                           string_scalar const& separator,
                                           string_scalar const& narep,
                                           separator_on_nulls separate_nulls,
                                           output_if_empty_list empty_list_policy,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(lists_strings_column.child().type().id() == type_id::STRING,
               "The input column must be a column of lists of strings");
  CUDF_EXPECTS(separator.is_valid(), "Parameter separator must be a valid string_scalar");

  auto const num_rows = lists_strings_column.size();
  if (num_rows == 0) { return make_empty_column(data_type{type_id::STRING}); }

  // Accessing the child strings column of the lists column must be done by calling `child()` on the
  // lists column, not `get_sliced_child()`. This is because calling to `offsets_begin()` on the
  // lists column returns a pointer to the offsets of the original lists column, which may not start
  // from `0`.
  auto const strings_col     = strings_column_view(lists_strings_column.child());
  auto const lists_dv_ptr    = column_device_view::create(lists_strings_column.parent(), stream);
  auto const strings_dv_ptr  = column_device_view::create(strings_col.parent(), stream);
  auto const sep_dv          = get_scalar_device_view(const_cast<string_scalar&>(separator));
  auto const string_narep_dv = get_scalar_device_view(const_cast<string_scalar&>(narep));

  auto const func = scalar_separator_fn{sep_dv};
  auto const comp_fn =
    compute_size_and_concatenate_fn<decltype(func)>{func,
                                                    *lists_dv_ptr,
                                                    lists_strings_column.offsets_begin(),
                                                    *strings_dv_ptr,
                                                    string_narep_dv,
                                                    separate_nulls,
                                                    empty_list_policy};
  auto [offsets_column, chars_column, null_mask, null_count] =
    make_strings_children_with_null_mask(comp_fn, num_rows, num_rows, stream, mr);

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
 * @brief Functor accompanying with `compute_size_and_concatenate_fn` for computing output string
 * sizes, output string validities, and concatenating strings within list elements; used when the
 * separators are given as a strings column.
 */
struct column_separators_fn {
  column_device_view const separators_dv;
  string_scalar_device_view const sep_narep_dv;

  __device__ bool is_null_list(column_device_view const& lists_dv,
                               size_type const idx) const noexcept
  {
    return lists_dv.is_null(idx) || (separators_dv.is_null(idx) && !sep_narep_dv.is_valid());
  }

  __device__ string_view separator(size_type const idx) const noexcept
  {
    return separators_dv.is_valid(idx) ? separators_dv.element<string_view>(idx)
                                       : sep_narep_dv.value();
  }
};

}  // namespace

std::unique_ptr<column> join_list_elements(lists_column_view const& lists_strings_column,
                                           strings_column_view const& separators,
                                           string_scalar const& separator_narep,
                                           string_scalar const& string_narep,
                                           separator_on_nulls separate_nulls,
                                           output_if_empty_list empty_list_policy,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(lists_strings_column.child().type().id() == type_id::STRING,
               "The input column must be a column of lists of strings");
  CUDF_EXPECTS(lists_strings_column.size() == separators.size(),
               "Separators column should be the same size as the lists columns");

  auto const num_rows = lists_strings_column.size();
  if (num_rows == 0) { return make_empty_column(data_type{type_id::STRING}); }

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

  auto const func = column_separators_fn{*sep_dv_ptr, sep_narep_dv};
  auto const comp_fn =
    compute_size_and_concatenate_fn<decltype(func)>{func,
                                                    *lists_dv_ptr,
                                                    lists_strings_column.offsets_begin(),
                                                    *strings_dv_ptr,
                                                    string_narep_dv,
                                                    separate_nulls,
                                                    empty_list_policy};
  auto [offsets_column, chars_column, null_mask, null_count] =
    make_strings_children_with_null_mask(comp_fn, num_rows, num_rows, stream, mr);

  return make_strings_column(num_rows,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail

std::unique_ptr<column> join_list_elements(lists_column_view const& lists_strings_column,
                                           string_scalar const& separator,
                                           string_scalar const& narep,
                                           separator_on_nulls separate_nulls,
                                           output_if_empty_list empty_list_policy,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::join_list_elements(lists_strings_column,
                                    separator,
                                    narep,
                                    separate_nulls,
                                    empty_list_policy,
                                    rmm::cuda_stream_default,
                                    mr);
}

std::unique_ptr<column> join_list_elements(lists_column_view const& lists_strings_column,
                                           strings_column_view const& separators,
                                           string_scalar const& separator_narep,
                                           string_scalar const& string_narep,
                                           separator_on_nulls separate_nulls,
                                           output_if_empty_list empty_list_policy,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::join_list_elements(lists_strings_column,
                                    separators,
                                    separator_narep,
                                    string_narep,
                                    separate_nulls,
                                    empty_list_policy,
                                    rmm::cuda_stream_default,
                                    mr);
}

}  // namespace strings
}  // namespace cudf
