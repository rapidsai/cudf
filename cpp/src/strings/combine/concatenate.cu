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
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/logical.h>

#include <algorithm>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Concatenate strings functor
 *
 * This will concatenate the strings from each row of the given table
 * and apply the separator. The null-replacement string `d_narep` is
 * used in place of any string in a row that contains a null entry.
 */
struct concat_strings_fn {
  table_device_view const d_table;
  string_view const d_separator;
  string_scalar_device_view const d_narep;
  offset_type* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    bool const null_element =
      thrust::any_of(thrust::seq, d_table.begin(), d_table.end(), [idx](auto const& col) {
        return col.is_null(idx);
      });
    // handle a null row
    if (null_element && !d_narep.is_valid()) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }

    char* d_buffer  = d_chars ? d_chars + d_offsets[idx] : nullptr;
    size_type bytes = 0;
    for (auto itr = d_table.begin(); itr < d_table.end(); ++itr) {
      auto const d_column = *itr;
      auto const d_str =
        d_column.is_null(idx) ? d_narep.value() : d_column.element<string_view>(idx);
      if (d_buffer) d_buffer = detail::copy_string(d_buffer, d_str);
      bytes += d_str.size_bytes();
      // separator goes only in between elements
      if (itr + 1 < d_table.end()) {
        if (d_buffer) d_buffer = detail::copy_string(d_buffer, d_separator);
        bytes += d_separator.size_bytes();
      }
    }
    if (!d_chars) d_offsets[idx] = bytes;
  }
};

}  // namespace

std::unique_ptr<column> concatenate(table_view const& strings_columns,
                                    string_scalar const& separator,
                                    string_scalar const& narep,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  auto const num_columns = strings_columns.num_columns();
  CUDF_EXPECTS(num_columns > 0, "At least one column must be specified");
  // check all columns are of type string
  CUDF_EXPECTS(std::all_of(strings_columns.begin(),
                           strings_columns.end(),
                           [](auto c) { return c.type().id() == type_id::STRING; }),
               "All columns must be of type string");
  if (num_columns == 1)  // single strings column returns a copy
    return std::make_unique<column>(*(strings_columns.begin()), stream, mr);
  auto const strings_count = strings_columns.num_rows();
  if (strings_count == 0)  // empty begets empty
    return detail::make_empty_strings_column(stream, mr);

  CUDF_EXPECTS(separator.is_valid(), "Parameter separator must be a valid string_scalar");
  string_view d_separator(separator.data(), separator.size());
  auto d_narep = get_scalar_device_view(const_cast<string_scalar&>(narep));

  // Create device views from the strings columns.
  auto d_table = table_device_view::create(strings_columns, stream);
  concat_strings_fn fn{*d_table, d_separator, d_narep};
  auto children = make_strings_children(fn, strings_count, stream, mr);

  // create resulting null mask
  auto [null_mask, null_count] = cudf::detail::valid_if(
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    [d_table = *d_table, d_narep] __device__(size_type idx) {
      bool null_element = thrust::any_of(
        thrust::seq, d_table.begin(), d_table.end(), [idx](auto col) { return col.is_null(idx); });
      return (!null_element || d_narep.is_valid());
    },
    stream,
    mr);

  return make_strings_column(strings_count,
                             std::move(children.first),
                             std::move(children.second),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
}

namespace {

/**
 * @brief Concatenate strings functor using multiple separators.
 *
 * A unique separator is provided for each row along with a string to use
 * when a separator row is null `d_separator_narep`. The `d_narep` is
 * used in place of a null entry in the strings columns.
 */
struct multi_separator_concat_fn {
  table_device_view const d_table;
  column_device_view const d_separators;
  string_scalar_device_view const d_separator_narep;
  string_scalar_device_view const d_narep;
  offset_type* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    bool const all_nulls =
      thrust::all_of(thrust::seq, d_table.begin(), d_table.end(), [idx](auto const& col) {
        return col.is_null(idx);
      });

    if ((d_separators.is_null(idx) && !d_separator_narep.is_valid()) ||
        (all_nulls && !d_narep.is_valid())) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }

    // point to output location
    char* d_buffer    = d_chars ? d_chars + d_offsets[idx] : nullptr;
    offset_type bytes = 0;

    // there is at least one non-null column value
    auto const d_separator = d_separators.is_valid(idx) ? d_separators.element<string_view>(idx)
                                                        : d_separator_narep.value();
    auto const d_null_rep = d_narep.is_valid() ? d_narep.value() : string_view{};

    // write output entry for this row
    bool colval_written = false;  // state variable for writing separators
    for (auto const d_column : d_table) {
      // if the row is null and if there is no replacement, skip it
      if (d_column.is_null(idx) && !d_narep.is_valid()) continue;

      // separator in this row is written only after the first output
      if (colval_written) {
        if (d_buffer) d_buffer = detail::copy_string(d_buffer, d_separator);
        bytes += d_separator.size_bytes();
      }

      // write out column's row data (or narep if the row is null)
      string_view const d_str =
        d_column.is_null(idx) ? d_null_rep : d_column.element<string_view>(idx);
      if (d_buffer) d_buffer = detail::copy_string(d_buffer, d_str);
      bytes += d_str.size_bytes();

      // column's string or narep could by empty so we need this flag
      // to know we got this far even if no actual bytes were copied
      colval_written = true;  // use the separator before the next column
    }

    if (!d_chars) d_offsets[idx] = bytes;
  }
};
}  // namespace

std::unique_ptr<column> concatenate(table_view const& strings_columns,
                                    strings_column_view const& separators,
                                    string_scalar const& separator_narep,
                                    string_scalar const& col_narep,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  auto const num_columns = strings_columns.num_columns();
  CUDF_EXPECTS(num_columns > 0, "At least one column must be specified");
  // Check if all columns are of type string
  CUDF_EXPECTS(std::all_of(strings_columns.begin(),
                           strings_columns.end(),
                           [](auto c) { return c.type().id() == type_id::STRING; }),
               "All columns must be of type string");

  auto const strings_count = strings_columns.num_rows();
  CUDF_EXPECTS(strings_count == separators.size(),
               "Separators column should be the same size as the strings columns");
  if (strings_count == 0)  // Empty begets empty
    return detail::make_empty_strings_column(stream, mr);

  // Invalid output column strings - null rows
  string_view const invalid_str{nullptr, 0};
  auto const separator_rep = get_scalar_device_view(const_cast<string_scalar&>(separator_narep));
  auto const col_rep       = get_scalar_device_view(const_cast<string_scalar&>(col_narep));
  auto const separator_col_view_ptr = column_device_view::create(separators.parent(), stream);
  auto const separator_col_view     = *separator_col_view_ptr;

  // Create device views from the strings columns.
  auto d_table = table_device_view::create(strings_columns, stream);

  multi_separator_concat_fn mscf{*d_table, separator_col_view, separator_rep, col_rep};
  auto children = make_strings_children(mscf, strings_count, stream, mr);

  // Create resulting null mask
  auto [null_mask, null_count] = cudf::detail::valid_if(
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    [d_table = *d_table, separator_col_view, separator_rep, col_rep] __device__(size_type ridx) {
      if (!separator_col_view.is_valid(ridx) && !separator_rep.is_valid()) return false;
      bool all_nulls =
        thrust::all_of(thrust::seq, d_table.begin(), d_table.end(), [ridx](auto const& col) {
          return col.is_null(ridx);
        });
      return all_nulls ? col_rep.is_valid() : true;
    },
    stream,
    mr);

  return make_strings_column(strings_count,
                             std::move(children.first),
                             std::move(children.second),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail

// APIs

std::unique_ptr<column> concatenate(table_view const& strings_columns,
                                    string_scalar const& separator,
                                    string_scalar const& narep,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate(strings_columns, separator, narep, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> concatenate(table_view const& strings_columns,
                                    strings_column_view const& separators,
                                    string_scalar const& separator_narep,
                                    string_scalar const& col_narep,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate(
    strings_columns, separators, separator_narep, col_narep, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
