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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/detail/utilities.cuh>
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

struct concat_strings_base {
  table_device_view const d_table;
  string_scalar_device_view const d_narep;
  separator_on_nulls separate_nulls;
  offset_type* d_offsets{};
  char* d_chars{};

  /**
   * @brief Concatenate each table row to a single output string.
   *
   * This will concatenate the strings from each row of the given table
   * and apply the separator. The null-replacement string `d_narep` is
   * used in place of any string in a row that contains a null entry.
   *
   * @param idx The current row to process
   * @param d_separator String to place in between each column's row
   */
  __device__ void process_row(size_type idx, string_view const d_separator)
  {
    if (!d_narep.is_valid() &&
        thrust::any_of(thrust::seq, d_table.begin(), d_table.end(), [idx](auto const& col) {
          return col.is_null(idx);
        })) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }

    char* d_buffer       = d_chars ? d_chars + d_offsets[idx] : nullptr;
    offset_type bytes    = 0;
    bool write_separator = false;

    for (auto itr = d_table.begin(); itr < d_table.end(); ++itr) {
      auto const d_column     = *itr;
      bool const null_element = d_column.is_null(idx);

      if (write_separator && (separate_nulls == separator_on_nulls::YES || !null_element)) {
        if (d_buffer) d_buffer = detail::copy_string(d_buffer, d_separator);
        bytes += d_separator.size_bytes();
        write_separator = false;
      }

      // write out column's row data (or narep if the row is null)
      auto const d_str = null_element ? d_narep.value() : d_column.element<string_view>(idx);
      if (d_buffer) d_buffer = detail::copy_string(d_buffer, d_str);
      bytes += d_str.size_bytes();

      write_separator =
        write_separator || (separate_nulls == separator_on_nulls::YES) || !null_element;
    }

    if (!d_chars) d_offsets[idx] = bytes;
  }
};

/**
 * @brief Single separator concatenate functor
 */
struct concat_strings_fn : concat_strings_base {
  string_view const d_separator;

  concat_strings_fn(table_device_view const& d_table,
                    string_view const& d_separator,
                    string_scalar_device_view const& d_narep,
                    separator_on_nulls separate_nulls)
    : concat_strings_base{d_table, d_narep, separate_nulls}, d_separator(d_separator)
  {
  }

  __device__ void operator()(size_type idx) { process_row(idx, d_separator); }
};

}  // namespace

std::unique_ptr<column> concatenate(table_view const& strings_columns,
                                    string_scalar const& separator,
                                    string_scalar const& narep,
                                    separator_on_nulls separate_nulls,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  auto const num_columns = strings_columns.num_columns();
  CUDF_EXPECTS(num_columns > 1, "At least two columns must be specified");
  // check all columns are of type string
  CUDF_EXPECTS(std::all_of(strings_columns.begin(),
                           strings_columns.end(),
                           [](auto c) { return c.type().id() == type_id::STRING; }),
               "All columns must be of type string");
  auto const strings_count = strings_columns.num_rows();
  if (strings_count == 0)  // empty begets empty
    return make_empty_column(data_type{type_id::STRING});

  CUDF_EXPECTS(separator.is_valid(), "Parameter separator must be a valid string_scalar");
  string_view d_separator(separator.data(), separator.size());
  auto d_narep = get_scalar_device_view(const_cast<string_scalar&>(narep));

  // Create device views from the strings columns.
  auto d_table = table_device_view::create(strings_columns, stream);
  concat_strings_fn fn{*d_table, d_separator, d_narep, separate_nulls};
  auto children = make_strings_children(fn, strings_count, stream, mr);

  // create resulting null mask
  auto [null_mask, null_count] = cudf::detail::valid_if(
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    [d_table = *d_table, d_narep] __device__(size_type idx) {
      if (d_narep.is_valid()) return true;
      return !thrust::any_of(
        thrust::seq, d_table.begin(), d_table.end(), [idx](auto col) { return col.is_null(idx); });
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
struct multi_separator_concat_fn : concat_strings_base {
  column_device_view const d_separators;
  string_scalar_device_view const d_separator_narep;

  multi_separator_concat_fn(table_device_view const& d_table,
                            column_device_view const& d_separators,
                            string_scalar_device_view const& d_separator_narep,
                            string_scalar_device_view const& d_narep,
                            separator_on_nulls separate_nulls)
    : concat_strings_base{d_table, d_narep, separate_nulls},
      d_separators(d_separators),
      d_separator_narep(d_separator_narep)
  {
  }

  __device__ void operator()(size_type idx)
  {
    if (d_separators.is_null(idx) && !d_separator_narep.is_valid()) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }

    auto const d_separator = d_separators.is_valid(idx) ? d_separators.element<string_view>(idx)
                                                        : d_separator_narep.value();
    // base class utility function handles the rest
    process_row(idx, d_separator);
  }
};

}  // namespace

std::unique_ptr<column> concatenate(table_view const& strings_columns,
                                    strings_column_view const& separators,
                                    string_scalar const& separator_narep,
                                    string_scalar const& col_narep,
                                    separator_on_nulls separate_nulls,
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
    return make_empty_column(data_type{type_id::STRING});

  // Invalid output column strings - null rows
  string_view const invalid_str{nullptr, 0};
  auto const separator_rep = get_scalar_device_view(const_cast<string_scalar&>(separator_narep));
  auto const col_rep       = get_scalar_device_view(const_cast<string_scalar&>(col_narep));
  auto const separator_col_view_ptr = column_device_view::create(separators.parent(), stream);
  auto const separator_col_view     = *separator_col_view_ptr;

  // Create device views from the strings columns.
  auto d_table = table_device_view::create(strings_columns, stream);

  multi_separator_concat_fn mscf{
    *d_table, separator_col_view, separator_rep, col_rep, separate_nulls};
  auto children = make_strings_children(mscf, strings_count, stream, mr);

  // Create resulting null mask
  auto [null_mask, null_count] = cudf::detail::valid_if(
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    [d_table = *d_table, separator_col_view, separator_rep, col_rep] __device__(size_type idx) {
      if (!separator_col_view.is_valid(idx) && !separator_rep.is_valid()) return false;
      if (col_rep.is_valid()) return true;
      return !thrust::any_of(
        thrust::seq, d_table.begin(), d_table.end(), [idx](auto col) { return col.is_null(idx); });
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
                                    separator_on_nulls separate_nulls,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate(
    strings_columns, separator, narep, separate_nulls, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> concatenate(table_view const& strings_columns,
                                    strings_column_view const& separators,
                                    string_scalar const& separator_narep,
                                    string_scalar const& col_narep,
                                    separator_on_nulls separate_nulls,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate(strings_columns,
                             separators,
                             separator_narep,
                             col_narep,
                             separate_nulls,
                             rmm::cuda_stream_default,
                             mr);
}

}  // namespace strings
}  // namespace cudf
