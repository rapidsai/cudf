/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <strings/utilities.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/logical.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <algorithm>

namespace cudf {
namespace strings {
namespace detail {
//
std::unique_ptr<column> concatenate(table_view const& strings_columns,
                                    string_scalar const& separator,
                                    string_scalar const& narep,
                                    rmm::mr::device_memory_resource* mr,
                                    cudaStream_t stream = 0)
{
  auto num_columns = strings_columns.num_columns();
  CUDF_EXPECTS(num_columns > 0, "At least one column must be specified");
  // check all columns are of type string
  CUDF_EXPECTS(std::all_of(strings_columns.begin(),
                           strings_columns.end(),
                           [](auto c) { return c.type().id() == type_id::STRING; }),
               "All columns must be of type string");
  if (num_columns == 1)  // single strings column returns a copy
    return std::make_unique<column>(*(strings_columns.begin()), stream, mr);
  auto strings_count = strings_columns.num_rows();
  if (strings_count == 0)  // empty begets empty
    return detail::make_empty_strings_column(mr, stream);

  CUDF_EXPECTS(separator.is_valid(), "Parameter separator must be a valid string_scalar");
  string_view d_separator(separator.data(), separator.size());
  auto d_narep = get_scalar_device_view(const_cast<string_scalar&>(narep));

  // Create device views from the strings columns.
  auto table   = table_device_view::create(strings_columns, stream);
  auto d_table = *table;

  // create resulting null mask
  auto valid_mask = cudf::detail::valid_if(
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    [d_table, d_narep] __device__(size_type idx) {
      bool null_element = thrust::any_of(
        thrust::seq, d_table.begin(), d_table.end(), [idx](auto col) { return col.is_null(idx); });
      return (!null_element || d_narep.is_valid());
    },
    stream,
    mr);
  auto& null_mask       = valid_mask.first;
  auto const null_count = valid_mask.second;

  // build offsets column by computing sizes of each string in the output
  auto offsets_transformer = [d_table, num_columns, d_separator, d_narep] __device__(
                               size_type row_idx) {
    // for this row (idx), iterate over each column and add up the bytes
    bool null_element =
      thrust::any_of(thrust::seq, d_table.begin(), d_table.end(), [row_idx](auto const& d_column) {
        return d_column.is_null(row_idx);
      });
    if (null_element && !d_narep.is_valid()) return 0;
    size_type bytes = thrust::transform_reduce(
      thrust::seq,
      d_table.begin(),
      d_table.end(),
      [row_idx, d_separator, d_narep] __device__(column_device_view const& d_column) {
        return d_separator.size_bytes() + (d_column.is_null(row_idx)
                                             ? d_narep.size()
                                             : d_column.element<string_view>(row_idx).size_bytes());
      },
      0,
      thrust::plus<size_type>());
    // separator goes only in between elements
    if (bytes > 0)                        // if not null
      bytes -= d_separator.size_bytes();  // remove the last separator
    return bytes;
  };
  auto offsets_transformer_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_type>(0), offsets_transformer);
  auto offsets_column = detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
  auto d_results_offsets = offsets_column->view().data<int32_t>();

  // create the chars column
  size_type bytes = thrust::device_pointer_cast(d_results_offsets)[strings_count];
  auto chars_column =
    strings::detail::create_chars_child_column(strings_count, null_count, bytes, mr, stream);
  // fill the chars column
  auto d_results_chars = chars_column->mutable_view().data<char>();
  thrust::for_each_n(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<size_type>(0),
    strings_count,
    [d_table, num_columns, d_separator, d_narep, d_results_offsets, d_results_chars] __device__(
      size_type idx) {
      bool null_element = thrust::any_of(
        thrust::seq, d_table.begin(), d_table.end(), [idx](column_device_view const& col) {
          return col.is_null(idx);
        });
      if (null_element && !d_narep.is_valid())
        return;  // do not write to buffer at all if any column element for this row is null
      size_type offset = d_results_offsets[idx];
      char* d_buffer   = d_results_chars + offset;
      // write out each column's entry for this row
      for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
        auto d_column = d_table.column(col_idx);
        string_view d_str =
          d_column.is_null(idx) ? d_narep.value() : d_column.element<string_view>(idx);
        d_buffer = detail::copy_string(d_buffer, d_str);
        // separator goes only in between elements
        if (col_idx + 1 < num_columns) d_buffer = detail::copy_string(d_buffer, d_separator);
      }
    });

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
}

//
std::unique_ptr<column> join_strings(strings_column_view const& strings,
                                     string_scalar const& separator,
                                     string_scalar const& narep,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream = 0)
{
  auto strings_count = strings.size();
  if (strings_count == 0) return detail::make_empty_strings_column(mr, stream);

  CUDF_EXPECTS(separator.is_valid(), "Parameter separator must be a valid string_scalar");

  auto execpol = rmm::exec_policy(stream);
  string_view d_separator(separator.data(), separator.size());
  auto d_narep = get_scalar_device_view(const_cast<string_scalar&>(narep));

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  // create an offsets array for building the output memory layout
  rmm::device_vector<size_type> output_offsets(strings_count + 1);
  auto d_output_offsets = output_offsets.data().get();
  // using inclusive-scan to compute last entry which is the total size
  thrust::transform_inclusive_scan(
    execpol->on(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    d_output_offsets + 1,
    [d_strings, d_separator, d_narep] __device__(size_type idx) {
      size_type bytes = 0;
      if (d_strings.is_null(idx)) {
        if (!d_narep.is_valid()) return 0;  // skip nulls
        bytes += d_narep.size();
      } else
        bytes += d_strings.element<string_view>(idx).size_bytes();
      if ((idx + 1) < d_strings.size()) bytes += d_separator.size_bytes();
      return bytes;
    },
    thrust::plus<size_type>());
  CUDA_TRY(cudaMemsetAsync(d_output_offsets, 0, sizeof(size_type), stream));
  // total size is the last entry
  size_type bytes = output_offsets.back();

  // build offsets column (only 1 string so 2 offset entries)
  auto offsets_column =
    make_numeric_column(data_type{type_id::INT32}, 2, mask_state::UNALLOCATED, stream, mr);
  auto offsets_view = offsets_column->mutable_view();
  // set the first entry to 0 and the last entry to bytes
  int32_t new_offsets[] = {0, bytes};
  CUDA_TRY(cudaMemcpyAsync(offsets_view.data<int32_t>(),
                           new_offsets,
                           sizeof(new_offsets),
                           cudaMemcpyHostToDevice,
                           stream));

  // build null mask
  // only one entry so it is either all valid or all null
  size_type null_count = 0;
  rmm::device_buffer null_mask{0, stream, mr};  // init to null null-mask
  if (strings.null_count() == strings_count && !narep.is_valid()) {
    null_mask  = create_null_mask(1, cudf::mask_state::ALL_NULL, stream, mr);
    null_count = 1;
  }
  auto chars_column =
    detail::create_chars_child_column(strings_count, null_count, bytes, mr, stream);
  auto chars_view = chars_column->mutable_view();
  auto d_chars    = chars_view.data<char>();
  thrust::for_each_n(
    execpol->on(stream),
    thrust::make_counting_iterator<size_type>(0),
    strings_count,
    [d_strings, d_separator, d_narep, d_output_offsets, d_chars] __device__(size_type idx) {
      size_type offset = d_output_offsets[idx];
      char* d_buffer   = d_chars + offset;
      if (d_strings.is_null(idx)) {
        if (!d_narep.is_valid())
          return;  // do not write to buffer if element is null (including separator)
        d_buffer = detail::copy_string(d_buffer, d_narep.value());
      } else {
        string_view d_str = d_strings.element<string_view>(idx);
        d_buffer          = detail::copy_string(d_buffer, d_str);
      }
      if ((idx + 1) < d_strings.size()) d_buffer = detail::copy_string(d_buffer, d_separator);
    });

  return make_strings_column(1,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
}

//
std::unique_ptr<column> concatenate(table_view const& strings_columns,
                                    strings_column_view const& separators,
                                    string_scalar const& separator_narep,
                                    string_scalar const& col_narep,
                                    rmm::mr::device_memory_resource* mr,
                                    cudaStream_t stream = 0)
{
  auto num_columns = strings_columns.num_columns();
  CUDF_EXPECTS(num_columns > 0, "At least one column must be specified");
  // Check if all columns are of type string
  CUDF_EXPECTS(std::all_of(strings_columns.begin(),
                           strings_columns.end(),
                           [](auto c) { return c.type().id() == type_id::STRING; }),
               "All columns must be of type string");

  auto strings_count = strings_columns.num_rows();
  CUDF_EXPECTS(strings_count == separators.size(),
               "Separators column should be the same size as the strings columns");
  if (strings_count == 0)  // Empty begets empty
    return detail::make_empty_strings_column(mr, stream);

  // Invalid output column strings - null rows
  string_view const invalid_str{nullptr, 0};
  auto const separator_rep = get_scalar_device_view(const_cast<string_scalar&>(separator_narep));
  auto const col_rep       = get_scalar_device_view(const_cast<string_scalar&>(col_narep));
  auto const separator_col_view_ptr = column_device_view::create(separators.parent(), stream);
  auto const separator_col_view     = *separator_col_view_ptr;

  if (num_columns == 1) {
    // Shallow copy of the resultant strings
    rmm::device_vector<string_view> out_col_strings(strings_count);

    // Device view of the only column in the table view
    auto const col0_ptr = column_device_view::create(strings_columns.column(0), stream);
    auto const col0     = *col0_ptr;

    // Execute it on every element
    thrust::transform(
      rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(strings_count),
      out_col_strings.data().get(),
      // Output depends on the separator
      [col0, invalid_str, separator_col_view, separator_rep, col_rep] __device__(auto ridx) {
        if (!separator_col_view.is_valid(ridx) && !separator_rep.is_valid()) return invalid_str;
        if (col0.is_valid(ridx)) {
          auto sv = col0.element<string_view>(ridx);
          return sv.empty() ? string_view{} : sv;
        } else if (col_rep.is_valid()) {
          auto cv = col_rep.value();
          return cv.empty() ? string_view{} : cv;
        } else
          return invalid_str;
      });

    return make_strings_column(out_col_strings, invalid_str, stream, mr);
  }

  // Create device views from the strings columns.
  auto table   = table_device_view::create(strings_columns, stream);
  auto d_table = *table;

  // Create resulting null mask
  auto valid_mask = cudf::detail::valid_if(
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    [d_table, separator_col_view, separator_rep, col_rep] __device__(size_type ridx) {
      if (!separator_col_view.is_valid(ridx) && !separator_rep.is_valid()) return false;
      bool all_nulls =
        thrust::all_of(thrust::seq, d_table.begin(), d_table.end(), [ridx](auto const& col) {
          return col.is_null(ridx);
        });
      return all_nulls ? col_rep.is_valid() : true;
    },
    stream,
    mr);

  auto null_count = valid_mask.second;

  // Build offsets column by computing sizes of each string in the output
  auto offsets_transformer = [d_table, separator_col_view, separator_rep, col_rep] __device__(
                               size_type ridx) {
    // If the separator value for the row is null and if there aren't global separator
    // replacements, this row does not have any value - null row
    if (!separator_col_view.is_valid(ridx) && !separator_rep.is_valid()) return 0;

    // For this row (idx), iterate over each column and add up the bytes
    bool all_nulls =
      thrust::all_of(thrust::seq, d_table.begin(), d_table.end(), [ridx](auto const& d_column) {
        return d_column.is_null(ridx);
      });
    // If all column values are null and there isn't a global column replacement value, this row
    // is a null row
    if (all_nulls && !col_rep.is_valid()) return 0;

    // There is at least one non-null column value (it can still be empty though)
    auto separator_str = separator_col_view.is_valid(ridx)
                           ? separator_col_view.element<string_view>(ridx)
                           : separator_rep.value();

    size_type bytes = thrust::transform_reduce(
      thrust::seq,
      d_table.begin(),
      d_table.end(),
      [ridx, separator_str, col_rep] __device__(column_device_view const& d_column) {
        // If column is null and there isn't a valid column replacement, this isn't used in
        // final string concatenate
        if (d_column.is_null(ridx) && !col_rep.is_valid()) return 0;
        return separator_str.size_bytes() + (d_column.is_null(ridx)
                                               ? col_rep.size()
                                               : d_column.element<string_view>(ridx).size_bytes());
      },
      0,
      thrust::plus<size_type>());

    // Null/empty separator and columns doesn't produce a non-empty string
    if (bytes == 0) assert(separator_str.size_bytes() == 0);

    // Separator goes only in between elements
    return bytes - separator_str.size_bytes();
  };
  auto offsets_transformer_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_type>(0), offsets_transformer);
  auto offsets_column = detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
  auto d_results_offsets = offsets_column->view().data<int32_t>();

  // Create the chars column
  size_type bytes = thrust::device_pointer_cast(d_results_offsets)[strings_count];
  auto chars_column =
    strings::detail::create_chars_child_column(strings_count, null_count, bytes, mr, stream);

  // Fill the chars column
  auto d_results_chars = chars_column->mutable_view().data<char>();
  thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     strings_count,
                     [d_table,
                      num_columns,
                      d_results_offsets,
                      d_results_chars,
                      separator_col_view,
                      separator_rep,
                      col_rep] __device__(size_type ridx) {
                       // If the separator for this row is null and if there isn't a valid separator
                       // to replace, do not write anything for this row
                       if (!separator_col_view.is_valid(ridx) && !separator_rep.is_valid()) return;

                       bool all_nulls = thrust::all_of(
                         thrust::seq, d_table.begin(), d_table.end(), [ridx](auto const& col) {
                           return col.is_null(ridx);
                         });

                       // If all column values are null and there isn't a valid column replacement,
                       // skip this row
                       if (all_nulls && !col_rep.is_valid()) return;

                       size_type offset    = d_results_offsets[ridx];
                       char* d_buffer      = d_results_chars + offset;
                       bool colval_written = false;

                       // There is at least one non-null column value (it can still be empty though)
                       auto separator_str = separator_col_view.is_valid(ridx)
                                              ? separator_col_view.element<string_view>(ridx)
                                              : separator_rep.value();

                       // Write out each column's entry for this row
                       for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
                         auto d_column = d_table.column(col_idx);
                         // If the column isn't valid and if there isn't a replacement for it, skip
                         // it
                         if (d_column.is_null(ridx) && !col_rep.is_valid()) continue;

                         // Separator goes only in between elements
                         if (colval_written)
                           d_buffer = detail::copy_string(d_buffer, separator_str);

                         string_view d_str = d_column.is_null(ridx)
                                               ? col_rep.value()
                                               : d_column.element<string_view>(ridx);
                         d_buffer       = detail::copy_string(d_buffer, d_str);
                         colval_written = true;
                       }
                     });

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             (null_count) ? std::move(valid_mask.first) : rmm::device_buffer{},
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
  return detail::concatenate(strings_columns, separator, narep, mr);
}

std::unique_ptr<column> join_strings(strings_column_view const& strings,
                                     string_scalar const& separator,
                                     string_scalar const& narep,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::join_strings(strings, separator, narep, mr);
}

std::unique_ptr<column> concatenate(table_view const& strings_columns,
                                    strings_column_view const& separators,
                                    string_scalar const& separator_narep,
                                    string_scalar const& col_narep,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate(strings_columns, separators, separator_narep, col_narep, mr);
}

}  // namespace strings
}  // namespace cudf
