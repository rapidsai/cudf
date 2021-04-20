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
#include <cudf/detail/valid_if.cuh>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform_reduce.h>

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

  //  __device__ void operator()(size_type idx) {}
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

  // Accessing the child strings column of the lists column must be done by calling `child()`
  //  on the lists column, not `get_sliced_child()`
  // This is because calling to `offsets_begin()` on the lists column returns a pointer to the
  //  offsets of the original lists column, not sliced offsets (offsets starting from `0`)
  auto const lists_dv_ptr    = column_device_view::create(lists_strings_column.parent(), stream);
  auto const lists_dv        = *lists_dv_ptr;
  auto const strings_col     = strings_column_view(lists_strings_column.child());
  auto const strings_dv_ptr  = column_device_view::create(strings_col.parent(), stream);
  auto const strings_dv      = *strings_dv_ptr;
  auto const sep_dv_ptr      = column_device_view::create(separators.parent(), stream);
  auto const sep_dv          = *sep_dv_ptr;
  auto const sep_narep_dv    = get_scalar_device_view(const_cast<string_scalar&>(separator_narep));
  auto const string_narep_dv = get_scalar_device_view(const_cast<string_scalar&>(string_narep));

  // Compute sizes of strings in the output column along with their validity
  // An invalid size will be returned to indicate that the corresponding row is null
  static constexpr auto invalid_size = std::numeric_limits<size_type>::lowest();
  auto const string_size_comp_fn     = [lists_offsets = lists_strings_column.offsets_begin(),
                                    lists_dv,
                                    strings_dv,
                                    sep_dv,
                                    sep_narep_dv,
                                    string_narep_dv] __device__(size_type lidx) -> size_type {
    if (lists_dv.is_null(lidx) || (sep_dv.is_null(lidx) && !sep_narep_dv.is_valid())) {
      return invalid_size;
    }

    auto const separator_size =
      sep_dv.is_valid(lidx) ? sep_dv.element<string_view>(lidx).size_bytes() : sep_narep_dv.size();

    auto size_bytes = size_type{0};
    for (size_type str_idx = lists_offsets[lidx], idx_end = lists_offsets[lidx + 1];
         str_idx < idx_end;
         ++str_idx) {
      if (strings_dv.is_null(str_idx) && !string_narep_dv.is_valid()) {
        size_bytes = invalid_size;
        break;  // early termination: the entire list of strings will result in a null string
      }
      size_bytes += separator_size + (strings_dv.is_null(str_idx)
                                            ? string_narep_dv.size()
                                            : strings_dv.element<string_view>(str_idx).size_bytes());
    }

    // Null/empty separator and strings don't produce a non-empty string
    assert(size_bytes == invalid_size || size_bytes > separator_size ||
           (size_bytes == 0 && separator_size == 0));

    // Separator is inserted only in between strings
    return size_bytes != invalid_size ? static_cast<size_type>(size_bytes - separator_size)
                                          : invalid_size;
  };

  // Offset of the output strings
  static_assert(sizeof(offset_type) == sizeof(int32_t));
  auto offsets_column = make_numeric_column(
    data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);
  auto const output_offsets_ptr = offsets_column->mutable_view().begin<offset_type>();

  // Firstly, store the strings' sizes into output_str_offsets from the second element
  auto const count_it = thrust::make_counting_iterator<size_type>(0);
  CUDA_TRY(cudaMemsetAsync(output_offsets_ptr, 0, sizeof(int32_t), stream.value()));
  thrust::transform(rmm::exec_policy(stream),
                    count_it,
                    count_it + num_rows,
                    output_offsets_ptr + 1,
                    string_size_comp_fn);

  // Use the strings's sizes to compute null_mask and null_count of the output strings column
  auto [null_mask, null_count] = cudf::detail::valid_if(
    count_it,
    count_it + num_rows,
    [str_sizes = output_offsets_ptr + 1] __device__(size_type idx) {
      return str_sizes[idx] != invalid_size;
    },
    stream,
    mr);

  // Build the strings's offsets from strings' sizes
  auto const iter_trans_begin = thrust::make_transform_iterator(
    output_offsets_ptr + 1,
    [] __device__(auto const size) { return size != invalid_size ? size : 0; });
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         iter_trans_begin,
                         iter_trans_begin + num_rows,
                         output_offsets_ptr + 1);

  // Create the chars column to store the result strings
  auto const total_bytes = thrust::device_pointer_cast(output_offsets_ptr)[num_rows];
  auto chars_column = strings::detail::create_chars_child_column(num_rows, total_bytes, stream, mr);

  auto const concat_strings_fn = [lists_offsets = lists_strings_column.offsets_begin(),
                                  str_offsets   = output_offsets_ptr,
                                  output_begin  = chars_column->mutable_view().begin<char>(),
                                  strings_dv,
                                  sep_dv,
                                  sep_narep_dv,
                                  string_narep_dv] __device__(size_type out_idx) {
    if (str_offsets[out_idx + 1] == str_offsets[out_idx]) { return; }

    auto const separator =
      sep_dv.is_valid(out_idx) ? sep_dv.element<string_view>(out_idx) : sep_narep_dv.value();
    bool written    = false;
    auto output_ptr = output_begin + str_offsets[out_idx];

    for (size_type str_idx = lists_offsets[out_idx], idx_end = lists_offsets[out_idx + 1];
         str_idx < idx_end;
         ++str_idx) {
      auto const d_str = strings_dv.is_null(str_idx) ? string_narep_dv.value()
                                                     : strings_dv.element<string_view>(str_idx);
      // Separator is inserted only in between strings
      if (written) { output_ptr = detail::copy_string(output_ptr, separator); }
      output_ptr = detail::copy_string(output_ptr, d_str);
      written    = true;
    }
  };

  // Finally, fill the output chars column
  thrust::for_each_n(rmm::exec_policy(stream), count_it, num_rows, concat_strings_fn);

  return make_strings_column(num_rows,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             (null_count) ? std::move(null_mask) : rmm::device_buffer{},
                             stream,
                             mr);
}

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

  // Accessing the child strings column of the lists column must be done by calling `child()`
  //  on the lists column, not `get_sliced_child()`
  // This is because calling to `offsets_begin()` on the lists column returns a pointer to the
  //  offsets of the original lists column, not sliced offsets (offsets starting from `0`)
  auto const lists_dv_ptr    = column_device_view::create(lists_strings_column.parent(), stream);
  auto const lists_dv        = *lists_dv_ptr;
  auto const strings_col     = strings_column_view(lists_strings_column.child());
  auto const strings_dv_ptr  = column_device_view::create(strings_col.parent(), stream);
  auto const strings_dv      = *strings_dv_ptr;
  auto const sep_dv          = get_scalar_device_view(const_cast<string_scalar&>(separator));
  auto const string_narep_dv = get_scalar_device_view(const_cast<string_scalar&>(narep));

  // Compute sizes of strings in the output column along with their validity
  // An invalid size will be returned to indicate that the corresponding row is null
  static constexpr auto invalid_size = std::numeric_limits<size_type>::lowest();
  auto const separator_size          = separator.size();
  auto const string_size_comp_fn     = [lists_offsets = lists_strings_column.offsets_begin(),
                                    lists_dv,
                                    strings_dv,
                                    separator_size,
                                    string_narep_dv] __device__(size_type lidx) -> size_type {
    if (lists_dv.is_null(lidx)) { return invalid_size; }

    auto size_bytes = size_type{0};
    for (size_type str_idx = lists_offsets[lidx], idx_end = lists_offsets[lidx + 1];
         str_idx < idx_end;
         ++str_idx) {
      if (strings_dv.is_null(str_idx) && !string_narep_dv.is_valid()) {
        size_bytes = invalid_size;
        break;  // early termination: the entire list of strings will result in a null string
      }
      size_bytes += separator_size + (strings_dv.is_null(str_idx)
                                            ? string_narep_dv.size()
                                            : strings_dv.element<string_view>(str_idx).size_bytes());
    }

    // Null/empty separator and strings don't produce a non-empty string
    assert(size_bytes == invalid_size || size_bytes > separator_size ||
           (size_bytes == 0 && separator_size == 0));

    // Separator is inserted only in between strings
    return size_bytes != invalid_size ? static_cast<size_type>(size_bytes - separator_size)
                                          : invalid_size;
  };

  // Offset of the output strings
  static_assert(sizeof(offset_type) == sizeof(int32_t));
  auto offsets_column = make_numeric_column(
    data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);
  auto const output_offsets_ptr = offsets_column->mutable_view().begin<offset_type>();

  // Firstly, store the strings' sizes into output_str_offsets from the second element
  auto const count_it = thrust::make_counting_iterator<size_type>(0);
  CUDA_TRY(cudaMemsetAsync(output_offsets_ptr, 0, sizeof(int32_t), stream.value()));
  thrust::transform(rmm::exec_policy(stream),
                    count_it,
                    count_it + num_rows,
                    output_offsets_ptr + 1,
                    string_size_comp_fn);

  // Use the strings's sizes to compute null_mask and null_count of the output strings column
  auto [null_mask, null_count] = cudf::detail::valid_if(
    count_it,
    count_it + num_rows,
    [str_sizes = output_offsets_ptr + 1] __device__(size_type idx) {
      return str_sizes[idx] != invalid_size;
    },
    stream,
    mr);

  // Build the strings's offsets from strings' sizes
  auto const iter_trans_begin = thrust::make_transform_iterator(
    output_offsets_ptr + 1,
    [] __device__(auto const size) { return size != invalid_size ? size : 0; });
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         iter_trans_begin,
                         iter_trans_begin + num_rows,
                         output_offsets_ptr + 1);

  // Create the chars column to store the result strings
  auto const total_bytes = thrust::device_pointer_cast(output_offsets_ptr)[num_rows];
  auto chars_column = strings::detail::create_chars_child_column(num_rows, total_bytes, stream, mr);

  auto const concat_strings_fn = [lists_offsets = lists_strings_column.offsets_begin(),
                                  str_offsets   = output_offsets_ptr,
                                  output_begin  = chars_column->mutable_view().begin<char>(),
                                  strings_dv,
                                  sep_dv,
                                  string_narep_dv] __device__(size_type out_idx) {
    if (str_offsets[out_idx + 1] == str_offsets[out_idx]) { return; }

    auto const separator = sep_dv.value();
    bool written         = false;
    auto output_ptr      = output_begin + str_offsets[out_idx];

    for (size_type str_idx = lists_offsets[out_idx], idx_end = lists_offsets[out_idx + 1];
         str_idx < idx_end;
         ++str_idx) {
      auto const d_str = strings_dv.is_null(str_idx) ? string_narep_dv.value()
                                                     : strings_dv.element<string_view>(str_idx);
      // Separator is inserted only in between strings
      if (written) { output_ptr = detail::copy_string(output_ptr, separator); }
      output_ptr = detail::copy_string(output_ptr, d_str);
      written    = true;
    }
  };

  // Finally, fill the output chars column
  thrust::for_each_n(rmm::exec_policy(stream), count_it, num_rows, concat_strings_fn);

  return make_strings_column(num_rows,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             (null_count) ? std::move(null_mask) : rmm::device_buffer{},
                             stream,
                             mr);
}
}  // namespace detail

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
std::unique_ptr<column> concatenate_list_elements(lists_column_view const& lists_strings_column,
                                                  string_scalar const& separator,
                                                  string_scalar const& narep,
                                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate_list_elements(
    lists_strings_column, separator, narep, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
