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
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/utilities.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/logical.h>
#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {
namespace {
/**
 * @brief Concatenate lists within the same row into one list, ignoring any null list during
 * concatenation.
 */
std::unique_ptr<column> concatenate_lists_ignore_null(column_view const& input,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource* mr)
{
  auto const num_rows = input.size();

  static_assert(std::is_same_v<offset_type, int32_t> and std::is_same_v<size_type, int32_t>);
  auto out_offsets = make_numeric_column(
    data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);

  auto const d_out_offsets  = out_offsets->mutable_view().template begin<offset_type>();
  auto const d_row_offsets  = lists_column_view(input).offsets_begin();
  auto const d_list_offsets = lists_column_view(lists_column_view(input).child()).offsets_begin();

  // Concatenating the lists at the same row by converting the entry offsets from the child column
  // into row offsets of the root column. Those entry offsets are subtracted by the first entry
  // offset to output zero-based offsets.
  auto const iter = thrust::make_counting_iterator<size_type>(0);
  thrust::transform(rmm::exec_policy(stream),
                    iter,
                    iter + num_rows + 1,
                    d_out_offsets,
                    [d_row_offsets, d_list_offsets] __device__(auto const idx) {
                      auto const start_offset = d_list_offsets[d_row_offsets[0]];
                      return d_list_offsets[d_row_offsets[idx]] - start_offset;
                    });

  // The child column of the output lists column is just copied from the input column.
  auto out_entries = std::make_unique<column>(
    lists_column_view(lists_column_view(input).get_sliced_child(stream)).get_sliced_child(stream));

  return make_lists_column(num_rows,
                           std::move(out_offsets),
                           std::move(out_entries),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input, stream, mr),
                           stream,
                           mr);
}

/**
 * @brief Generate list offsets and list validities for the output lists column.
 *
 * This function is called only when (has_null_list == true and null_policy == NULLIFY_OUTPUT_ROW).
 */
std::pair<std::unique_ptr<column>, rmm::device_uvector<int8_t>>
generate_list_offsets_and_validities(column_view const& input,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  auto const num_rows = input.size();

  static_assert(std::is_same_v<offset_type, int32_t> and std::is_same_v<size_type, int32_t>);
  auto out_offsets = make_numeric_column(
    data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);

  auto const lists_dv_ptr   = column_device_view::create(lists_column_view(input).child());
  auto const d_out_offsets  = out_offsets->mutable_view().template begin<offset_type>();
  auto const d_row_offsets  = lists_column_view(input).offsets_begin();
  auto const d_list_offsets = lists_column_view(lists_column_view(input).child()).offsets_begin();

  // The array of int8_t stores validities for the output list elements.
  auto validities = rmm::device_uvector<int8_t>(num_rows, stream);

  // Compute output list sizes and validities.
  auto const iter = thrust::make_counting_iterator<size_type>(0);
  thrust::transform(
    rmm::exec_policy(stream),
    iter,
    iter + num_rows,
    d_out_offsets,
    [lists_dv = *lists_dv_ptr,
     d_row_offsets,
     d_list_offsets,
     d_validities = validities.begin(),
     iter] __device__(auto const idx) {
      if (d_row_offsets[idx] == d_row_offsets[idx + 1]) {  // This is a null row.
        return size_type{0};
      }
      // The output row will not be null only if all lists of the input row are not null.
      auto const is_valid =
        thrust::all_of(thrust::seq,
                       iter + d_row_offsets[idx],
                       iter + d_row_offsets[idx + 1],
                       [&] __device__(auto const list_idx) { return lists_dv.is_valid(list_idx); });
      d_validities[idx] = static_cast<int8_t>(is_valid);
      if (not is_valid) { return size_type{0}; }

      // Compute size of the output list as sum of sizes of all lists in the current input row.
      return d_list_offsets[d_row_offsets[idx + 1]] - d_list_offsets[d_row_offsets[idx]];
    });

  // Compute offsets from sizes.
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_out_offsets, d_out_offsets + num_rows + 1, d_out_offsets);

  return {std::move(out_offsets), std::move(validities)};
}

/**
 * @brief Compute string sizes, string validities, and concatenate string lists functor.
 *
 * This functor is called only when (has_null_list == true and null_policy == NULLIFY_OUTPUT_ROW).
 * It is executed twice. In the first pass, the sizes and validities of the output strings will be
 * computed. In the second pass, this will concatenate the lists of strings within each row of the
 * given lists columns.
 */
struct compute_string_sizes_and_concatenate_lists_fn {
  column_device_view const strs_col;

  // Offsets of the input lists column and its child.
  offset_type const* const row_offsets;
  offset_type const* const list_offsets;

  // Store list offsets of the output lists column.
  offset_type const* const out_list_offsets;

  // Store offsets of the strings.
  offset_type* d_offsets{nullptr};

  // If d_chars == nullptr: only compute sizes and validities of the output strings.
  // If d_chars != nullptr: only concatenate lists.
  char* d_chars{nullptr};

  // We need to set `1` or `0` for the validities of the strings in the child column.
  int8_t* d_validities{nullptr};

  __device__ void operator()(size_type const idx)
  {
    // The current row contain null, which has been identified during generating the offsets.
    if (out_list_offsets[idx + 1] == out_list_offsets[idx]) { return; }

    // read_idx and write_idx are indices of string elements.
    size_type write_idx = out_list_offsets[idx];
    auto const str_offsets =
      strs_col.child(strings_column_view::offsets_column_index).template data<offset_type>();

    // The range of indices of the strings within the same row.
    auto const start_str_idx = list_offsets[row_offsets[idx]];
    auto const end_str_idx   = list_offsets[row_offsets[idx + 1]];

    if (not d_chars) {  // just compute sizes of strings within a list
      for (auto read_idx = start_str_idx; read_idx < end_str_idx; ++read_idx, ++write_idx) {
        d_validities[write_idx] = static_cast<int8_t>(strs_col.is_valid(read_idx));
        d_offsets[write_idx]    = str_offsets[read_idx + 1] - str_offsets[read_idx];
      }
    } else {  // just copy the entire memory region containing all strings in the row `idx`
      // start_byte and end_byte are indices of character of the string elements.
      auto const start_byte = str_offsets[start_str_idx];
      auto const end_byte   = str_offsets[end_str_idx];
      if (start_byte < end_byte) {
        auto const input_ptr =
          strs_col.child(strings_column_view::chars_column_index).template data<char>() +
          start_byte;
        auto const output_ptr = d_chars + d_offsets[write_idx];
        thrust::copy(thrust::seq, input_ptr, input_ptr + end_byte - start_byte, output_ptr);
      }
      write_idx += end_str_idx - start_str_idx;
    }
  }
};

/**
 * @brief Struct used in type_dispatcher to concatenate list elements at the same row and
 * output the results into a destination column.
 *
 * This functor is called only when (has_null_list == true and null_policy == NULLIFY_OUTPUT_ROW).
 */
struct concatenate_lists_fn {
  template <class T>
  std::enable_if_t<std::is_same_v<T, cudf::string_view>, std::unique_ptr<column>> operator()(
    column_view const& input,
    column_view const& output_list_offsets,
    size_type num_rows,
    size_type num_output_entries,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const noexcept
  {
    auto const child_col      = lists_column_view(input).child();
    auto const d_row_offsets  = lists_column_view(input).offsets_begin();
    auto const d_list_offsets = lists_column_view(child_col).offsets_begin();
    auto const strs_col       = lists_column_view(child_col).child();
    auto const strs_dv_ptr    = column_device_view::create(strs_col);
    auto const comp_fn        = compute_string_sizes_and_concatenate_lists_fn{
      *strs_dv_ptr,
      d_row_offsets,
      d_list_offsets,
      output_list_offsets.template begin<offset_type>()};

    auto [offsets_column, chars_column, null_mask, null_count] =
      cudf::strings::detail::make_strings_children_with_null_mask(
        comp_fn, num_rows, num_output_entries, stream, mr);

    return make_strings_column(num_output_entries,
                               std::move(offsets_column),
                               std::move(chars_column),
                               null_count,
                               std::move(null_mask),
                               stream,
                               mr);
  }

  template <class T>
  std::enable_if_t<cudf::is_fixed_width<T>(), std::unique_ptr<column>> operator()(
    column_view const& input,
    column_view const& output_list_offsets,
    size_type num_rows,
    size_type num_output_entries,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const noexcept
  {
    auto const child_col      = lists_column_view(input).child();
    auto const d_row_offsets  = lists_column_view(input).offsets_begin();
    auto const d_list_offsets = lists_column_view(child_col).offsets_begin();
    auto const entry_col      = lists_column_view(child_col).child();
    auto const entry_dv_ptr   = column_device_view::create(entry_col);

    // The output child column.
    auto output =
      allocate_like(entry_col, num_output_entries, mask_allocation_policy::NEVER, stream, mr);
    auto output_dv_ptr = mutable_column_device_view::create(*output);

    // The array of int8_t to store entry validities.
    auto validities = rmm::device_uvector<int8_t>(num_output_entries, stream);

    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<size_type>(0),
      num_rows,
      [d_row_offsets,
       d_list_offsets,
       entry_dv           = *entry_dv_ptr,
       d_out_list_offsets = output_list_offsets.template begin<offset_type>(),
       d_entry_validities = validities.begin(),
       d_entries          = output_dv_ptr->template begin<T>()] __device__(size_type const idx) {
        // The output row has been identified as a null list during list size computation.
        if (d_out_list_offsets[idx + 1] == d_out_list_offsets[idx]) { return; }

        auto write_start = d_out_list_offsets[idx];

        // The range of indices of the entries within the same row.
        auto const start_idx = d_list_offsets[d_row_offsets[idx]];
        auto const end_idx   = d_list_offsets[d_row_offsets[idx + 1]];

        // Fill the validities array.
        for (auto read_idx = start_idx, write_idx = write_start; read_idx < end_idx;
             ++read_idx, ++write_idx) {
          d_entry_validities[write_idx] = static_cast<int8_t>(entry_dv.is_valid(read_idx));
        }
        // Do a memcopy for all entries in the same row.
        auto const input_ptr =
          reinterpret_cast<char const*>(entry_dv.template data<T>() + start_idx);
        auto const output_ptr = reinterpret_cast<char*>(&d_entries[write_start]);
        thrust::copy(
          thrust::seq, input_ptr, input_ptr + sizeof(T) * (end_idx - start_idx), output_ptr);
        write_start += end_idx - start_idx;
      });

    auto [null_mask, null_count] = cudf::detail::valid_if(
      validities.begin(), validities.end(), thrust::identity<int8_t>{}, stream, mr);
    if (null_count > 0) { output->set_null_mask(null_mask, null_count); }

    return output;
  }

  template <class T>
  std::enable_if_t<not std::is_same_v<T, cudf::string_view> and not cudf::is_fixed_width<T>(),
                   std::unique_ptr<column>>
  operator()(column_view const&,
             column_view const&,
             size_type,
             size_type,
             rmm::cuda_stream_view,
             rmm::mr::device_memory_resource*) const
  {
    // Currently, only support string_view and fixed-width types
    CUDF_FAIL("Called `concatenate_lists_fn()` on non-supported types.");
  }
};

std::unique_ptr<column> concatenate_lists_nullifying_rows(column_view const& input,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::mr::device_memory_resource* mr)
{
  // Generate offsets of the output lists column.
  auto [list_offsets, list_validities] = generate_list_offsets_and_validities(input, stream, mr);
  auto const offsets_view              = list_offsets->view();

  // Copy entries from the input lists columns to the output lists column - this needed to be
  // specialized for different types.
  auto const num_rows = input.size();
  auto const num_output_entries =
    cudf::detail::get_value<size_type>(offsets_view, num_rows, stream);
  auto const entry_type = lists_column_view(lists_column_view(input).child()).child().type();
  auto list_entries     = type_dispatcher<dispatch_storage_type>(entry_type,
                                                             concatenate_lists_fn{},
                                                             input,
                                                             offsets_view,
                                                             num_rows,
                                                             num_output_entries,
                                                             stream,
                                                             mr);

  auto [null_mask, null_count] = cudf::detail::valid_if(
    list_validities.begin(), list_validities.end(), thrust::identity<int8_t>{}, stream, mr);
  return make_lists_column(num_rows,
                           std::move(list_offsets),
                           std::move(list_entries),
                           null_count,
                           null_count ? std::move(null_mask) : rmm::device_buffer{},
                           stream,
                           mr);
}

}  // namespace

/**
 * @copydoc cudf::lists::concatenate_list_elements
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> concatenate_list_elements(column_view const& input,
                                                  concatenate_null_policy null_policy,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  auto type = input.type();  // Column that is lists of lists.
  CUDF_EXPECTS(type.id() == type_id::LIST, "Input column must be a lists column.");

  auto col = lists_column_view(input).child();  // Rows, which are lists.
  type     = col.type();
  CUDF_EXPECTS(type.id() == type_id::LIST, "Rows of the input column must be lists.");

  col  = lists_column_view(col).child();  // Lowest level entries, which are of non-nested types.
  type = col.type();
  CUDF_EXPECTS(not cudf::is_nested(type),
               "Entry of the input lists column must be of non-nested types.");

  if (input.size() == 0) { return cudf::empty_like(input); }

  return (null_policy == concatenate_null_policy::IGNORE or
          not lists_column_view(input).child().has_nulls())
           ? concatenate_lists_ignore_null(input, stream, mr)
           : concatenate_lists_nullifying_rows(input, stream, mr);
}

}  // namespace detail

/**
 * @copydoc cudf::lists::concatenate_list_elements
 */
std::unique_ptr<column> concatenate_list_elements(column_view const& input,
                                                  concatenate_null_policy null_policy,
                                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate_list_elements(input, null_policy, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
