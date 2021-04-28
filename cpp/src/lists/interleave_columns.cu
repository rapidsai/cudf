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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {
namespace {
/**
 * @brief Generate list offsets and list validities for the output lists column from the table_view
 * of the input lists columns.
 */
std::pair<std::unique_ptr<column>, rmm::device_uvector<int8_t>>
generate_list_offsets_and_validities(table_view const& input,
                                     bool has_null_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  auto const num_cols         = input.num_columns();
  auto const num_rows         = input.num_rows();
  auto const num_output_lists = num_rows * num_cols;
  auto const table_dv_ptr     = table_device_view::create(input);

  // The output offsets column.
  static_assert(sizeof(offset_type) == sizeof(int32_t));
  static_assert(sizeof(size_type) == sizeof(int32_t));
  auto list_offsets = make_numeric_column(
    data_type{type_id::INT32}, num_output_lists + 1, mask_state::UNALLOCATED, stream, mr);
  auto const d_offsets = list_offsets->mutable_view().begin<offset_type>();

  // The array of int8_t to store validities for list elements.
  auto validities = rmm::device_uvector<int8_t>(has_null_mask ? num_output_lists : 0, stream);

  // Compute list sizes and validities.
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(num_output_lists),
    d_offsets,
    [num_cols,
     table_dv     = *table_dv_ptr,
     d_validities = validities.begin(),
     has_null_mask] __device__(size_type const dst_list_id) {
      auto const src_col_id     = dst_list_id % num_cols;
      auto const src_list_id    = dst_list_id / num_cols;
      auto const& src_lists_col = table_dv.column(src_col_id);
      if (has_null_mask) {
        d_validities[dst_list_id] = static_cast<int8_t>(src_lists_col.is_valid(src_list_id));
      }
      auto const src_list_offsets =
        src_lists_col.child(lists_column_view::offsets_column_index).data<size_type>() +
        src_lists_col.offset();
      return src_list_offsets[src_list_id + 1] - src_list_offsets[src_list_id];
    });

  // Compute offsets from sizes.
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_offsets, d_offsets + num_output_lists + 1, d_offsets);

  return {std::move(list_offsets), std::move(validities)};
}

/**
 * @brief Compute string sizes, string validities, and interleave string lists functor.
 *
 * This functor is executed twice. In the first pass, the sizes and validities of the output strings
 * will be computed. In the second pass, this will interleave the lists of strings of the given
 * table of lists columns.
 */
struct compute_string_sizes_and_interleave_lists_fn {
  table_device_view const table_dv;

  // Store list offsets of the output lists column.
  offset_type const* const dst_list_offsets;

  // Flag to specify whether to compute string validities.
  bool const has_null_mask;

  // Store offsets of the strings in the chars column.
  offset_type* d_offsets{nullptr};

  // If d_chars == nullptr: only compute sizes and validities of the output strings.
  // If d_chars != nullptr: only concatenate strings.
  char* d_chars{nullptr};

  // We need to set `1` or `0` for the validities of the strings in the child column.
  int8_t* d_validities{nullptr};

  __device__ void operator()(size_type const dst_list_id)
  {
    auto const num_cols    = table_dv.num_columns();
    auto const src_col_id  = dst_list_id % num_cols;
    auto const src_list_id = dst_list_id / num_cols;

    auto const& src_lists_col = table_dv.column(src_col_id);
    if (has_null_mask and src_lists_col.is_null(src_list_id)) { return; }

    auto const src_list_offsets =
      src_lists_col.child(lists_column_view::offsets_column_index).data<size_type>() +
      src_lists_col.offset();
    auto const& src_child = src_lists_col.child(lists_column_view::child_column_index);
    auto const src_child_offsets =
      src_child.child(strings_column_view::offsets_column_index).data<size_type>() +
      src_child.offset();

    // read_idx and write_idx are indices of string elements.
    size_type write_idx = dst_list_offsets[dst_list_id];
    if (not d_chars) {  // just compute sizes of strings within a list
      for (auto read_idx = src_list_offsets[src_list_id],
                end_idx  = src_list_offsets[src_list_id + 1];
           read_idx < end_idx;
           ++read_idx, ++write_idx) {
        if (has_null_mask) {
          d_validities[write_idx] = static_cast<int8_t>(src_child.is_valid(read_idx));
        }
        d_offsets[write_idx] = src_child_offsets[read_idx + 1] - src_child_offsets[read_idx];
      }
    } else {  // just copy the entire memory region containing all strings in the list
      // start_idx and end_idx are indices of character elements.
      auto const start_idx = src_child_offsets[src_list_offsets[src_list_id]];
      auto const end_idx   = src_child_offsets[src_list_offsets[src_list_id + 1]];
      if (start_idx < end_idx) {
        auto const input_ptr =
          src_child.child(strings_column_view::chars_column_index).data<char>() +
          src_child.offset() + start_idx;
        auto const output_ptr = d_chars + d_offsets[write_idx];
        thrust::copy(thrust::seq, input_ptr, input_ptr + end_idx - start_idx, output_ptr);
      }
    }
  }
};

/**
 * @brief Struct used in type_dispatcher to interleave list entries of the input lists columns and
 * output the results into a destination column.
 */
struct interleave_list_entries_fn {
  template <class T>
  std::enable_if_t<std::is_same_v<T, cudf::string_view>, std::unique_ptr<column>> operator()(
    table_view const& input,
    column_view const& output_list_offsets,
    size_type num_output_lists,
    size_type num_output_entries,
    bool has_null_mask,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const noexcept
  {
    auto const table_dv_ptr = table_device_view::create(input);
    auto const comp_fn      = compute_string_sizes_and_interleave_lists_fn{
      *table_dv_ptr, output_list_offsets.begin<offset_type>(), has_null_mask};

    // Cannot call `cudf::strings::detail::make_strings_children` because that function executes the
    // functor `comp_fn` on the range equal to `num_output_entries`.
    auto [offsets_column, chars_column, null_mask, null_count] =
      cudf::strings::detail::make_strings_children_with_null_mask(
        comp_fn, num_output_lists, num_output_entries, stream, mr);

    if (has_null_mask and null_count > 0) {
      return make_strings_column(num_output_entries,
                                 std::move(offsets_column),
                                 std::move(chars_column),
                                 null_count,
                                 std::move(null_mask),
                                 stream,
                                 mr);
    }
    return make_strings_column(num_output_entries,
                               std::move(offsets_column),
                               std::move(chars_column),
                               0,
                               rmm::device_buffer{},
                               stream,
                               mr);
  }

  template <class T>
  std::enable_if_t<cudf::is_fixed_width<T>(), std::unique_ptr<column>> operator()(
    table_view const& input,
    column_view const& output_list_offsets,
    size_type num_output_lists,
    size_type num_output_entries,
    bool has_null_mask,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const noexcept
  {
    auto const num_cols     = input.num_columns();
    auto const num_rows     = input.num_rows();
    auto const table_dv_ptr = table_device_view::create(input);

    // The output child column.
    auto const child_col = lists_column_view(*input.begin()).child();
    auto output =
      allocate_like(child_col, num_output_entries, mask_allocation_policy::NEVER, stream, mr);
    auto output_dv_ptr = mutable_column_device_view::create(*output);

    // The array of int8_t to store entry validities.
    auto validities = rmm::device_uvector<int8_t>(has_null_mask ? num_output_entries : 0, stream);

    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<size_type>(0),
      num_output_lists,
      [num_cols,
       table_dv         = *table_dv_ptr,
       d_validities     = validities.begin(),
       dst_list_offsets = output_list_offsets.begin<offset_type>(),
       d_output         = output_dv_ptr->begin<T>(),
       has_null_mask] __device__(size_type const dst_list_id) {
        auto const src_col_id     = dst_list_id % num_cols;
        auto const src_list_id    = dst_list_id / num_cols;
        auto const& src_lists_col = table_dv.column(src_col_id);
        auto const src_list_offsets =
          src_lists_col.child(lists_column_view::offsets_column_index).data<size_type>() +
          src_lists_col.offset();
        auto const& src_child = src_lists_col.child(lists_column_view::child_column_index);

        // The indices of the entries in the list to gather from.
        auto const start_idx   = src_list_offsets[src_list_id];
        auto const end_idx     = src_list_offsets[src_list_id + 1];
        auto const write_start = dst_list_offsets[dst_list_id];

        // Fill the validities array if necessary.
        if (has_null_mask) {
          for (auto read_idx = start_idx, write_idx = write_start; read_idx < end_idx;
               ++read_idx, ++write_idx) {
            d_validities[write_idx] = static_cast<int8_t>(src_child.is_valid(read_idx));
          }
        }

        // Do a copy for the entire list entries.
        auto const input_ptr  = reinterpret_cast<char const*>(src_child.data<T>() + start_idx);
        auto const output_ptr = reinterpret_cast<char*>(&d_output[write_start]);
        thrust::copy(
          thrust::seq, input_ptr, input_ptr + sizeof(T) * (end_idx - start_idx), output_ptr);
      });

    if (has_null_mask) {
      auto [null_mask, null_count] = cudf::detail::valid_if(
        validities.begin(), validities.end(), thrust::identity<int8_t>{}, stream, mr);
      if (null_count > 0) { output->set_null_mask(null_mask, null_count); }
    }

    return output;
  }

  template <class T>
  std::enable_if_t<not std::is_same_v<T, cudf::string_view> and not cudf::is_fixed_width<T>(),
                   std::unique_ptr<column>>
  operator()(table_view const&,
             column_view const&,
             size_type,
             size_type,
             bool,
             rmm::cuda_stream_view,
             rmm::mr::device_memory_resource*) const
  {
    // Currently, only support string_view and fixed-width types
    CUDF_FAIL("Called `interleave_list_entries_fn()` on non-supported types.");
  }
};

}  // anonymous namespace

/**
 * @copydoc cudf::lists::detail::interleave_columns
 *
 */
std::unique_ptr<column> interleave_columns(table_view const& input,
                                           bool has_null_mask,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  auto const entry_type = lists_column_view(*input.begin()).child().type();
  for (auto const& col : input) {
    CUDF_EXPECTS(col.type().id() == type_id::LIST,
                 "All columns of the input table must be of lists column type.");

    auto const child_col = lists_column_view(col).child();
    CUDF_EXPECTS(not cudf::is_nested(child_col.type()), "Nested types are not supported.");
    CUDF_EXPECTS(entry_type == child_col.type(),
                 "The types of entries in the input columns must be the same.");
  }

  if (input.num_rows() == 0) { return cudf::empty_like(input.column(0)); }
  if (input.num_columns() == 1) { return std::make_unique<column>(*(input.begin()), stream, mr); }

  // Generate offsets of the output lists column.
  auto [list_offsets, list_validities] =
    generate_list_offsets_and_validities(input, has_null_mask, stream, mr);
  auto const offsets_view = list_offsets->view();

  // Copy entries from the input lists columns to the output lists column - this needed to be
  // specialized for different types.
  auto const num_output_lists = input.num_rows() * input.num_columns();
  auto const num_output_entries =
    cudf::detail::get_value<size_type>(offsets_view, num_output_lists, stream);
  auto list_entries = type_dispatcher<dispatch_storage_type>(entry_type,
                                                             interleave_list_entries_fn{},
                                                             input,
                                                             offsets_view,
                                                             num_output_lists,
                                                             num_output_entries,
                                                             has_null_mask,
                                                             stream,
                                                             mr);

  if (not has_null_mask) {
    return make_lists_column(num_output_lists,
                             std::move(list_offsets),
                             std::move(list_entries),
                             0,
                             rmm::device_buffer{},
                             stream,
                             mr);
  }

  auto [null_mask, null_count] = cudf::detail::valid_if(
    list_validities.begin(), list_validities.end(), thrust::identity<int8_t>{}, stream, mr);
  return make_lists_column(num_output_lists,
                           std::move(list_offsets),
                           std::move(list_entries),
                           null_count,
                           null_count ? std::move(null_mask) : rmm::device_buffer{},
                           stream,
                           mr);
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
