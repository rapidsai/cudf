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

#include <strings/utilities.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/detail/sorting.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <cudf_test/column_utilities.hpp>

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
  auto const num_cols  = input.num_columns();
  auto const num_rows  = input.num_rows();
  auto const num_lists = num_rows * num_cols;

  // The output offsets column
  static_assert(sizeof(offset_type) == sizeof(int32_t));
  static_assert(sizeof(size_type) == sizeof(int32_t));
  auto list_offsets = make_numeric_column(
    data_type{type_id::INT32}, num_lists + 1, mask_state::UNALLOCATED, stream, mr);
  auto const d_out_offsets = list_offsets->mutable_view().begin<offset_type>();
  auto const table_dv_ptr  = table_device_view::create(input);

  // The array of int8_t to store element validities
  auto validities = has_null_mask ? rmm::device_uvector<int8_t>(num_lists, stream)
                                  : rmm::device_uvector<int8_t>(0, stream);

  // Compute list sizes
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(num_lists),
                    d_out_offsets,
                    [num_cols,
                     table_dv     = *table_dv_ptr,
                     d_validities = validities.begin(),
                     has_null_mask] __device__(size_type const dst_list_id) {
                      auto const src_col_id  = dst_list_id % num_cols;
                      auto const src_list_id = dst_list_id / num_cols;
                      auto const& src_col    = table_dv.column(src_col_id);
                      auto const is_valid    = src_col.is_valid(src_list_id);
                      if (has_null_mask) {
                        d_validities[dst_list_id] = static_cast<int8_t>(is_valid);
                      }
                      if (not is_valid) { return size_type{0}; }
                      auto const d_offsets =
                        src_col.child(lists_column_view::offsets_column_index).data<size_type>() +
                        src_col.offset();
                      return d_offsets[src_list_id + 1] - d_offsets[src_list_id];
                    });

  // Compute offsets from sizes
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_out_offsets, d_out_offsets + num_lists + 1, d_out_offsets);

  return {std::move(list_offsets), std::move(validities)};
}

/**
 * @brief Creates child offsets, chars columns and null mask, null count of a strings column by
 * applying the template function that can be used for computing the output size of each string as
 * well as create the output.
 *
 * @tparam SizeAndExecuteFunction Function must accept an index and return a size.
 *         It must have members `d_offsets`, `d_chars`, and `d_validities` which are set to memory
 *         containing the offsets column, chars column and string validities during write.
 *
 * @param size_and_exec_fn This is called twice. Once for the output size of each string, which is
 *                         written into the `d_offsets` array. After that, `d_chars` is set and this
 *                         is called again to fill in the chars memory. The `d_validities` array may
 *                         be modified to set the value `0` for the corresponding rows that contain
 *                         null string elements.
 * @param strings_count Number of strings.
 * @param mr Device memory resource used to allocate the returned columns' device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return offsets child column, chars child column, null_mask, and null_count for a strings column.
 */
template <typename SizeAndExecuteFunction>
std::tuple<std::unique_ptr<column>, std::unique_ptr<column>, rmm::device_buffer, size_type>
make_strings_children_with_null_mask(
  SizeAndExecuteFunction size_and_exec_fn,
  size_type strings_count,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto offsets_column = make_numeric_column(
    data_type{type_id::INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto offsets_view          = offsets_column->mutable_view();
  auto d_offsets             = offsets_view.template data<int32_t>();
  size_and_exec_fn.d_offsets = d_offsets;

  auto validities               = rmm::device_uvector<int8_t>(strings_count, stream);
  size_and_exec_fn.d_validities = validities.begin();

  // This is called twice: once for offsets and validities, and once for chars
  auto for_each_fn = [strings_count, stream](SizeAndExecuteFunction& size_and_exec_fn) {
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       strings_count,
                       size_and_exec_fn);
  };

  // Compute the string sizes (storing in `d_offsets`) and string validities
  for_each_fn(size_and_exec_fn);

  // Compute the offsets from string sizes
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_offsets, d_offsets + strings_count + 1, d_offsets);

  // Now build the chars column
  auto const bytes = cudf::detail::get_value<int32_t>(offsets_view, strings_count, stream);
  auto chars_column =
    cudf::strings::detail::create_chars_child_column(strings_count, 0, bytes, stream, mr);

  // Execute the function fn again to fill the chars column.
  // Note that if the output chars column has zero size, the function fn should not be called to
  // avoid accidentally overwriting the offsets.
  if (bytes > 0) {
    size_and_exec_fn.d_chars = chars_column->mutable_view().template data<char>();
    for_each_fn(size_and_exec_fn);
  }

  // Finally compute null mask and null count from the validities array
  auto [null_mask, null_count] = cudf::detail::valid_if(
    validities.begin(),
    validities.end(),
    [] __device__(auto const valid) { return valid; },
    stream,
    mr);

  return std::make_tuple(std::move(offsets_column),
                         std::move(chars_column),
                         null_count > 0 ? std::move(null_mask) : rmm::device_buffer{},
                         null_count);
}

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
struct compute_size_and_copy_fn {
  table_device_view const table_dv;

  // Store offsets of the lists in the output lists column
  offset_type const* const dst_list_offsets;

  // Store offsets of the strings in the chars column
  offset_type* d_offsets{nullptr};

  // If d_chars == nullptr: only compute sizes and validities of the output strings.
  // If d_chars != nullptr: only concatenate strings.
  char* d_chars{nullptr};

  // We need to set `1` or `0` for the validities of the strings in the child column.
  int8_t* d_validities{nullptr};

  bool has_null_mask;

  __device__ void operator()(size_type const dst_list_id)
  {
    auto const num_cols       = table_dv.num_columns();
    auto const src_col_id     = dst_list_id % num_cols;
    auto const src_list_id    = dst_list_id / num_cols;
    auto const& src_lists_col = table_dv.column(src_col_id);
    if (has_null_mask and src_lists_col.is_null(src_list_id)) { return; }

    auto const src_list_offsets =
      src_lists_col.child(lists_column_view::offsets_column_index).data<size_type>() +
      src_lists_col.offset();
    auto const& src_child = src_lists_col.child(lists_column_view::child_column_index);
    auto const src_child_offsets =
      src_child.child(strings_column_view::offsets_column_index).data<size_type>() +
      src_child.offset();

    size_type write_idx = dst_list_offsets[dst_list_id];
    if (not d_chars) {  // just compute sizes of strings within a list
      for (auto read_idx = src_list_offsets[src_list_id],
                end_idx  = src_list_offsets[src_list_id + 1];
           read_idx < end_idx;
           ++read_idx, ++write_idx) {
        auto const is_valid = src_child.is_valid(read_idx);
        if (has_null_mask) { d_validities[write_idx] = static_cast<int8_t>(is_valid); }
        d_offsets[write_idx] =
          is_valid ? src_child_offsets[read_idx + 1] - src_child_offsets[read_idx] : 0;
      }
    } else {  // just copy the entire list of strings
      auto const start_idx = src_child_offsets[src_list_offsets[src_list_id]];
      auto const end_idx   = src_child_offsets[src_list_offsets[src_list_id + 1] - 1];
      if (start_idx < end_idx) {
        auto const input_ptr =
          src_child.child(strings_column_view::chars_column_index).data<char>() +
          src_child.offset() + start_idx;
        auto const output_ptr = d_chars + d_offsets[write_idx];
        std::memcpy(output_ptr, input_ptr, end_idx - start_idx);
      }
    }
  }
};

/**
 * @brief Struct used in type_dispatcher to copy entries to the output lists column
 */
struct copy_list_entries_fn {
  template <class T>
  std::enable_if_t<std::is_same_v<T, cudf::string_view>, std::unique_ptr<column>> operator()(
    table_view const& input,
    column_view const& list_offsets,
    size_type num_strings,
    bool has_null_mask,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const noexcept
  {
    auto const table_dv_ptr = table_device_view::create(input);
    auto const comp_fn = compute_size_and_copy_fn{*table_dv_ptr, list_offsets.begin<offset_type>()};
    if (not has_null_mask) {
      auto [offsets_column, chars_column] =
        cudf::strings::detail::make_strings_children(comp_fn, num_strings, 0, stream, mr);
      return make_strings_column(num_strings,
                                 std::move(offsets_column),
                                 std::move(chars_column),
                                 0,
                                 rmm::device_buffer{},
                                 stream,
                                 mr);
    }

    auto [offsets_column, chars_column, null_mask, null_count] =
      make_strings_children_with_null_mask(comp_fn, num_strings, stream, mr);
    return make_strings_column(num_strings,
                               std::move(offsets_column),
                               std::move(chars_column),
                               null_count,
                               std::move(null_mask),
                               stream,
                               mr);
  }

  template <class T>
  std::enable_if_t<cudf::is_fixed_width<T>(), std::unique_ptr<column>> operator()(
    table_view const& input,
    column_view const& list_offsets,
    size_type num_output_entries,
    bool has_null_mask,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const noexcept
  {
    auto const child_col    = lists_column_view(*input.begin()).child();
    auto const num_cols     = input.num_columns();
    auto const num_rows     = input.num_rows();
    auto const num_lists    = num_rows * num_cols;
    auto const table_dv_ptr = table_device_view::create(input);

    auto output =
      allocate_like(child_col, num_output_entries, mask_allocation_policy::NEVER, stream, mr);
    auto output_dv_ptr = mutable_column_device_view::create(*output);

    // The array of int8_t to store element validities
    auto validities = has_null_mask ? rmm::device_uvector<int8_t>(num_output_entries, stream)
                                    : rmm::device_uvector<int8_t>(0, stream);

    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<size_type>(0),
      num_lists,
      [num_cols,
       table_dv         = *table_dv_ptr,
       out_validities   = validities.begin(),
       dst_list_offsets = list_offsets.begin<offset_type>(),
       out_entries      = output_dv_ptr->begin<T>(),
       has_null_mask] __device__(size_type const dst_list_id) {
        auto const src_col_id     = dst_list_id % num_cols;
        auto const src_list_id    = dst_list_id / num_cols;
        auto const& src_lists_col = table_dv.column(src_col_id);
        auto const src_list_offsets =
          src_lists_col.child(lists_column_view::offsets_column_index).data<size_type>() +
          src_lists_col.offset();
        auto const& src_child = src_lists_col.child(lists_column_view::child_column_index);
        for (auto read_idx  = src_list_offsets[src_list_id],
                  end_idx   = src_list_offsets[src_list_id + 1],
                  write_idx = dst_list_offsets[dst_list_id];
             read_idx < end_idx;
             ++read_idx, ++write_idx) {
          auto const is_valid = src_child.is_valid(read_idx);
          if (has_null_mask) { out_validities[write_idx] = static_cast<int8_t>(is_valid); }
          out_entries[write_idx] = is_valid ? src_child.element<T>(read_idx) : T{};
        }
      });

    if (has_null_mask) {
      auto [null_mask, null_count] = cudf::detail::valid_if(
        validities.begin(),
        validities.end(),
        [] __device__(auto const valid) { return valid; },
        stream,
        mr);
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
             bool,
             rmm::cuda_stream_view,
             rmm::mr::device_memory_resource*) const
  {
    // Currently, only support string_view and fixed-width types
    CUDF_FAIL("Called `copy_list_entries_fn()` on non-supported types.");
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

  if (input.num_columns() == 1) { return std::make_unique<column>(*(input.begin()), stream, mr); }
  if (input.num_rows() == 0) { return cudf::empty_like(input.column(0)); }

  // Generate offsets of the output lists column
  auto [list_offsets, list_validities] =
    generate_list_offsets_and_validities(input, has_null_mask, stream, mr);

  // Copy entries from the input lists columns to the output lists column - this needed to be
  // specialized for different types
  auto const num_output_lists = input.num_rows() * input.num_columns();
  auto const num_output_entries =
    cudf::detail::get_value<size_type>(list_offsets->view(), num_output_lists, stream);
  auto list_entries = type_dispatcher<dispatch_storage_type>(entry_type,
                                                             copy_list_entries_fn{},
                                                             input,
                                                             list_offsets->view(),
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
    list_validities.begin(),
    list_validities.end(),
    [] __device__(auto const valid) { return valid; },
    stream,
    mr);
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
