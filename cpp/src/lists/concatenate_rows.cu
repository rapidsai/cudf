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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/concatenate_rows.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/uninitialized_fill.h>

namespace cudf {
namespace lists {
namespace detail {

namespace {
/**
 * @brief A sentinel value used to mark that the entry count/size of a list has not been computed.
 */
constexpr auto uninitialized_size = std::numeric_limits<size_type>::lowest();

/**
 * @brief A sentinel value used to mark a null list element has been detected
 */
constexpr auto invalid_size = size_type{-1};

/**
 * @brief Count the number of entries for a list element at the row index `idx`.
 */
struct entry_count_fn {
  column_device_view const lists_dv;
  offset_type const* const list_offsets;
  size_type* const list_counts;
  concatenate_null_policy const null_policy;
  bool const nullable;

  entry_count_fn() = delete;

  __device__ void operator()(size_type idx) const noexcept
  {
    auto const current_count = list_counts[idx];

    // A null list has been assigned to this row, thus nothing to do anymore
    if (current_count == invalid_size) { return; }

    auto count = (current_count == uninitialized_size) ? size_type{0} : current_count;
    if (nullable and lists_dv.is_null_nocheck(idx)) {
      if (null_policy == concatenate_null_policy::NULLIFY_OUTPUT_ROW) { count = invalid_size; }
    } else {
      count += list_offsets[idx + 1] - list_offsets[idx];
    }
    list_counts[idx] = count;
  }
};

/**
 * @brief Struct used in type_dispatcher to count (and accumulate) the number of entries for
 * all list elements in the given lists column.
 */
struct lists_column_entry_count_fn {
  template <class T>
  __device__ void operator()(column_device_view lists_dv,
                             offset_type const* list_offsets,
                             size_type* list_counts,
                             concatenate_null_policy null_policy,
                             bool nullable,
                             rmm::cuda_stream_view stream) const noexcept
  {
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       lists_dv.size(),
                       entry_count_fn{lists_dv, list_offsets, list_counts, null_policy, nullable});
  }
};

/**
 * @brief For non fixed-width type, compute the size for each element and the total size (in bytes)
 * for the entire output list containing those elements.
 * // TODO: update doc
 */
template <class T_>
struct non_fixed_with_size_fn {
  column_device_view const lists_dv;
  column_device_view const child_dv;
  offset_type const* const list_offsets;
  size_type* const copying_offsets;
  size_type* const list_size_bytes;
  size_type* const entry_sizes;
  concatenate_null_policy const null_policy;
  bool const nullable;

  non_fixed_with_size_fn() = delete;

  template <class T = T_>
  std::enable_if_t<not std::is_same<T, cudf::string_view>::value, size_type> compute_list_size(
    size_type) const
  {
    // Currently, only support string_view
    CUDF_FAIL("Called `compute_list_size` on non-supported types.");
  }

  template <class T = T_>
  std::enable_if_t<std::is_same<T, cudf::string_view>::value, size_type> __device__
  compute_list_size(size_type idx) const noexcept
  {
    size_type dst_idx = copying_offsets[idx];  // offset of the string element, not the character
    auto list_size    = size_type{0};
    for (size_type src_idx = list_offsets[idx], idx_end = list_offsets[idx + 1]; src_idx < idx_end;
         ++dst_idx, ++src_idx) {
      if (not nullable or not child_dv.is_null_nocheck(src_idx)) {
        auto const str_size = child_dv.element<string_view>(src_idx).size_bytes();
        list_size += str_size;

        // Store entry size along the way (it is necessary to generate entry offsets)
        entry_sizes[dst_idx] = str_size;
      }
    }
    // Next time, we will store element size from this `dst_idx` value
    copying_offsets[idx] = dst_idx;

    return list_size;
  }

  template <class T>
  __device__ void operator()(size_type idx) const noexcept
  {
    auto const current_size = list_size_bytes[idx];

    // A null list has been assigned to this row, thus nothing to do anymore
    if (current_size == invalid_size) { return; }

    auto size = (current_size == uninitialized_size) ? size_type{0} : current_size;
    if (nullable and lists_dv.is_null_nocheck(idx)) {
      if (null_policy == concatenate_null_policy::NULLIFY_OUTPUT_ROW) { size = invalid_size; }
    } else {
      size += compute_list_size<T>(idx);
    }
    list_size_bytes[idx] = size;
  }
};

/**
 * @brief Struct used in type_dispatcher to compute the size for each non fixed-width element and
 * the total size (in bytes) for the entire output list containing those element. The list size is
 * necessary to compute offsets for copying entry.
 */
struct column_list_sizes_and_entry_sizes_fn {
  template <class T>
  std::enable_if_t<cudf::is_fixed_width<T>(), void> __device__ operator()(column_device_view,
                                                                          column_device_view,
                                                                          offset_type const*,
                                                                          size_type*,
                                                                          size_type*,
                                                                          size_type*,
                                                                          concatenate_null_policy,
                                                                          bool,
                                                                          rmm::cuda_stream_view)
  {
    // Currently, for non fixed-with types, only strings column are supported and need to be
    // computed. Other non fixed-with types (and nested types) are not supported.
  }

  template <class T>
  std::enable_if_t<not cudf::is_fixed_width<T>(), void> __device__
  operator()(column_device_view lists_dv,
             column_device_view child_dv,
             offset_type const* list_offsets,
             size_type* copying_offsets,
             size_type* list_size_bytes,
             size_type* entry_sizes,
             concatenate_null_policy null_policy,
             bool nullable,
             rmm::cuda_stream_view stream)
  {
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       lists_dv.size(),
                       non_fixed_with_size_fn<T>{lists_dv,
                                                 child_dv,
                                                 list_offsets,
                                                 copying_offsets,
                                                 list_size_bytes,
                                                 entry_sizes,
                                                 null_policy,
                                                 nullable});
  }
};

/**
 * @brief Struct used in type_dispatcher to create a child column for the result lists column
 */
struct create_child_column_fn {
  template <class T>
  std::enable_if_t<not cudf::is_fixed_width<T>() and not std::is_same<T, cudf::string_view>::value,
                   std::unique_ptr<column>>
  operator()(data_type,
             size_type,
             size_type,
             std::unique_ptr<column>&,
             rmm::cuda_stream_view,
             rmm::mr::device_memory_resource*) const
  {
    CUDF_FAIL("Called `create_child_column_fn()` on non supported type.");
  }

  template <class T>
  std::enable_if_t<cudf::is_fixed_width<T>(), std::unique_ptr<column>> operator()(
    data_type type,
    size_type num_rows,
    size_type,
    std::unique_ptr<column>&,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const noexcept
  {
    return std::make_unique<column>(type,
                                    num_rows,
                                    rmm::device_buffer(num_rows * size_of(type), stream, mr),
                                    rmm::device_buffer{},
                                    0);
  }

  template <class T>
  std::enable_if_t<std::is_same<T, cudf::string_view>::value, std::unique_ptr<column>> operator()(
    data_type,
    size_type num_rows,
    size_type total_size,
    std::unique_ptr<column>& child_offsets,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const noexcept
  {
    return make_strings_column(
      num_rows,
      std::move(child_offsets),
      strings::detail::create_chars_child_column(num_rows, 0, total_size, stream, mr),
      0,
      rmm::device_buffer{},
      stream,
      mr);
  }
};

/**
 * @brief Copy a list onto the destination lists column at the given offset position.
 */
template <class T_>
struct copy_list_fn {
  column_device_view const src_lists_dv;      ///< The source lists column to copy list from
  column_device_view const src_entries_dv;    ///< The source child column to copy entry from
  offset_type const* const src_list_offsets;  ///< The list offsets of the source list column
  mutable_column_device_view dst_entries_dv;  ///< The target child column to copy entry to
  offset_type* const copying_offsets;  ///< The offsets of the destination to copy list entry to
  // offset_type* const dst_entry_offsets;       ///< The entry offsets of the target child column
  // ///<  (only apply for string type)
  bool const nullable;

  copy_list_fn() = delete;

  template <class T = T_>
  std::enable_if_t<not cudf::is_fixed_width<T>() and not std::is_same<T, cudf::string_view>::value,
                   void>
    copy_entry_and_advance(size_type, size_type) const
  {
    CUDF_FAIL("Called `copy_entry_and_advance()` on none-supported data type.");
  }

  template <class T = T_>
  std::enable_if_t<cudf::is_fixed_width<T>(), void> __device__
  copy_entry_and_advance(size_type& dst_idx, size_type src_idx) const noexcept
  {
    // For fixed-with type, just copy one entry at a time and advance to the next entry
    dst_entries_dv.element<T>(dst_idx++) = src_entries_dv.element<T>(src_idx);
  }

  template <class T = T_>
  std::enable_if_t<std::is_same<T, cudf::string_view>::value, void> __device__
  copy_entry_and_advance(size_type& dst_idx, size_type src_idx) const noexcept
  {
    // For strings, `src_idx` is the string_view index of the source string, while `dst_idx` is the
    // character index of the first character of the copying string entry. This is because we can
    // only copy string to the char* pointer of the target column.

    auto const d_str = src_entries_dv.element<string_view>(src_idx);
    cudf::strings::detail::copy_string(dst_entries_dv.data<char>()[dst_idx], d_str);
    dst_idx += d_str.size_bytes();  // advance to the past-end character of the copying string
  }

  template <class T = T_>
  void __device__ operator()(size_type out_idx) const noexcept
  {
    // The row at index `out_idx` has been determined to be a null
    auto dst_idx = copying_offsets[out_idx];
    if (dst_idx == invalid_size) { return; }

    // The source list column has null at index `out_idx`
    if (nullable and src_lists_dv.is_null_nocheck(out_idx)) { return; }

    for (size_type src_idx = src_list_offsets[out_idx], idx_end = src_list_offsets[out_idx + 1];
         src_idx < idx_end;
         ++src_idx) {
      if (not nullable or not src_entries_dv.is_null_nocheck(src_idx)) {
        copy_entry_and_advance<T>(dst_idx, src_idx);
      }
    }

    // Next time when we call `copy_list_fn()`, the list will be copied to this `dst_idx` position
    copying_offsets[out_idx] = dst_idx;
  }
};

/**
 * Struct used in type_dispatcher to concatenate lists columns
 */
struct scatter_lists_column_fn {
  template <class T>
  __device__ void operator()(column_device_view src_lists_dv,
                             column_device_view src_entries_dv,
                             size_type const* src_list_offsets,
                             mutable_column_device_view dst_entries_dv,
                             size_type* copying_offsets,
                             bool nullable,
                             rmm::cuda_stream_view stream)
  {
    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<size_type>(0),
      src_lists_dv.size(),
      copy_list_fn<T>{
        src_lists_dv, src_entries_dv, src_list_offsets, dst_entries_dv, copying_offsets, nullable});
  }
};

}  // namespace

std::unique_ptr<column> concatenate_rows(
  table_view const& lists_columns,
  concatenate_null_policy null_policy,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  CUDF_EXPECTS(lists_columns.num_columns() > 0, "The input table must have at least one column.");

  auto const entry_type = lists_column_view(*lists_columns.begin()).child().type();
  for (auto const& col : lists_columns) {
    CUDF_EXPECTS(col.type().id() == type_id::LIST,
                 "All columns of the input table must be of lists column type.");

    auto const child_col = lists_column_view(col).child();
    CUDF_EXPECTS(not cudf::is_nested(child_col.type()), "Nested types are not supported.");
    CUDF_EXPECTS(entry_type.id() == child_col.type().id(),
                 "The types of entries in the input columns must be the same.");
  }

  // Single column returns a copy
  if (lists_columns.num_columns() == 1) {
    return std::make_unique<column>(*(lists_columns.begin()), stream, mr);
  }

  auto const num_rows = lists_columns.num_rows();
  if (num_rows == 0) { return cudf::empty_like(lists_columns.column(0)); }

  // Prepare data to process in device
  std::vector<std::tuple<
    std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>,  // the original
                                                                                    // lists column
    std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>,  // child column
                                                                                    // of the lists
                                                                                    // column
    offset_type const*,  // offsets of the lists column
    bool>>               // whether the original lists column is nullable
    lists_cols_dv;
  for (auto const& col : lists_columns) {
    lists_cols_dv.emplace_back(column_device_view::create(col, stream),
                               column_device_view::create(lists_column_view(col).child(), stream),
                               lists_column_view(col).offsets_begin(),
                               col.nullable());
  }

  static_assert(sizeof(offset_type) == sizeof(int32_t));
  static_assert(sizeof(size_type) == sizeof(int32_t));

  // Offsets of the output lists column
  // This offset column is also used to store list counts
  auto list_offsets = make_numeric_column(
    data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);
  auto const output_offsets_ptr = list_offsets->mutable_view().begin<offset_type>();

  // The offset values are initialized by `uninitialized_size` value before computation
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), output_offsets_ptr, output_offsets_ptr + 1, 0);
  thrust::uninitialized_fill(rmm::exec_policy(stream),
                             output_offsets_ptr + 1,
                             output_offsets_ptr + num_rows + 1,
                             uninitialized_size);

  // A temporary array to store lists sizes (in bytes) of the non fixed-width type
  auto child_offsets =
    cudf::is_fixed_width(entry_type)
      ? nullptr
      : make_numeric_column(
          data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);
  auto list_size_bytes = child_offsets->mutable_view();

  for (auto const& [lists_dv_ptr, child_dv_ptr, offsets_ptr, nullable] : lists_cols_dv) {
    // Accumulate the lists' entry counts from the 2nd position
    type_dispatcher(entry_type,
                    lists_column_entry_count_fn{},
                    *lists_dv_ptr,
                    offsets_ptr,
                    output_offsets_ptr + 1,
                    null_policy,
                    nullable,
                    stream);

    // Accumulate list sizes(only apply to non fixed-with types)
    type_dispatcher(entry_type,
                    column_list_sizes_and_entry_sizes_fn{},
                    *lists_dv_ptr,
                    *child_dv_ptr,
                    offsets_ptr,
                    copying_offsets.begin(),
                    list_size_bytes.begin<size_type>() + 1,
                    entry_sizes,
                    null_policy,
                    nullable,
                    stream);
  }

  auto const count_it = thrust::make_counting_iterator<size_type>(0);

  // Use the list count array to compute null_mask and null_count of the output column
  auto [null_mask, null_count] = cudf::detail::valid_if(
    count_it,
    count_it + num_rows,
    [str_sizes = output_offsets_ptr + 1] __device__(size_type idx) {
      return str_sizes[idx] != invalid_size;
    },
    stream,
    mr);

  // Build the list offsets from list counts
  auto const iter_trans_begin = thrust::make_transform_iterator(
    output_offsets_ptr + 1,
    [] __device__(auto const size) { return size != invalid_size ? size : 0; });

  // output_offsets.set_element_async(0, 0, stream);
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         iter_trans_begin + 1,
                         iter_trans_begin + num_rows,
                         output_offsets_ptr + 1);

  // A temporary array to store offsets for copying data, initialized with `invalid_size` value
  auto copying_offsets = rmm::device_uvector<offset_type>(num_rows, stream);
  //  thrust::uninitialized_fill(
  //    rmm::exec_policy(stream), copying_offsets.begin(), copying_offsets.end(), invalid_size);
  if (cudf::is_fixed_width(entry_type)) {
    thrust::copy(rmm::exec_policy(stream),
                 output_offsets_ptr,
                 output_offsets_ptr + num_rows,
                 copying_offsets.begin());
  } else {
    thrust::uninitialized_fill(rmm::exec_policy(stream),
                               list_size_bytes.begin<size_type>(),
                               list_size_bytes.begin<size_type>() + 1,
                               0);
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           list_size_bytes.begin<size_type>() + 1,
                           list_size_bytes.end<size_type>(),
                           list_size_bytes.begin<size_type>());

    // remove list_size_bytes, and use copying_offset to store it
    thrust::copy(rmm::exec_policy(stream),
                 list_size_bytes.begin<size_type>(),
                 list_size_bytes.end<size_type>(),
                 copying_offsets.begin());
  }

  // total size is the last entry
  // Note this call does a synchronize on the stream and thereby also protects the
  // set_element_async parameter from going out of scope before it is used.
  // Note: Calling to `get_value` does a stream synchronization
  auto const total_size =
    cudf::is_fixed_width(entry_type)
      ? cudf::detail::get_value<size_type>(list_offsets->view(), num_rows, stream)
      : cudf::detail::get_value<size_type>(child_offsets->view(), num_rows, stream);

  // child_offsets are created during computing entry size
  // todo: add compting entry size

  // Child column of the output lists column
  auto list_entries = type_dispatcher(entry_type,
                                      create_child_column_fn{},
                                      entry_type,
                                      num_rows,
                                      total_size,
                                      child_offsets,
                                      stream,
                                      mr);
  //    std::make_unique<column>(entry_type,
  //                             total_size,
  //                             rmm::device_buffer(total_size * size_of(entry_type), stream, mr),
  //                             detail::create_null_mask(size, allocate_mask, stream, mr),
  //                             state_null_count(allocate_mask, input.size()),
  //                             std::move(children));

  auto const entries_dv_ptr = mutable_column_device_view::create(list_entries->mutable_view());

  // Generate the offsets for the child column (only apply for string)

  // Create a temporary array to store the offsets of copying entries
  // The entries of each resulting list will be copied to the position starting from the list offset
  //  rmm::device_uvector<offset_type> offsets(num_rows, stream);

  // Scatter each lists column onto the resulting lists column, using the `offsets` array as the
  // scattering positions
  for (auto const& [lists_dv_ptr, child_dv_ptr, offsets_ptr, nullable] : lists_cols_dv) {
    type_dispatcher(entry_type,
                    scatter_lists_column_fn{},
                    *lists_dv_ptr,
                    *child_dv_ptr,
                    offsets_ptr,
                    *entries_dv_ptr,
                    copying_offsets.begin(),
                    nullable,
                    stream);
  }

  // Generate null_mask and null_count for the list entries
  //  list_entries->set_null_mask(std::move(null_mask), null_count);

  return make_lists_column(num_rows,
                           std::move(list_offsets),
                           std::move(list_entries),
                           null_count,
                           null_count ? std::move(null_mask) : rmm::device_buffer{},
                           stream,
                           mr);
}

}  // namespace detail

/**
 * @copydoc cudf::lists::concatenate_rows
 */
std::unique_ptr<column> concatenate_rows(table_view const& lists_columns,
                                         concatenate_null_policy null_policy,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate_rows(lists_columns, null_policy, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
