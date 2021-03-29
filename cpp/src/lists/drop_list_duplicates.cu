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
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/detail/sorting.hpp>
#include <cudf/lists/drop_list_duplicates.hpp>
#include <cudf/table/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {
namespace {
using offset_type = lists_column_view::offset_type;
/**
 * @brief Copy list entries and entry list offsets ignoring duplicates
 *
 * Given an array of all entries flattened from a list column and an array that maps each entry to
 * the offset of the list containing that entry, those entries and list offsets are copied into
 * new arrays such that the duplicated entries within each list will be ignored.
 *
 * @param all_lists_entries    The input array containing all list entries
 * @param entries_list_offsets A map from list entries to their corresponding list offsets
 * @param nulls_equal          Flag to specify whether null entries should be considered equal
 * @param stream               CUDA stream used for device memory operations and kernel launches
 * @param mr                   Device resource used to allocate memory
 *
 * @return A pair of columns, the first one contains unique list entries and the second one
 * contains their corresponding list offsets
 */
template <bool has_nulls>
std::vector<std::unique_ptr<column>> get_unique_entries_and_list_offsets(
  column_view const& all_lists_entries,
  column_view const& entries_list_offsets,
  null_equality nulls_equal,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  // Create an intermediate table, since the comparator only work on tables
  auto const device_input_table =
    cudf::table_device_view::create(table_view{{all_lists_entries}}, stream);
  auto const comp = row_equality_comparator<has_nulls>(
    *device_input_table, *device_input_table, nulls_equal == null_equality::EQUAL);

  auto const num_entries = all_lists_entries.size();
  // Allocate memory to store the indices of the unique entries
  auto const unique_indices = cudf::make_numeric_column(
    entries_list_offsets.type(), num_entries, mask_state::UNALLOCATED, stream);
  auto const unique_indices_begin = unique_indices->mutable_view().begin<offset_type>();

  auto const copy_end = thrust::unique_copy(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(num_entries),
    unique_indices_begin,
    [list_offsets = entries_list_offsets.begin<offset_type>(), comp] __device__(auto i, auto j) {
      return list_offsets[i] == list_offsets[j] && comp(i, j);
    });

  // Collect unique entries and entry list offsets
  auto const indices = cudf::detail::slice(
    unique_indices->view(), 0, thrust::distance(unique_indices_begin, copy_end));
  return cudf::detail::gather(table_view{{all_lists_entries, entries_list_offsets}},
                              indices,
                              cudf::out_of_bounds_policy::DONT_CHECK,
                              cudf::detail::negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr)
    ->release();
}

/**
 * @brief Generate a 0-based offset column for a lists column
 *
 * Given a lists_column_view, which may have a non-zero offset, generate a new column containing
 * 0-based list offsets. This is done by subtracting each of the input list offset by the first
 * offset.
 *
 * @code{.pseudo}
 * Given a list column having offsets = { 3, 7, 9, 13 },
 * then output_offsets = { 0, 4, 6, 10 }
 * @endcode
 *
 * @param lists_column The input lists column
 * @param stream       CUDA stream used for device memory operations and kernel launches
 * @param mr           Device resource used to allocate memory
 *
 * @return A column containing 0-based list offsets
 */
std::unique_ptr<column> generate_clean_offsets(lists_column_view const& lists_column,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  auto output_offsets = make_numeric_column(data_type{type_to_id<offset_type>()},
                                            lists_column.size() + 1,
                                            mask_state::UNALLOCATED,
                                            stream,
                                            mr);
  thrust::transform(
    rmm::exec_policy(stream),
    lists_column.offsets_begin(),
    lists_column.offsets_end(),
    output_offsets->mutable_view().begin<offset_type>(),
    [first = lists_column.offsets_begin()] __device__(auto offset) { return offset - *first; });
  return output_offsets;
}

/**
 * @brief Populate list offsets for all list entries
 *
 * Given an `offsets` column_view containing offsets of a lists column and a number of all list
 * entries in the column, generate an array that maps from each list entry to the offset of the list
 * containing that entry.
 *
 * @code{.pseudo}
 * num_entries = 10, offsets = { 0, 4, 6, 10 }
 * output = { 1, 1, 1, 1, 2, 2, 3, 3, 3, 3 }
 * @endcode
 *
 * @param num_entries The number of list entries
 * @param offsets     Column view to the list offsets
 * @param stream      CUDA stream used for device memory operations and kernel launches
 * @param mr          Device resource used to allocate memory
 *
 * @return A column containing entry list offsets
 */
std::unique_ptr<column> generate_entry_list_offsets(size_type num_entries,
                                                    column_view const& offsets,
                                                    rmm::cuda_stream_view stream)
{
  auto entry_list_offsets = make_numeric_column(offsets.type(),
                                                num_entries,
                                                mask_state::UNALLOCATED,
                                                stream,
                                                rmm::mr::get_current_device_resource());
  thrust::upper_bound(rmm::exec_policy(stream),
                      offsets.begin<offset_type>(),
                      offsets.end<offset_type>(),
                      thrust::make_counting_iterator<offset_type>(0),
                      thrust::make_counting_iterator<offset_type>(num_entries),
                      entry_list_offsets->mutable_view().begin<offset_type>());
  return entry_list_offsets;
}

/**
 * @brief Generate list offsets from entry offsets
 *
 * Generate an array of list offsets for the final result lists column. The list
 * offsets of the original lists column are also taken into account to make sure the result lists
 * column will have the same empty list rows (if any) as in the original lists column.
 *
 * @param[in] num_entries          The number of unique entries after removing duplicates
 * @param[in] entries_list_offsets The mapping from list entries to their list offsets
 * @param[out] original_offsets    The list offsets of the original lists column, which
 * will also be used to store the new list offsets
 * @param[in] stream               CUDA stream used for device memory operations and kernel launches
 * @param[in] mr                   Device resource used to allocate memory
 */
void generate_offsets(size_type num_entries,
                      column_view const& entries_list_offsets,
                      mutable_column_view const& original_offsets,
                      rmm::cuda_stream_view stream)
{
  // Firstly, generate temporary list offsets for the unique entries, ignoring empty lists (if any)
  // If entries_list_offsets = {1, 1, 1, 1, 2, 3, 3, 3, 4, 4 }, num_entries = 10,
  // then new_offsets = { 0, 4, 5, 8, 10 }
  auto const new_offsets = allocate_like(
    original_offsets, mask_allocation_policy::NEVER, rmm::mr::get_current_device_resource());
  thrust::copy_if(rmm::exec_policy(stream),
                  thrust::make_counting_iterator<offset_type>(0),
                  thrust::make_counting_iterator<offset_type>(num_entries + 1),
                  new_offsets->mutable_view().begin<offset_type>(),
                  [num_entries, offsets_ptr = entries_list_offsets.begin<offset_type>()] __device__(
                    auto i) -> bool {
                    return i == 0 || i == num_entries || offsets_ptr[i] != offsets_ptr[i - 1];
                  });

  // Generate a prefix sum of number of empty lists, storing inplace to the original lists
  // offsets
  // If the original list offsets is { 0, 0, 5, 5, 6, 6 } (there are 2 empty lists),
  // and new_offsets = { 0, 4, 6 },
  // then output = { 0, 1, 1, 2, 2, 3}
  auto const iter_trans_begin = cudf::detail::make_counting_transform_iterator(
    0, [offsets = original_offsets.begin<offset_type>()] __device__(auto i) {
      return (i > 0 && offsets[i] == offsets[i - 1]) ? 1 : 0;
    });
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         iter_trans_begin,
                         iter_trans_begin + original_offsets.size(),
                         original_offsets.begin<offset_type>());

  // Generate the final list offsets
  // If the original list offsets are { 0, 0, 5, 5, 6, 6 }, the new offsets are { 0, 4, 6 },
  //  and the prefix sums of empty lists are { 0, 1, 1, 2, 2, 3 },
  //  then output = { 0, 0, 4, 4, 5, 5 }
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<offset_type>(0),
                    thrust::make_counting_iterator<offset_type>(original_offsets.size()),
                    original_offsets.begin<offset_type>(),
                    [prefix_sum_empty_lists = original_offsets.begin<offset_type>(),
                     offsets = new_offsets->view().begin<offset_type>()] __device__(auto i) {
                      return offsets[i - prefix_sum_empty_lists[i]];
                    });
}
}  // anonymous namespace

/**
 * @copydoc cudf::lists::drop_list_duplicates
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<column> drop_list_duplicates(lists_column_view const& lists_column,
                                             null_equality nulls_equal,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  if (lists_column.is_empty()) return cudf::empty_like(lists_column.parent());
  if (cudf::is_nested(lists_column.child().type())) {
    CUDF_FAIL("Nested types are not supported in drop_list_duplicates.");
  }

  // Call segmented sort on the list elements and store them in a temporary column sorted_list
  auto const sorted_lists =
    detail::sort_lists(lists_column, order::ASCENDING, null_order::AFTER, stream);

  // Flatten all entries (depth = 1) of the lists column
  auto const all_lists_entries = lists_column_view(sorted_lists->view()).get_sliced_child(stream);

  // Generate a 0-based offset column
  auto lists_offsets = detail::generate_clean_offsets(lists_column, stream, mr);

  // Generate a mapping from list entries to offsets of the lists containing those entries
  auto const entries_list_offsets =
    detail::generate_entry_list_offsets(all_lists_entries.size(), lists_offsets->view(), stream);

  // Copy non-duplicated entries (along with their list offsets) to new arrays
  auto unique_entries_and_list_offsets =
    all_lists_entries.has_nulls()
      ? detail::get_unique_entries_and_list_offsets<true>(
          all_lists_entries, entries_list_offsets->view(), nulls_equal, stream, mr)
      : detail::get_unique_entries_and_list_offsets<false>(
          all_lists_entries, entries_list_offsets->view(), nulls_equal, stream, mr);

  // Generate offsets for the new lists column
  detail::generate_offsets(unique_entries_and_list_offsets.front()->size(),
                           unique_entries_and_list_offsets.back()->view(),
                           lists_offsets->mutable_view(),
                           stream);

  // Construct a new lists column without duplicated entries
  return make_lists_column(lists_column.size(),
                           std::move(lists_offsets),
                           std::move(unique_entries_and_list_offsets.front()),
                           lists_column.null_count(),
                           cudf::detail::copy_bitmask(lists_column.parent(), stream, mr));
}

}  // namespace detail

/**
 * @copydoc cudf::lists::drop_list_duplicates
 */
std::unique_ptr<column> drop_list_duplicates(lists_column_view const& lists_column,
                                             null_equality nulls_equal,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::drop_list_duplicates(lists_column, nulls_equal, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
