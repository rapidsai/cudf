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
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/detail/sorting.hpp>
#include <cudf/lists/drop_list_duplicates.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {
namespace {
template <typename Type>
struct has_negative_nans {
  column_device_view const d_entries;
  bool const has_nulls;

  __device__ Type operator()(size_type idx) const noexcept
  {
    if (has_nulls && d_entries.is_null_nocheck(idx)) { return false; }

    auto const val = d_entries.element<Type>(idx);
    return std::isnan(val) && std::signbit(val);  // std::signbit(x) == true if x is negative
  }
};

/**
 * @brief A structure to be used along with type_dispatcher to check if a
 * `column_view` has any negative NaN entry
 */
struct has_negative_nans_fn {
  template <typename Type, std::enable_if_t<cuda::std::is_floating_point_v<Type>>* = nullptr>
  bool operator()(column_view const& lists_entries, rmm::cuda_stream_view stream) const noexcept
  {
    auto const d_entries = column_device_view::create(lists_entries, stream);
    return thrust::count_if(rmm::exec_policy(stream),
                            thrust::make_counting_iterator(0),
                            thrust::make_counting_iterator(lists_entries.size()),
                            detail::has_negative_nans<Type>{*d_entries, lists_entries.has_nulls()});
  }

  template <typename Type, std::enable_if_t<not cuda::std::is_floating_point_v<Type>>* = nullptr>
  bool operator()(column_view const&, rmm::cuda_stream_view) const noexcept
  {
    // Columns of non floating-point data will never contain NaN
    return false;
  }
};

template <typename Type>
struct replace_negative_nans {
  __device__ Type operator()(Type val) const noexcept
  {
    return std::isnan(val) ? std::numeric_limits<Type>::quiet_NaN() : val;
  }
};

/**
 * @brief A structure to be used along with type_dispatcher to replace -NaN by NaN for all entries
 * of a floating-point data column
 */
struct replace_negative_nans_fn {
  template <typename Type, std::enable_if_t<not cuda::std::is_floating_point_v<Type>>* = nullptr>
  void operator()(column_view const&, mutable_column_view const&, rmm::cuda_stream_view) const
  {
    CUDF_FAIL("Cannot operate on a type that is not floating-point.");
  }

  template <typename Type, std::enable_if_t<cuda::std::is_floating_point_v<Type>>* = nullptr>
  void operator()(column_view const& lists_entries,
                  mutable_column_view const& new_entries,
                  rmm::cuda_stream_view stream) const noexcept
  {
    // Do not care whether an entry is null or not, just consider it as a floating-point value
    thrust::transform(rmm::exec_policy(stream),
                      lists_entries.begin<Type>(),
                      lists_entries.end<Type>(),
                      new_entries.begin<Type>(),
                      detail::replace_negative_nans<Type>{});
  }
};

/**
 * @brief Transform a given lists column to a new lists column in which all the list entries holding
 * -NaN value are replaced by (positive) NaN
 */
std::unique_ptr<column> replace_negative_nans_entries(column_view const& lists_entries,
                                                      lists_column_view const& lists_column,
                                                      rmm::cuda_stream_view stream)
{
  auto new_offsets = std::make_unique<column>(lists_column.offsets());
  auto new_entries = std::make_unique<column>(lists_entries);

  type_dispatcher(lists_entries.type(),
                  detail::replace_negative_nans_fn{},
                  lists_entries,
                  new_entries->mutable_view(),
                  stream);

  return make_lists_column(
    lists_column.size(),
    std::move(new_offsets),
    std::move(new_entries),
    lists_column.null_count(),
    cudf::detail::copy_bitmask(
      lists_column.parent(), stream, rmm::mr::get_current_device_resource()));
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
 * @brief Performs an equality comparison between two entries in a lists column
 *
 * For the two elements that are in the same list in the lists column, they will always be
 * considered as different. If they are from the same list and their type is one of floating
 * point types, this functor will return the same comparison result as
 * `cudf::element_equality_comparator`.
 *
 * For floating-point types, entries holding NaN value can be considered as different values or the
 * same value depending on the nans_equal parameter.
 *
 * @tparam Type       The data type of entries
 * @tparam nans_equal Flag to specify whether NaN entries should be considered as equal value (only
 * applicable for floating-point data column)
 */
template <class Type, bool nans_equal>
class list_entry_comparator {
 public:
  list_entry_comparator(offset_type const* list_offsets,
                        column_device_view d_view,
                        null_equality nulls_equal,
                        bool has_nulls)
    : list_offsets(list_offsets), d_view{d_view}, nulls_equal{nulls_equal}, has_nulls(has_nulls)
  {
  }

  template <bool nans_equal_ = nans_equal>
  std::enable_if_t<cuda::std::is_floating_point_v<Type> and nans_equal_, bool> __device__
  operator()(size_type i, size_type j) const noexcept
  {
    // Two entries are not considered for equality if they belong to different lists
    if (list_offsets[i] != list_offsets[j]) { return false; }

    if (has_nulls) {
      bool const nullable = d_view.nullable();
      bool const lhs_is_null{nullable and d_view.is_null_nocheck(i)};
      bool const rhs_is_null{nullable and d_view.is_null_nocheck(j)};
      if (lhs_is_null and rhs_is_null) {
        return nulls_equal == null_equality::EQUAL;
      } else if (lhs_is_null != rhs_is_null) {
        return false;
      }
    }

    // For floating-point types, if both element(i) and element(j) are NaNs then this comparison
    // will return `true`. This is the desired behavior in Pandas.
    auto const lhs = d_view.element<Type>(i);
    auto const rhs = d_view.element<Type>(j);
    if (std::isnan(lhs) and std::isnan(rhs)) { return true; }
    return lhs == rhs;
  }

  template <bool nans_equal_ = nans_equal>
  std::enable_if_t<not cuda::std::is_floating_point_v<Type> or not nans_equal_, bool> __device__
  operator()(size_type i, size_type j) const noexcept
  {
    // Two entries are not considered for equality if they belong to different lists
    if (list_offsets[i] != list_offsets[j]) { return false; }

    if (has_nulls) {
      bool const nullable = d_view.nullable();
      bool const lhs_is_null{nullable and d_view.is_null_nocheck(i)};
      bool const rhs_is_null{nullable and d_view.is_null_nocheck(j)};
      if (lhs_is_null and rhs_is_null) {
        return nulls_equal == null_equality::EQUAL;
      } else if (lhs_is_null != rhs_is_null) {
        return false;
      }
    }

    // For floating-point types, if both element(i) and element(j) are NaNs then this comparison
    // will return `false`. This is the desired behavior in Apache Spark.
    return d_view.element<Type>(i) == d_view.element<Type>(j);
  }

 private:
  offset_type const* list_offsets;
  column_device_view d_view;
  null_equality nulls_equal;
  bool has_nulls;
};

/**
 *  @brief Construct type-dispatched function object for copying indices of the list entries
 * ignoring duplicates
 */
struct get_unique_entries_fn {
  template <class Type, std::enable_if_t<not cudf::is_equality_comparable<Type, Type>()>* = nullptr>
  offset_type* operator()(offset_type const*,
                          column_device_view&,
                          size_type,
                          offset_type*,
                          null_equality,
                          nan_equality,
                          bool,
                          rmm::cuda_stream_view) const
  {
    CUDF_FAIL("Cannot operate on types that are not equally comparable.");
  }

  template <class Type, std::enable_if_t<cudf::is_equality_comparable<Type, Type>()>* = nullptr>
  offset_type* operator()(offset_type const* list_offsets,
                          column_device_view& d_view,
                          size_type num_entries,
                          offset_type* output_begin,
                          null_equality nulls_equal,
                          nan_equality nans_equal,
                          bool has_nulls,
                          rmm::cuda_stream_view stream) const noexcept
  {
    if (nans_equal == nan_equality::ALL_EQUAL) {
      list_entry_comparator<Type, true> const comp{list_offsets, d_view, nulls_equal, has_nulls};
      return thrust::unique_copy(rmm::exec_policy(stream),
                                 thrust::make_counting_iterator(0),
                                 thrust::make_counting_iterator(num_entries),
                                 output_begin,
                                 comp);
    } else {
      list_entry_comparator<Type, false> const comp{list_offsets, d_view, nulls_equal, has_nulls};
      return thrust::unique_copy(rmm::exec_policy(stream),
                                 thrust::make_counting_iterator(0),
                                 thrust::make_counting_iterator(num_entries),
                                 output_begin,
                                 comp);
    }
  }
};

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
 * @param nans_equal           Flag to specify whether NaN entries should be considered as equal
 * value (only applicable for floating-point data column)
 * @param stream               CUDA stream used for device memory operations and kernel launches
 * @param mr                   Device resource used to allocate memory
 *
 * @return A pair of columns, the first one contains unique list entries and the second one
 * contains their corresponding list offsets
 */
std::vector<std::unique_ptr<column>> get_unique_entries_and_list_offsets(
  column_view const& all_lists_entries,
  column_view const& entries_list_offsets,
  null_equality nulls_equal,
  nan_equality nans_equal,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto const num_entries    = all_lists_entries.size();
  auto const d_view_entries = column_device_view::create(all_lists_entries, stream);

  // Allocate memory to store the indices of the unique entries
  auto unique_indices     = rmm::device_uvector<offset_type>(num_entries, stream);
  auto const output_begin = unique_indices.begin();
  auto const output_end   = type_dispatcher(all_lists_entries.type(),
                                          get_unique_entries_fn{},
                                          entries_list_offsets.begin<offset_type>(),
                                          *d_view_entries,
                                          num_entries,
                                          output_begin,
                                          nulls_equal,
                                          nans_equal,
                                          all_lists_entries.has_nulls(),
                                          stream);

  // Collect unique entries and entry list offsets
  // The new null_count and bitmask of the unique entries will also be generated
  // by the gather function
  return cudf::detail::gather(table_view{{all_lists_entries, entries_list_offsets}},
                              output_begin,
                              output_end,
                              cudf::out_of_bounds_policy::DONT_CHECK,
                              stream,
                              mr)
    ->release();
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
                                             nan_equality nans_equal,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  if (lists_column.is_empty()) return cudf::empty_like(lists_column.parent());
  if (cudf::is_nested(lists_column.child().type())) {
    CUDF_FAIL("Nested types are not supported in drop_list_duplicates.");
  }

  // Flatten all entries (depth = 1) of the lists column
  auto const lists_entries = lists_column.get_sliced_child(stream);

  // sorted_lists will store the results of the original lists after calling segmented_sort
  auto const sorted_lists = [&]() {
    // If nans_equal == ALL_EQUAL and the column contains lists of floating-point data type,
    // we need to replace -NaN by NaN before sorting
    auto const replace_negative_nan =
      nans_equal == nan_equality::ALL_EQUAL and
      type_dispatcher(lists_entries.type(), detail::has_negative_nans_fn{}, lists_entries, stream);
    if (replace_negative_nan) {
      // The column new_lists_column is temporary, thus we will not pass in `mr`
      auto const new_lists_column =
        detail::replace_negative_nans_entries(lists_entries, lists_column, stream);
      return detail::sort_lists(
        lists_column_view(new_lists_column->view()), order::ASCENDING, null_order::AFTER, stream);
    } else {
      return detail::sort_lists(lists_column, order::ASCENDING, null_order::AFTER, stream);
    }
  }();

  auto const sorted_lists_entries =
    lists_column_view(sorted_lists->view()).get_sliced_child(stream);

  // Generate a 0-based offset column
  auto lists_offsets = detail::generate_clean_offsets(lists_column, stream, mr);

  // Generate a mapping from list entries to offsets of the lists containing those entries
  auto const entries_list_offsets =
    detail::generate_entry_list_offsets(sorted_lists_entries.size(), lists_offsets->view(), stream);

  // Copy non-duplicated entries (along with their list offsets) to new arrays
  auto unique_entries_and_list_offsets = detail::get_unique_entries_and_list_offsets(
    sorted_lists_entries, entries_list_offsets->view(), nulls_equal, nans_equal, stream, mr);

  // Generate offsets for the new lists column
  detail::generate_offsets(unique_entries_and_list_offsets.front()->size(),
                           unique_entries_and_list_offsets.back()->view(),
                           lists_offsets->mutable_view(),
                           stream);

  // Construct a new lists column without duplicated entries
  // Reuse the null_count and bitmask of the lists_column: those are the null information for
  // the list elements (rows)
  // For the entries of those lists (rows), their null_count and bitmask were generated separately
  // during the step `get_unique_entries_and_list_offsets` above
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
                                             nan_equality nans_equal,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::drop_list_duplicates(
    lists_column, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
