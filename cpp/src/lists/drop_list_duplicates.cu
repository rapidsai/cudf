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

#include <stream_compaction/drop_duplicates.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/lists/drop_list_duplicates.hpp>
#include <cudf/structs/struct_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

namespace cudf {
namespace lists {
namespace detail {
namespace {
template <typename Type>
struct has_negative_nans_fn {
  column_device_view const d_entries;
  bool const has_nulls;

  has_negative_nans_fn(column_device_view const& d_entries, bool const has_nulls)
    : d_entries(d_entries), has_nulls(has_nulls)
  {
  }

  __device__ Type operator()(size_type idx) const noexcept
  {
    if (has_nulls && d_entries.is_null_nocheck(idx)) { return false; }

    auto const val = d_entries.element<Type>(idx);
    return std::isnan(val) && std::signbit(val);  // std::signbit(x) == true if x is negative
  }
};

/**
 * @brief A structure to be used along with type_dispatcher to check if a column has any
 * negative NaN value.
 *
 * This functor is used to check for replacing negative NaN if there exists one. It is neccessary
 * because when calling to `lists::detail::sort_lists`, the negative NaN and positive NaN values (if
 * both exist) are separated to the two ends of the output column. This is due to the API
 * `lists::detail::sort_lists` internally calls `cub::DeviceSegmentedRadixSort`, which performs
 * sorting by comparing bits of the input numbers. Since negative and positive NaN have
 * different bits representation, they may not be moved to be close to each other after sorted.
 */
struct has_negative_nans_dispatch {
  template <typename Type, std::enable_if_t<cuda::std::is_floating_point_v<Type>>* = nullptr>
  bool operator()(column_view const& lists_entries, rmm::cuda_stream_view stream) const noexcept
  {
    auto const d_entries = column_device_view::create(lists_entries, stream);
    return thrust::count_if(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(lists_entries.size()),
      detail::has_negative_nans_fn<Type>{*d_entries, lists_entries.has_nulls()});
  }

  template <typename Type, std::enable_if_t<std::is_same_v<Type, cudf::struct_view>>* = nullptr>
  bool operator()(column_view const& lists_entries, rmm::cuda_stream_view stream) const
  {
    // Recursively check negative NaN on the children columns.
    return std::any_of(
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(lists_entries.num_children()),
      [structs_view = structs_column_view{lists_entries}, stream](auto const child_idx) {
        auto const col = structs_view.get_sliced_child(child_idx);
        return type_dispatcher(col.type(), detail::has_negative_nans_dispatch{}, col, stream);
      });
  }

  template <typename Type,
            std::enable_if_t<!cuda::std::is_floating_point_v<Type> &&
                             !std::is_same_v<Type, cudf::struct_view>>* = nullptr>
  bool operator()(column_view const&, rmm::cuda_stream_view) const
  {
    // Columns of non floating-point data will never contain NaN.
    return false;
  }
};

template <typename Type>
struct replace_negative_nans_fn {
  __device__ Type operator()(Type val) const noexcept
  {
    return std::isnan(val) ? std::numeric_limits<Type>::quiet_NaN() : val;
  }
};

/**
 * @brief A structure to be used along with type_dispatcher to replace -NaN by NaN for all rows
 * in a floating-point data column.
 *
 * Replacing -NaN by NaN is necessary before sorting (individual) lists because the sorting API may
 * separate -NaN and NaN to the two ends of each result list while we want to group all NaN
 * together.
 */
struct replace_negative_nans_dispatch {
  template <typename Type,
            std::enable_if_t<!cuda::std::is_floating_point_v<Type> &&
                             !std::is_same_v<Type, cudf::struct_view>>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& lists_entries,
                                     rmm::cuda_stream_view) const noexcept
  {
    // For non floating point type and non struct, just return a copy of the input.
    return std::make_unique<column>(lists_entries);
  }

  template <typename Type, std::enable_if_t<cuda::std::is_floating_point_v<Type>>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& lists_entries,
                                     rmm::cuda_stream_view stream) const noexcept
  {
    auto new_entries = cudf::detail::allocate_like(
      lists_entries, lists_entries.size(), cudf::mask_allocation_policy::NEVER, stream);
    new_entries->set_null_mask(cudf::detail::copy_bitmask(lists_entries, stream),
                               lists_entries.null_count());

    // Replace all negative NaN values.
    thrust::transform(rmm::exec_policy(stream),
                      lists_entries.template begin<Type>(),
                      lists_entries.template end<Type>(),
                      new_entries->mutable_view().template begin<Type>(),
                      detail::replace_negative_nans_fn<Type>{});

    return new_entries;
  }

  template <typename Type, std::enable_if_t<std::is_same_v<Type, cudf::struct_view>>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& lists_entries,
                                     rmm::cuda_stream_view stream) const noexcept
  {
    std::vector<std::unique_ptr<cudf::column>> output_struct_members;
    std::transform(
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(lists_entries.num_children()),
      std::back_inserter(output_struct_members),
      [structs_view = structs_column_view{lists_entries}, stream](auto const child_idx) {
        auto const col = structs_view.get_sliced_child(child_idx);
        return type_dispatcher(col.type(), detail::replace_negative_nans_dispatch{}, col, stream);
      });

    return cudf::make_structs_column(lists_entries.size(),
                                     std::move(output_struct_members),
                                     lists_entries.null_count(),
                                     cudf::detail::copy_bitmask(lists_entries, stream),
                                     stream);
  }
};

/**
 * @brief Generate a 0-based offset column for a lists column.
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
 * @param lists_column The input lists column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return An array containing 0-based list offsets.
 */
rmm::device_uvector<offset_type> generate_clean_offsets(lists_column_view const& lists_column,
                                                        rmm::cuda_stream_view stream)
{
  auto output_offsets = rmm::device_uvector<offset_type>(lists_column.size() + 1, stream);
  thrust::transform(
    rmm::exec_policy(stream),
    lists_column.offsets_begin(),
    lists_column.offsets_end(),
    output_offsets.begin(),
    [first = lists_column.offsets_begin()] __device__(auto offset) { return offset - *first; });
  return output_offsets;
}

/**
 * @brief Populate list offsets for all list entries.
 *
 * Given an array of a list offsets and a number of all list entries in the column,
 * generate an array that maps from each list entry to a 1-based index of the list containing
 * that entry.
 *
 * @code{.pseudo}
 * num_entries = 10, offsets = { 0, 4, 6, 10 }
 * output = { 1, 1, 1, 1, 2, 2, 3, 3, 3, 3 }
 * @endcode
 *
 * @param num_entries The number of list entries.
 * @param offsets Array containing the list offsets.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return An array containing 1-based list indices corresponding to each list entry.
 */
rmm::device_uvector<offset_type> generate_entry_list_indices(
  size_type num_entries,
  rmm::device_uvector<offset_type> const& offsets,
  rmm::cuda_stream_view stream)
{
  auto entry_list_indices = rmm::device_uvector<offset_type>(num_entries, stream);
  thrust::upper_bound(rmm::exec_policy(stream),
                      offsets.begin(),
                      offsets.end(),
                      thrust::make_counting_iterator<offset_type>(0),
                      thrust::make_counting_iterator<offset_type>(num_entries),
                      entry_list_indices.begin());
  return entry_list_indices;
}

/**
 * @brief Performs an equality comparison between two entries in a lists column.
 *
 * For the two entries that are NOT in the same list, they will always be considered as different.
 * If they are from the same list and their type is not floating point, this functor will return the
 * same comparison result as `cudf::element_equality_comparator`.
 *
 * For floating-point types, entries holding NaN value can be considered as different or the same
 * value depending on the `nans_equal` parameter.
 */
template <class Type>
struct column_row_comparator_fn {
  offset_type const* const list_indices;
  column_device_view const lhs;
  column_device_view const rhs;
  null_equality const nulls_equal;
  bool const has_nulls;
  bool const nans_equal;

  __host__ __device__ column_row_comparator_fn(offset_type const* const list_indices,
                                               column_device_view const& lhs,
                                               column_device_view const& rhs,
                                               null_equality const nulls_equal,
                                               bool const has_nulls,
                                               bool const nans_equal)
    : list_indices(list_indices),
      lhs(lhs),
      rhs(rhs),
      nulls_equal(nulls_equal),
      has_nulls(has_nulls),
      nans_equal(nans_equal)
  {
  }

  template <typename T = Type, std::enable_if_t<!cuda::std::is_floating_point_v<T>>* = nullptr>
  bool __device__ compare(T const& lhs_val, T const& rhs_val) const noexcept
  {
    return lhs_val == rhs_val;
  }

  template <typename T = Type, std::enable_if_t<cuda::std::is_floating_point_v<T>>* = nullptr>
  bool __device__ compare(T const& lhs_val, T const& rhs_val) const noexcept
  {
    // If both element(i) and element(j) are NaNs and nans are considered as equal value then this
    // comparison will return `true`. This is the desired behavior in Pandas.
    if (nans_equal && std::isnan(lhs_val) && std::isnan(rhs_val)) { return true; }

    // If nans are considered as NOT equal, even both element(i) and element(j) are NaNs this
    // comparison will still return `false`. This is the desired behavior in Apache Spark.
    return lhs_val == rhs_val;
  }

  bool __device__ operator()(size_type i, size_type j) const noexcept
  {
    // Two entries are not considered for equality if they belong to different lists.
    if (list_indices[i] != list_indices[j]) { return false; }

    if (has_nulls) {
      bool const lhs_is_null{lhs.nullable() && lhs.is_null_nocheck(i)};
      bool const rhs_is_null{rhs.nullable() && rhs.is_null_nocheck(j)};
      if (lhs_is_null && rhs_is_null) {
        return nulls_equal == null_equality::EQUAL;
      } else if (lhs_is_null != rhs_is_null) {
        return false;
      }
    }

    return compare(lhs.element<Type>(i), lhs.element<Type>(j));
  }
};

/**
 * @brief Struct used in type_dispatcher for comparing two entries in a lists column.
 */
struct column_row_comparator_dispatch {
  offset_type const* const list_indices;
  column_device_view const lhs;
  column_device_view const rhs;
  null_equality const nulls_equal;
  bool const has_nulls;
  bool const nans_equal;

  __device__ column_row_comparator_dispatch(offset_type const* const list_indices,
                                            column_device_view const& lhs,
                                            column_device_view const& rhs,
                                            null_equality const nulls_equal,
                                            bool const has_nulls,
                                            bool const nans_equal)
    : list_indices(list_indices),
      lhs(lhs),
      rhs(rhs),
      nulls_equal(nulls_equal),
      has_nulls(has_nulls),
      nans_equal(nans_equal)
  {
  }

  template <class Type, std::enable_if_t<cudf::is_equality_comparable<Type, Type>()>* = nullptr>
  bool __device__ operator()(size_type i, size_type j) const noexcept
  {
    return column_row_comparator_fn<Type>{
      list_indices, lhs, rhs, nulls_equal, has_nulls, nans_equal}(i, j);
  }

  template <class Type, std::enable_if_t<!cudf::is_equality_comparable<Type, Type>()>* = nullptr>
  bool operator()(size_type, size_type) const
  {
    CUDF_FAIL(
      "column_row_comparator_dispatch cannot operate on types that are not equally comparable.");
  }
};

/**
 * @brief Performs an equality comparison between rows of two tables using `column_row_comparator`
 * to compare rows of their corresponding columns.
 */
struct table_row_comparator_fn {
  offset_type const* const list_indices;
  table_device_view const lhs;
  table_device_view const rhs;
  null_equality const nulls_equal;
  bool const has_nulls;
  bool const nans_equal;

  table_row_comparator_fn(offset_type const* const list_indices,
                          table_device_view const& lhs,
                          table_device_view const& rhs,
                          null_equality const nulls_equal,
                          bool const has_nulls,
                          bool const nans_equal)
    : list_indices(list_indices),
      lhs(lhs),
      rhs(rhs),
      nulls_equal(nulls_equal),
      has_nulls(has_nulls),
      nans_equal(nans_equal)
  {
  }

  bool __device__ operator()(size_type i, size_type j) const noexcept
  {
    auto column_comp = [=](column_device_view const& lhs, column_device_view const& rhs) {
      return type_dispatcher(
        lhs.type(),
        column_row_comparator_dispatch{list_indices, lhs, rhs, nulls_equal, has_nulls, nans_equal},
        i,
        j);
    };

    return thrust::equal(thrust::seq, lhs.begin(), lhs.end(), rhs.begin(), column_comp);
  }
};

/**
 *  @brief Struct used in type_dispatcher for copying indices of the list entries ignoring
 * duplicates.
 */
struct get_unique_entries_dispatch {
  template <class Type,
            std::enable_if_t<!cudf::is_equality_comparable<Type, Type>() &&
                             !std::is_same_v<Type, cudf::struct_view>>* = nullptr>
  offset_type* operator()(offset_type const*,
                          column_view const&,
                          size_type,
                          offset_type*,
                          null_equality,
                          nan_equality,
                          bool,
                          duplicate_keep_option,
                          rmm::cuda_stream_view) const
  {
    CUDF_FAIL(
      "get_unique_entries_dispatch cannot operate on types that are not equally comparable or not "
      "STRUCT type.");
  }

  template <class Type, std::enable_if_t<cudf::is_equality_comparable<Type, Type>()>* = nullptr>
  offset_type* operator()(offset_type const* list_indices,
                          column_view const& all_lists_entries,
                          size_type num_entries,
                          offset_type* output_begin,
                          null_equality nulls_equal,
                          nan_equality nans_equal,
                          bool has_nulls,
                          duplicate_keep_option keep_option,
                          rmm::cuda_stream_view stream) const noexcept
  {
    auto const d_view = column_device_view::create(all_lists_entries, stream);
    auto const comp   = column_row_comparator_fn<Type>{list_indices,
                                                     *d_view,
                                                     *d_view,
                                                     nulls_equal,
                                                     has_nulls,
                                                     nans_equal == nan_equality::ALL_EQUAL};
    return cudf::detail::unique_copy(thrust::make_counting_iterator(0),
                                     thrust::make_counting_iterator(num_entries),
                                     output_begin,
                                     comp,
                                     keep_option,
                                     stream);
  }

  template <class Type, std::enable_if_t<std::is_same_v<Type, cudf::struct_view>>* = nullptr>
  offset_type* operator()(offset_type const* list_indices,
                          column_view const& all_lists_entries,
                          size_type num_entries,
                          offset_type* output_begin,
                          null_equality nulls_equal,
                          nan_equality nans_equal,
                          bool has_nulls,
                          duplicate_keep_option keep_option,
                          rmm::cuda_stream_view stream) const noexcept
  {
    auto const flattened_entries = cudf::structs::detail::flatten_nested_columns(
      table_view{{all_lists_entries}}, {order::ASCENDING}, {null_order::AFTER}, {});
    auto const dview_ptr = table_device_view::create(flattened_entries, stream);

    auto const comp = table_row_comparator_fn{list_indices,
                                              *dview_ptr,
                                              *dview_ptr,
                                              nulls_equal,
                                              has_nulls,
                                              nans_equal == nan_equality::ALL_EQUAL};
    return cudf::detail::unique_copy(thrust::make_counting_iterator(0),
                                     thrust::make_counting_iterator(num_entries),
                                     output_begin,
                                     comp,
                                     keep_option,
                                     stream);
  }
};

/**
 * @brief Extract list entries and their corresponding list indices ignoring duplicates.
 */
std::vector<std::unique_ptr<column>> get_unique_entries_and_list_indices(
  column_view const& keys_entries,
  std::optional<column_view> const& values_entries,
  rmm::device_uvector<offset_type> const& entries_list_indices,
  null_equality nulls_equal,
  nan_equality nans_equal,
  duplicate_keep_option keep_option,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto const num_entries = keys_entries.size();

  // Allocate memory to store the indices of the unique entries.
  auto unique_indices     = rmm::device_uvector<offset_type>(num_entries, stream);
  auto const output_begin = unique_indices.begin();
  auto const output_end   = type_dispatcher(keys_entries.type(),
                                          get_unique_entries_dispatch{},
                                          entries_list_indices.begin(),
                                          keys_entries,
                                          num_entries,
                                          output_begin,
                                          nulls_equal,
                                          nans_equal,
                                          keys_entries.has_nulls(),
                                          keep_option,
                                          stream);

  // The indices of the unique entries will be used as a gather map to collect these entries.
  auto const gather_map =
    column_view(data_type{type_to_id<offset_type>()},
                static_cast<size_type>(thrust::distance(output_begin, output_end)),
                unique_indices.data());

  auto const list_indices_view = column_view(data_type{type_to_id<offset_type>()},
                                             static_cast<size_type>(entries_list_indices.size()),
                                             entries_list_indices.data());
  auto const input_table       = values_entries.has_value()
                                   ? table_view{{keys_entries, values_entries.value(), list_indices_view}}
                                   : table_view{{keys_entries, list_indices_view}};

  // Collect unique entries and entry list indices.
  // The new null_count and bitmask of the unique entries will also be generated by the gather
  // function.
  return cudf::detail::gather(input_table,
                              gather_map,
                              cudf::out_of_bounds_policy::DONT_CHECK,
                              cudf::detail::negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr)
    ->release();
}

/**
 * @brief Generate list offsets from entry indices.
 *
 * Generate a column of list offsets for the final result lists column(s). The list offsets of the
 * original lists column are also taken into account to make sure the result lists column will have
 * the same empty list rows (if any) as in the original lists column.
 *
 * @param num_entries The number of unique entries after removing duplicates.
 * @param entries_list_indices The mapping from list entries to their list indices.
 * @param original_offsets The list offsets of the original lists column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device resource used to allocate memory.
 */
std::unique_ptr<column> generate_output_offsets(
  size_type num_entries,
  column_view const& entries_list_indices,
  rmm::device_uvector<offset_type> const& original_offsets,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  // Let consider an example:
  // Given the original offsets of the input lists column is [0, 4, 5, 6, 7, 10, 11, 13].
  // The original entries_list_indices is [1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 6, 7, 7], and after
  // copying unique entries we have the entries_list_indices becomes [1, 1, 1, 4, 5, 5, 5, 7, 7] and
  // num_entries is 9.
  //
  // That means, one entry in the list index 1 has been removed (first list, as we are using 1-based
  // list index), and entries in the lists with indices {3, 3, 6} have been removed completely.

  // Store the (1-based) list indices of the unique entries, only one index value per non-empty
  // list. Given the example above, we will have this array contains the values [1, 4, 5, 7]
  auto list_indices = rmm::device_uvector<offset_type>(original_offsets.size() - 1, stream);

  // Store the non-zero numbers of unique entries per list.
  // Given the example above, we will have this array contains the values [3, 1, 3, 2]
  auto list_sizes = rmm::device_uvector<offset_type>(original_offsets.size() - 1, stream);

  // Count the numbers of unique entries for each non-empty list.
  auto const end                 = thrust::reduce_by_key(rmm::exec_policy(stream),
                                         entries_list_indices.template begin<offset_type>(),
                                         entries_list_indices.template end<offset_type>(),
                                         thrust::make_constant_iterator<offset_type>(1),
                                         list_indices.begin(),
                                         list_sizes.begin());
  auto const num_non_empty_lists = thrust::distance(list_indices.begin(), end.first);

  // The output offsets for the output lists column(s).
  auto new_offsets         = make_numeric_column(data_type{type_to_id<offset_type>()},
                                         original_offsets.size(),
                                         mask_state::UNALLOCATED,
                                         stream,
                                         mr);
  auto const d_new_offsets = new_offsets->mutable_view().template begin<offset_type>();

  // The new offsets need to be filled with 0 value.
  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), d_new_offsets, original_offsets.size(), offset_type{0});

  // Scatter non-zero sizes of the output lists into the correct positions.
  // Given the example above, we will have new_offsets = [0, 3, 0, 0, 1, 3, 0, 2]
  thrust::scatter(rmm::exec_policy(stream),
                  list_sizes.begin(),
                  list_sizes.begin() + num_non_empty_lists,
                  list_indices.begin(),
                  d_new_offsets);

  // Generate offsets from sizes.
  // Given the example above, we will have new_offsets = [0, 3, 3, 3, 4, 7, 7, 9]
  thrust::inclusive_scan(
    rmm::exec_policy(stream), d_new_offsets, d_new_offsets + new_offsets->size(), d_new_offsets);

  // Done. Hope that your head didn't explode after reading till this point.
  return new_offsets;
}

/**
 * @brief Common execution code called by all public `drop_list_duplicates` APIs.
 */
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> drop_list_duplicates_common(
  lists_column_view const& keys,
  std::optional<lists_column_view> const& values,
  null_equality nulls_equal,
  nan_equality nans_equal,
  duplicate_keep_option keep_option,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  if (auto const child_type = keys.child().type();
      cudf::is_nested(child_type) && child_type.id() != type_id::STRUCT) {
    CUDF_FAIL(
      "Keys of nested types other than STRUCT are not supported in `drop_list_duplicates`.");
  }

  auto const has_values = values.has_value();
  CUDF_EXPECTS(!has_values || keys.size() == values.value().size(),
               "Keys and values columns must have the same size.");

  if (keys.is_empty()) {
    return std::pair{cudf::empty_like(keys.parent()),
                     has_values ? cudf::empty_like(values.value().parent()) : nullptr};
  }

  // Generate a 0-based offset column.
  auto lists_offsets = detail::generate_clean_offsets(keys, stream);

  auto const keys_child = keys.get_sliced_child(stream);

  // Generate a mapping from list entries to their 1-based list indices.
  auto const entries_list_indices =
    detail::generate_entry_list_indices(keys_child.size(), lists_offsets, stream);

  // Generate the segmented sorted order of the keys lists column.
  // The keys column will be sorted (gathered) using this order.
  auto const sorted_order = [&]() {
    auto const list_indices_view = column_view(data_type{type_to_id<offset_type>()},
                                               static_cast<size_type>(entries_list_indices.size()),
                                               entries_list_indices.data());

    // If nans_equal == ALL_EQUAL and the keys column contains lists of floating-point data type,
    // we need to replace `-NaN` by `NaN` before sorting.
    auto const replace_negative_nan =
      nans_equal == nan_equality::ALL_EQUAL &&
      type_dispatcher(keys_child.type(), detail::has_negative_nans_dispatch{}, keys_child, stream);

    if (replace_negative_nan) {
      auto const replaced_nan_keys_child = type_dispatcher(
        keys_child.type(), detail::replace_negative_nans_dispatch{}, keys_child, stream);
      auto const sorting_cols =
        std::vector<column_view>{list_indices_view, replaced_nan_keys_child->view()};
      return cudf::detail::stable_sorted_order(table_view{sorting_cols},
                                               {order::ASCENDING, order::ASCENDING},
                                               {null_order::AFTER, null_order::AFTER},
                                               stream);
    } else {
      auto const sorting_cols = std::vector<column_view>{list_indices_view, keys_child};
      return cudf::detail::stable_sorted_order(table_view{sorting_cols},
                                               {order::ASCENDING, order::ASCENDING},
                                               {null_order::AFTER, null_order::AFTER},
                                               stream);
    }
  }();

  auto const sorting_table = has_values
                               ? table_view{{keys_child, values.value().get_sliced_child(stream)}}
                               : table_view{{keys_child}};
  auto const sorted_table  = cudf::detail::gather(sorting_table,
                                                 sorted_order->view(),
                                                 out_of_bounds_policy::DONT_CHECK,
                                                 cudf::detail::negative_index_policy::NOT_ALLOWED,
                                                 stream);

  // Extract the segmented sorted entries.
  auto const sorted_keys_entries = sorted_table->get_column(0).view();
  auto const sorted_values_entries =
    has_values ? std::optional<column_view>(sorted_table->get_column(1).view()) : std::nullopt;

  // Generate columns containing unique entries (along with their list indices).
  // null_count and bitmask of these column will also be generated in this function.
  auto unique_entries_and_list_indices =
    detail::get_unique_entries_and_list_indices(sorted_keys_entries,
                                                sorted_values_entries,
                                                entries_list_indices,
                                                nulls_equal,
                                                nans_equal,
                                                keep_option,
                                                stream,
                                                mr);

  // Generate offsets for the output lists column(s).
  auto output_offsets = detail::generate_output_offsets(
    unique_entries_and_list_indices.front()->size(),  // num unique entries
    unique_entries_and_list_indices.back()->view(),   // unique entries' list indices
    lists_offsets,
    stream,
    mr);

  // If the input values lists column is not given, its corresponding output will be nullptr.
  auto out_values =
    has_values ? make_lists_column(keys.size(),
                                   std::make_unique<column>(output_offsets->view()),
                                   std::move(unique_entries_and_list_indices[1]),
                                   values.value().null_count(),
                                   cudf::detail::copy_bitmask(values.value().parent(), stream, mr))
               : nullptr;

  auto out_keys = make_lists_column(keys.size(),
                                    std::move(output_offsets),
                                    std::move(unique_entries_and_list_indices[0]),
                                    keys.null_count(),
                                    cudf::detail::copy_bitmask(keys.parent(), stream, mr));

  return std::pair{std::move(out_keys), std::move(out_values)};
}

}  // anonymous namespace

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> drop_list_duplicates(
  lists_column_view const& keys,
  lists_column_view const& values,
  null_equality nulls_equal,
  nan_equality nans_equal,
  duplicate_keep_option keep_option,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return drop_list_duplicates_common(keys,
                                     std::optional<lists_column_view>(values),
                                     nulls_equal,
                                     nans_equal,
                                     keep_option,
                                     stream,
                                     mr);
}

std::unique_ptr<column> drop_list_duplicates(lists_column_view const& input,
                                             null_equality nulls_equal,
                                             nan_equality nans_equal,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  return drop_list_duplicates_common(input,
                                     std::nullopt,
                                     nulls_equal,
                                     nans_equal,
                                     duplicate_keep_option::KEEP_FIRST,
                                     stream,
                                     mr)
    .first;
}

}  // namespace detail

/**
 * @copydoc cudf::lists::drop_list_duplicates(lists_column_view const&,
 *                                            lists_column_view const&,
 *                                            duplicate_keep_option,
 *                                            null_equality,
 *                                            nan_equality,
 *                                            rmm::mr::device_memory_resource*)
 */
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> drop_list_duplicates(
  lists_column_view const& keys,
  lists_column_view const& values,
  duplicate_keep_option keep_option,
  null_equality nulls_equal,
  nan_equality nans_equal,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::drop_list_duplicates(
    keys, values, nulls_equal, nans_equal, keep_option, rmm::cuda_stream_default, mr);
}

/**
 * @copydoc cudf::lists::drop_list_duplicates(lists_column_view const&,
 *                                            null_equality,
 *                                            nan_equality,
 *                                            rmm::mr::device_memory_resource*)
 */
std::unique_ptr<column> drop_list_duplicates(lists_column_view const& input,
                                             null_equality nulls_equal,
                                             nan_equality nans_equal,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::drop_list_duplicates(input, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
