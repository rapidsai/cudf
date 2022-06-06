/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <stream_compaction/stream_compaction_common.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/replace.hpp>
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

#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include <optional>

namespace cudf::lists {
namespace detail {

namespace {
template <typename Type>
struct has_negative_nans_fn {
  column_device_view const d_view;

  has_negative_nans_fn(column_device_view const& d_view) : d_view(d_view) {}

  __device__ Type operator()(size_type idx) const noexcept
  {
    if (d_view.is_null(idx)) { return false; }

    auto const val = d_view.element<Type>(idx);
    return std::isnan(val) && std::signbit(val);  // std::signbit(x) == true if x is negative
  }
};

/**
 * @brief A structure to be used along with type_dispatcher to check if a column has any
 * negative NaN value.
 *
 * This functor is necessary because when calling to segmented sort on the list entries, the
 * negative NaN and positive NaN values (if both exist) are separated to the two ends of the output
 * lists. We want to move all NaN values close together in order to call unique_copy later on.
 */
struct has_negative_nans_dispatch {
  template <typename Type, std::enable_if_t<cuda::std::is_floating_point_v<Type>>* = nullptr>
  bool operator()(column_view const& input, rmm::cuda_stream_view stream) const noexcept
  {
    auto const d_entries_ptr = column_device_view::create(input, stream);
    return thrust::count_if(rmm::exec_policy(stream),
                            thrust::make_counting_iterator(0),
                            thrust::make_counting_iterator(input.size()),
                            has_negative_nans_fn<Type>{*d_entries_ptr});
  }

  template <typename Type, std::enable_if_t<std::is_same_v<Type, cudf::struct_view>>* = nullptr>
  bool operator()(column_view const& input, rmm::cuda_stream_view stream) const
  {
    // Recursively check negative NaN on the children columns.
    return std::any_of(thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(input.num_children()),
                       [structs_view = structs_column_view{input}, stream](auto const child_idx) {
                         auto const col = structs_view.get_sliced_child(child_idx);
                         return type_dispatcher(
                           col.type(), has_negative_nans_dispatch{}, col, stream);
                       });
  }

  template <typename Type,
            std::enable_if_t<!cuda::std::is_floating_point_v<Type> &&
                             !std::is_same_v<Type, cudf::struct_view>>* = nullptr>
  bool operator()(column_view const&, rmm::cuda_stream_view) const
  {
    // Non-nested columns of non floating-point data do not contain NaN.
    // Nested columns (not STRUCT) are not supported and should not reach this point.
    return false;
  }
};

/**
 * @brief A structure to be used along with type_dispatcher to replace -NaN by NaN for a
 * floating-point data column.
 *
 * Replacing -NaN by NaN is necessary before calling to segmented sort for lists because the sorting
 * API may separate -NaN and NaN to the two ends of each result list while we want to group all NaN
 * together.
 */
struct replace_negative_nans_dispatch {
  template <typename Type, std::enable_if_t<!std::is_same_v<Type, cudf::struct_view>>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input,
                                     rmm::cuda_stream_view stream) const noexcept
  {
    return cuda::std::is_floating_point_v<Type>
             ? cudf::detail::normalize_nans_and_zeros(input, stream)
             : std::make_unique<column>(input, stream);
  }

  template <typename Type, std::enable_if_t<std::is_same_v<Type, cudf::struct_view>>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input,
                                     rmm::cuda_stream_view stream) const noexcept
  {
    std::vector<std::unique_ptr<cudf::column>> output_struct_members;
    std::transform(thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(input.num_children()),
                   std::back_inserter(output_struct_members),
                   [structs_view = structs_column_view{input}, stream](auto const child_idx) {
                     auto const col = structs_view.get_sliced_child(child_idx);
                     return type_dispatcher(
                       col.type(), replace_negative_nans_dispatch{}, col, stream);
                   });

    return cudf::make_structs_column(input.size(),
                                     std::move(output_struct_members),
                                     input.null_count(),
                                     cudf::detail::copy_bitmask(input, stream),
                                     stream);
  }
};

/**
 * @brief Perform an equality comparison between two entries in a lists column, specialized from
 * `cudf::element_equality_comparator` to take into account both parameters `nulls_equal` and
 * `nans_equal` when comparing floating-point numbers.
 *
 * For the two entries that are NOT in the same list, they will always be considered as different.
 *
 * If they are from the same list and their type is not floating point, this functor will return the
 * same comparison result as `cudf::element_equality_comparator`.
 *
 * For floating-point types, entries holding NaN value can be considered as different or the same
 * value depending on the `nans_equal` parameter.
 */
template <class Type>
struct column_row_comparator_fn {
  size_type const* const list_indices;
  column_device_view const lhs;
  column_device_view const rhs;
  null_equality const nulls_equal;
  bool const has_nulls;
  bool const nans_equal;

  __host__ __device__ column_row_comparator_fn(size_type const* const list_indices,
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
    // If both element(i) and element(j) are NaNs and NaNs are considered as equal value then this
    // comparison will return `true`. This is the desired behavior in Pandas.
    if (nans_equal && std::isnan(lhs_val) && std::isnan(rhs_val)) { return true; }

    // If NaNs are considered as NOT equal, even both element(i) and element(j) are NaNs this
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
  size_type const* const list_indices;
  column_device_view const lhs;
  column_device_view const rhs;
  null_equality const nulls_equal;
  bool const has_nulls;
  bool const nans_equal;

  __device__ column_row_comparator_dispatch(size_type const* const list_indices,
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
 * @brief Performs an equality comparison between rows of two tables using
 * `column_row_comparator_fn` functor to compare rows of their corresponding columns.
 */
struct table_row_comparator_fn {
  size_type const* const list_indices;
  table_device_view const lhs;
  table_device_view const rhs;
  null_equality const nulls_equal;
  bool const has_nulls;
  bool const nans_equal;

  table_row_comparator_fn(size_type const* const list_indices,
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

  bool __device__ operator()(size_type i, size_type j) const
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
 *  @brief Struct used in type_dispatcher for copying indices of the list entries ignoring duplicate
 * list entries.
 */
struct get_indices_of_unique_entries_dispatch {
  template <class Type,
            std::enable_if_t<!cudf::is_equality_comparable<Type, Type>() &&
                             !std::is_same_v<Type, cudf::struct_view>>* = nullptr>
  size_type* operator()(size_type const*,
                        column_view const&,
                        size_type,
                        size_type*,
                        null_equality,
                        nan_equality,
                        duplicate_keep_option,
                        rmm::cuda_stream_view) const
  {
    CUDF_FAIL(
      "get_indices_of_unique_entries_dispatch cannot operate on types that are not equally "
      "comparable or not STRUCT type.");
  }

  template <class Type, std::enable_if_t<cudf::is_equality_comparable<Type, Type>()>* = nullptr>
  size_type* operator()(size_type const* list_indices,
                        column_view const& all_lists_entries,
                        size_type num_entries,
                        size_type* output_begin,
                        null_equality nulls_equal,
                        nan_equality nans_equal,
                        duplicate_keep_option keep_option,
                        rmm::cuda_stream_view stream) const noexcept
  {
    auto const d_view = column_device_view::create(all_lists_entries, stream);
    auto const comp   = column_row_comparator_fn<Type>{list_indices,
                                                     *d_view,
                                                     *d_view,
                                                     nulls_equal,
                                                     all_lists_entries.has_nulls(),
                                                     nans_equal == nan_equality::ALL_EQUAL};
    return cudf::detail::unique_copy(thrust::make_counting_iterator(0),
                                     thrust::make_counting_iterator(num_entries),
                                     output_begin,
                                     comp,
                                     keep_option,
                                     stream);
  }

  template <class Type, std::enable_if_t<std::is_same_v<Type, cudf::struct_view>>* = nullptr>
  size_type* operator()(size_type const* list_indices,
                        column_view const& all_lists_entries,
                        size_type num_entries,
                        size_type* output_begin,
                        null_equality nulls_equal,
                        nan_equality nans_equal,
                        duplicate_keep_option keep_option,
                        rmm::cuda_stream_view stream) const noexcept
  {
    auto const flattened_entries = cudf::structs::detail::flatten_nested_columns(
      table_view{{all_lists_entries}}, {order::ASCENDING}, {null_order::AFTER}, {});
    auto const dview_ptr = table_device_view::create(flattened_entries, stream);
    // Search through children of all levels for nulls.
    auto const nested_has_nulls = has_nulls(flattened_entries.flattened_columns());

    auto const comp = table_row_comparator_fn{list_indices,
                                              *dview_ptr,
                                              *dview_ptr,
                                              nulls_equal,
                                              nested_has_nulls,
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
 * @brief Extract list entries and their corresponding (1-based) list indices ignoring duplicate
 * entries.
 */
std::vector<std::unique_ptr<column>> get_unique_entries_and_list_indices(
  column_view const& keys_entries,
  std::optional<column_view> const& values_entries,
  device_span<size_type const> entries_list_indices,
  null_equality nulls_equal,
  nan_equality nans_equal,
  duplicate_keep_option keep_option,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto const num_entries = keys_entries.size();

  // Allocate memory to store the indices of the unique key entries.
  // These indices will be used as a gather map to collect keys and values.
  auto unique_indices     = rmm::device_uvector<size_type>(num_entries, stream);
  auto const output_begin = unique_indices.begin();
  auto const output_end   = type_dispatcher(keys_entries.type(),
                                          get_indices_of_unique_entries_dispatch{},
                                          entries_list_indices.begin(),
                                          keys_entries,
                                          num_entries,
                                          output_begin,
                                          nulls_equal,
                                          nans_equal,
                                          keep_option,
                                          stream);

  auto const list_indices_view = column_view(data_type{type_to_id<size_type>()},
                                             static_cast<size_type>(entries_list_indices.size()),
                                             entries_list_indices.data());
  auto const input_table       = values_entries
                                   ? table_view{{keys_entries, values_entries.value(), list_indices_view}}
                                   : table_view{{keys_entries, list_indices_view}};

  // Collect unique entries and entry list indices.
  // The new null_count and bitmask of the unique entries will also be generated by the gather
  // function.
  return cudf::detail::gather(input_table,
                              device_span<size_type const>(
                                unique_indices.data(), thrust::distance(output_begin, output_end)),
                              cudf::out_of_bounds_policy::DONT_CHECK,
                              cudf::detail::negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr)
    ->release();
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

  CUDF_EXPECTS(!values || keys.size() == values.value().size(),
               "Keys and values columns must have the same size.");

  if (keys.is_empty()) {
    return std::pair{cudf::empty_like(keys.parent()),
                     values ? cudf::empty_like(values.value().parent()) : nullptr};
  }

  // The child column containing list entries.
  auto const keys_child = keys.get_sliced_child(stream);

  // Generate a mapping from list entries to their list indices for the keys column.
  auto const entries_list_indices = [&] {
    auto labels = rmm::device_uvector<size_type>(keys_child.size(), stream);
    cudf::detail::label_segments(
      keys.offsets_begin(), keys.offsets_end(), labels.begin(), labels.end(), stream);
    return labels;
  }();

  // Generate segmented sorted order for key entries.
  // The keys column will be sorted (gathered) using this order.
  auto const sorted_order = [&]() {
    auto const list_indices_view = column_view(data_type{type_to_id<size_type>()},
                                               static_cast<size_type>(entries_list_indices.size()),
                                               entries_list_indices.data());

    // If nans_equal == ALL_EQUAL and the keys column contains floating-point data type,
    // we need to replace `-NaN` by `NaN` before sorting.
    auto const replace_negative_nan =
      nans_equal == nan_equality::ALL_EQUAL &&
      type_dispatcher(keys_child.type(), has_negative_nans_dispatch{}, keys_child, stream);

    if (replace_negative_nan) {
      auto const replaced_nan_keys_child =
        type_dispatcher(keys_child.type(), replace_negative_nans_dispatch{}, keys_child, stream);
      return cudf::detail::stable_sorted_order(
        table_view{{list_indices_view, replaced_nan_keys_child->view()}},
        {order::ASCENDING, order::ASCENDING},
        {null_order::AFTER, null_order::AFTER},
        stream);
    } else {
      return cudf::detail::stable_sorted_order(table_view{{list_indices_view, keys_child}},
                                               {order::ASCENDING, order::ASCENDING},
                                               {null_order::AFTER, null_order::AFTER},
                                               stream);
    }
  }();

  auto const sorting_table = values
                               ? table_view{{keys_child, values.value().get_sliced_child(stream)}}
                               : table_view{{keys_child}};
  auto const sorted_table  = cudf::detail::gather(sorting_table,
                                                 sorted_order->view(),
                                                 out_of_bounds_policy::DONT_CHECK,
                                                 cudf::detail::negative_index_policy::NOT_ALLOWED,
                                                 stream);

  // Extract the segmented sorted key entries.
  auto const sorted_keys_entries = sorted_table->get_column(0).view();
  auto const sorted_values_entries =
    values ? std::optional<column_view>(sorted_table->get_column(1).view()) : std::nullopt;

  // Generate child columns containing unique entries (along with their list indices).
  // null_count and bitmask of these columns will also be generated in this function.
  auto unique_entries_and_list_indices = get_unique_entries_and_list_indices(sorted_keys_entries,
                                                                             sorted_values_entries,
                                                                             entries_list_indices,
                                                                             nulls_equal,
                                                                             nans_equal,
                                                                             keep_option,
                                                                             stream,
                                                                             mr);

  // Generate offsets for the output lists column(s).
  auto output_offsets = [&] {
    auto out_offsets = make_numeric_column(
      data_type{type_to_id<offset_type>()}, keys.size() + 1, mask_state::UNALLOCATED, stream, mr);
    auto const offsets = out_offsets->mutable_view();
    auto const labels =
      unique_entries_and_list_indices.back()->view();  // unique entries' list indices
    cudf::detail::labels_to_offsets(labels.template begin<size_type>(),
                                    labels.template end<size_type>(),
                                    offsets.template begin<size_type>(),
                                    offsets.template end<size_type>(),
                                    stream);
    return out_offsets;
  }();

  // If the values lists column is not given, its corresponding output will be nullptr.
  auto out_values =
    values ? make_lists_column(keys.size(),
                               std::make_unique<column>(output_offsets->view(), stream, mr),
                               std::move(unique_entries_and_list_indices[1]),
                               values.value().null_count(),
                               cudf::detail::copy_bitmask(values.value().parent(), stream, mr),
                               stream,
                               mr)
           : nullptr;

  auto out_keys = make_lists_column(keys.size(),
                                    std::move(output_offsets),
                                    std::move(unique_entries_and_list_indices[0]),
                                    keys.null_count(),
                                    cudf::detail::copy_bitmask(keys.parent(), stream, mr),
                                    stream,
                                    mr);

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

}  // namespace cudf::lists
