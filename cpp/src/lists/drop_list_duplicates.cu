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
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/lists/drop_list_duplicates.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {
struct SortPairs {
  template <typename KeyT, typename ValueT, typename OffsetIteratorT>
  void SortPairsAscending(KeyT const* keys_in,
                          KeyT* keys_out,
                          ValueT const* values_in,
                          ValueT* values_out,
                          int num_items,
                          int num_segments,
                          OffsetIteratorT begin_offsets,
                          OffsetIteratorT end_offsets,
                          rmm::cuda_stream_view stream)
  {
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage.data(),
                                             temp_storage_bytes,
                                             keys_in,
                                             keys_out,
                                             values_in,
                                             values_out,
                                             num_items,
                                             num_segments,
                                             begin_offsets,
                                             end_offsets,
                                             0,
                                             sizeof(KeyT) * 8,
                                             stream.value());
    d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};

    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage.data(),
                                             temp_storage_bytes,
                                             keys_in,
                                             keys_out,
                                             values_in,
                                             values_out,
                                             num_items,
                                             num_segments,
                                             begin_offsets,
                                             end_offsets,
                                             0,
                                             sizeof(KeyT) * 8,
                                             stream.value());
  }

  template <typename T>
  std::enable_if_t<not is_numeric<T>(), std::unique_ptr<column>> operator()(
    column_view const& child,
    column_view const& segment_offsets,
    null_order null_precedence,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto child_table = segmented_sort_by_key(table_view{{child}},
                                             table_view{{child}},
                                             segment_offsets,
                                             {order::ASCENDING},
                                             {null_precedence},
                                             stream,
                                             mr);
    return std::move(child_table->release().front());
  }

  template <typename T>
  std::enable_if_t<is_numeric<T>(), std::unique_ptr<column>> operator()(
    column_view const& child,
    column_view const& offsets,
    null_order null_precedence,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto output =
      cudf::detail::allocate_like(child, child.size(), mask_allocation_policy::NEVER, stream, mr);
    mutable_column_view mutable_output_view = output->mutable_view();

    auto keys = [&]() {
      if (child.nullable()) {
        rmm::device_uvector<T> keys(child.size(), stream);
        auto const null_replace_T = null_precedence == null_order::AFTER
                                      ? std::numeric_limits<T>::max()
                                      : std::numeric_limits<T>::min();
        auto device_child         = column_device_view::create(child, stream);
        auto keys_in =
          cudf::detail::make_null_replacement_iterator<T>(*device_child, null_replace_T);
        thrust::copy_n(rmm::exec_policy(stream), keys_in, child.size(), keys.begin());
        return keys;
      }
      return rmm::device_uvector<T>{0, stream};
    }();

    std::unique_ptr<column> sorted_indices = cudf::make_numeric_column(
      data_type(type_to_id<size_type>()), child.size(), mask_state::UNALLOCATED, stream, mr);
    mutable_column_view mutable_indices_view = sorted_indices->mutable_view();
    thrust::sequence(rmm::exec_policy(stream),
                     mutable_indices_view.begin<size_type>(),
                     mutable_indices_view.end<size_type>(),
                     0);

    SortPairsAscending(child.nullable() ? keys.data() : child.begin<T>(),
                       mutable_output_view.begin<T>(),
                       mutable_indices_view.begin<size_type>(),
                       mutable_indices_view.begin<size_type>(),
                       child.size(),
                       offsets.size() - 1,
                       offsets.begin<size_type>(),
                       offsets.begin<size_type>() + 1,
                       stream);

    std::vector<std::unique_ptr<column>> output_cols;
    output_cols.push_back(std::move(output));
    // rearrange the null_mask.
    cudf::detail::gather_bitmask(cudf::table_view{{child}},
                                 mutable_indices_view.begin<size_type>(),
                                 output_cols,
                                 cudf::detail::gather_bitmask_op::DONT_CHECK,
                                 stream,
                                 mr);
    return std::move(output_cols.front());
  }
};
#if 1
namespace {
template <typename InputIterator, typename BinaryPredicate>
struct unique_copy_fn {
  /**
   * @brief Functor for unique_copy()
   *
   * The logic here is equivalent to:
   * @code
   *   ((keep == duplicate_keep_option::KEEP_LAST) ||
   *    (i == 0 || !comp(iter[i], iter[i - 1]))) &&
   *   ((keep == duplicate_keep_option::KEEP_FIRST) ||
   *    (i == last_index || !comp(iter[i], iter[i + 1])))
   * @endcode
   *
   * It is written this way so that the `comp` comparator
   * function appears only once minimizing the inlining
   * required and reducing the compile time.
   */
  __device__ bool operator()(size_type i)
  {
    size_type boundary = 0;
    size_type offset   = 1;
    auto keep_option   = duplicate_keep_option::KEEP_LAST;
    do {
      if ((keep_option != duplicate_keep_option::KEEP_FIRST) && (i != boundary) &&
          comp(iter[i], iter[i - offset])) {
        return false;
      }
      keep_option = duplicate_keep_option::KEEP_FIRST;
      boundary    = last_index;
      offset      = -offset;
    } while (offset < 0);
    return true;
  }

  InputIterator iter;
  BinaryPredicate comp;
  size_type const last_index;
};
}  // anonymous namespace

/**
 * @brief Copies unique elements from the range [first, last) to output iterator `output`.
 *
 * In a consecutive group of duplicate elements, depending on parameter `keep`,
 * only the first element is copied, or the last element is copied or neither is copied.
 *
 * @return End of the range to which the elements are copied.
 */
template <typename InputIterator, typename OutputIterator, typename BinaryPredicate>
OutputIterator unique_copy(InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate comp,
                           rmm::cuda_stream_view stream)
{
  size_type const last_index = thrust::distance(first, last) - 1;
  return thrust::copy_if(rmm::exec_policy(stream),
                         first,
                         last,
                         thrust::counting_iterator<size_type>(0),
                         output,
                         unique_copy_fn<InputIterator, BinaryPredicate>{first, comp, last_index});
}

/**
 * @brief Create a column_view of index values which represents the row values of a given column
 * without duplicated elements
 *
 * Given an `input` column_view, the output column `unique_indices` is generated such that it
 * copies row indices of the input column at rows without duplicate elements
 *
 * @param[in] input              The input column_view
 * @param[out] unique_indices    Column to store the indices of rows with unique elements
 * @param[in] nulls_equal        flag to denote nulls are equal if null_equality::EQUAL,
 *                               nulls are not equal if null_equality::UNEQUAL
 * @param[in] stream             CUDA stream used for device memory operations and kernel launches
 *
 * @return column_view of unique row indices
 */
std::unique_ptr<table> get_unique_entries(column_view const& input,
                                          column_view const& entry_offsets,

                                          null_equality nulls_equal,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  // sort only indices

  // extract unique indices
  auto device_input_table = cudf::table_device_view::create(table_view{{input}}, stream);

  auto unique_indices = cudf::make_numeric_column(
    data_type{type_id::INT32}, input.size(), mask_state::UNALLOCATED, stream);
  auto mutable_unique_indices_view = unique_indices->mutable_view();

  auto comp = row_equality_comparator<false>(
    *device_input_table, *device_input_table, nulls_equal == null_equality::EQUAL);

  auto begin = thrust::make_counting_iterator(0);
  auto end   = begin + input.size();
  auto result_end =
    unique_copy(begin, end, mutable_unique_indices_view.begin<cudf::size_type>(), comp, stream);

  auto indices = cudf::detail::slice(
    unique_indices->view(),
    0,
    thrust::distance(mutable_unique_indices_view.begin<cudf::size_type>(), result_end));

  return cudf::detail::gather(table_view{{input, entry_offsets}},
                              indices,
                              cudf::out_of_bounds_policy::DONT_CHECK,
                              cudf::detail::negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr);
  // return std::move(tbl->release().front());
}

#endif

/**
 * @brief Populate offsets for all corresponding indices of the entries
 *
 * Given an `offsets` column_view and a number of entries, generate an array of corresponding
 * offsets for all the entry indices
 *
 * @code{.pseudo}
 * size = 9, offsets = { 0, 4, 6, 10 }
 * output = { 0, 0, 0, 0, 1, 1, 2, 2, 2, 2 }
 * @endcode
 *
 * @param[in] size        The number of entries to populate offsets
 * @param[out] offsets    Column view to the offsets
 * @param[in] stream      CUDA stream used for device memory operations and kernel launches
 *
 * @return a device vector of entry offsets
 */
std::unique_ptr<column> get_entry_offsets(size_type size,
                                          column_view const& offsets,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  auto entry_offsets = make_numeric_column(
    data_type(type_to_id<size_type>()), size, mask_state::UNALLOCATED, stream, mr);
  auto entry_offsets_view = entry_offsets->mutable_view();

  auto offset_begin = offsets.begin<size_type>();
  auto offsets_minus_one =
    thrust::make_transform_iterator(offset_begin, [] __device__(auto i) { return i - 1; });
  auto counting_iter = thrust::make_counting_iterator<size_type>(0);
  thrust::lower_bound(rmm::exec_policy(stream),
                      offsets_minus_one,
                      offsets_minus_one + offsets.size(),
                      counting_iter,
                      counting_iter + size,
                      entry_offsets_view.begin<size_type>());
  return entry_offsets;
}

/**
 * @copydoc cudf::lists::drop_list_duplicates
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> drop_list_duplicates(lists_column_view const& lists_column,
                                             null_equality nulls_equal,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  /*
   * Given input = { {1, 1, 2, 1, 3}, {4}, {5, 6, 6, 6, 5} }
   * The output can be { {1, 2, 3}, {4}, {5, 6} }
   *
   * 1. Call segmented sort on the lists, obtaining
   *   sorted_lists = { {1, 1, 1, 2, 3}, {4}, {5, 5, 6, 6, 6} }, and
   *
   * 2. Generate ordered indices for the list entries
   *   indices = { {0, 1, 2, 3, 4}, {5}, {6, 7, 8, 9, 10} }
   *
   * 3. Remove list indices if the list entries are duplicated, obtaining
   *   unique_indices = { {0, 3, 4}, {5}, {6, 8} }
   *
   * 4. Gather list entries using the unique_indices as gather map
   *   (remember to deal with null elements)
   *
   * 5. Regenerate list offsets and null mask
   *
   *  Corner cases:
   *   - null entries in a list: depending on the nulls_equal policy, if it is set to EQUAL then
   * only one null entry is kept.
   *   - null rows: just return null rows, nothing changes.
   *   - NaN entries in a list: NaNs should be treated as equal, thus only one NaN value is kept.
   *   - Nested types are not supported---the function should throw logic_error.
   */

  if (lists_column.is_empty()) return cudf::empty_like(lists_column.parent());
  if (cudf::is_nested(lists_column.child().type()))
    CUDF_FAIL("Nested types are not supported in drop_list_duplicates.");

  /*
   * 1. Call segmented sort on the lists
   */
  //  std::unique_ptr<column> sorted_lists;
  //  if (nulls_equal == null_equality::EQUAL)
  //    sorted_lists = cudf::lists::sort_lists(lists_column, order::ASCENDING, null_order::AFTER);
  //  else
  //    sorted_lists = cudf::lists::sort_lists(lists_column, order::ASCENDING, null_order::BEFORE);

  auto print = [](const cudf::column_view& arr) {
    thrust::host_vector<int> host_data(arr.size());
    CUDA_TRY(cudaMemcpy(
      host_data.data(), arr.data<int>(), arr.size() * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < arr.size(); ++i) { printf("%d\n", host_data[i]); }
    printf("\n\n\n");
  };

  // Get the original offsets, which may not start from 0
  auto segment_offsets = cudf::detail::slice(
    lists_column.offsets(), {lists_column.offset(), lists_column.offsets().size()}, stream)[0];
  // Make 0-based offsets so we can create a new column using those offsets
  auto output_offset = allocate_like(segment_offsets, mask_allocation_policy::RETAIN, mr);
  thrust::transform(rmm::exec_policy(stream),
                    segment_offsets.begin<size_type>(),
                    segment_offsets.end<size_type>(),
                    output_offset->mutable_view().begin<size_type>(),
                    [first = segment_offsets.begin<size_type>()] __device__(auto offset_index) {
                      return offset_index - *first;
                    });

  // return sorted_lists;

  /*
   * 2,3. Generate ordered indices for the list entries and remove list indices if the list entries
   * are duplicated
   */
  auto const child         = lists_column.get_sliced_child(stream);
  auto const entry_offsets = get_entry_offsets(child.size(), output_offset->view(), stream, mr);

  //  if (nulls_equal == null_equality::EQUAL)
  //    sorted_lists = cudf::lists::sort_lists(lists_column, order::ASCENDING, null_order::AFTER);
  //  else
  //    sorted_lists = cudf::lists::sort_lists(lists_column, order::ASCENDING, null_order::BEFORE);

  auto output_child = type_dispatcher(
    child.type(), SortPairs{}, child, output_offset->view(), null_order::AFTER, stream, mr);
  //  output_child = type_dispatcher(
  //    child.type(), MakeUniquePairs{}, child, entry_offsets, null_precedence, stream, mr);

  printf("line %d\n", __LINE__);

  printf("line %d\n", __LINE__);

  auto output_child_and_entry_offsets = get_unique_entries(output_child->view(),
                                                           entry_offsets->view(),

                                                           nulls_equal,
                                                           stream,
                                                           mr)
                                          ->release();

  printf("line %d\n", __LINE__);

  auto null_mask = cudf::detail::copy_bitmask(lists_column.parent(), stream, mr);

  printf("line %d\n", __LINE__);

  print(output_child->view());

  print(output_offset->view());

  print(output_child_and_entry_offsets.back()->view());
  /*
   * 0 4 5 8 10
   */

  auto unique_entry_offsets = output_child_and_entry_offsets.back()->view().data<size_type>();

  auto counting_iter = thrust::make_counting_iterator<size_type>(0);
  //  auto result_end    =
  thrust::copy_if(rmm::exec_policy(stream),
                  counting_iter,
                  counting_iter + child.size(),
                  output_offset->mutable_view().begin<size_type>(),
                  [size = child.size(), unique_entry_offsets] __device__(size_type i) -> bool {
                    return i == 0 || i == size ||
                           unique_entry_offsets[i] != unique_entry_offsets[i - 1];
                  });

  print(output_offset->view());

  // Assemble list column & return
  return make_lists_column(lists_column.size(),
                           std::move(output_offset),
                           std::move(output_child_and_entry_offsets.front()),
                           lists_column.null_count(),
                           std::move(null_mask));

  /*
   * 4. Gather list entries using the unique_indices as gather map
   */
  //  return cudf::detail::gather(lists_data,
  //                              unique_indices,
  //                              cudf::out_of_bounds_policy::DONT_CHECK,
  //                              cudf::detail::negative_index_policy::NOT_ALLOWED,
  //                              stream,
  //                              mr);

  /*
   * 5. Regenerate list offsets and null mask
   */

  //  cudf::detail::gather_bitmask(cudf::table_view{{child}},
  //                               mutable_indices_view.begin<size_type>(),
  //                               output_cols,
  //                               cudf::detail::gather_bitmask_op::DONT_CHECK,
  //                               stream,
  //                               mr);

  return empty_like(lists_column.parent());
}

}  // namespace detail

/**
 * @copydoc cudf::lists::drop_list_duplicates
 */
std::unique_ptr<column> drop_list_duplicates(lists_column_view const& lists_column,
                                             null_equality nulls_equal,
                                             rmm::mr::device_memory_resource* mr)
{
  return detail::drop_list_duplicates(lists_column, nulls_equal, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
