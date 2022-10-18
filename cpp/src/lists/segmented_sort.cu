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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cub/device/device_segmented_radix_sort.cuh>

namespace cudf {
namespace lists {
namespace detail {

struct SegmentedSortColumn {
  /**
   * @brief Compile time check for allowing radix sort for column type.
   *
   * Floating point is not included here because of the special handling of NaNs.
   */
  template <typename T>
  static constexpr bool is_radix_sort_supported()
  {
    return std::is_integral<T>();
  }

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

  template <typename KeyT, typename ValueT, typename OffsetIteratorT>
  void SortPairsDescending(KeyT const* keys_in,
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
    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage.data(),
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

    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage.data(),
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
  std::enable_if_t<not is_radix_sort_supported<T>(), std::unique_ptr<column>> operator()(
    column_view const& child,
    column_view const& segment_offsets,
    order column_order,
    null_order null_precedence,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto child_table = segmented_sort_by_key(table_view{{child}},
                                             table_view{{child}},
                                             segment_offsets,
                                             {column_order},
                                             {null_precedence},
                                             stream,
                                             mr);
    return std::move(child_table->release().front());
  }

  template <typename T>
  std::enable_if_t<is_radix_sort_supported<T>(), std::unique_ptr<column>> operator()(
    column_view const& child,
    column_view const& offsets,
    order column_order,
    null_order null_precedence,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    // the average list size at which to prefer radixsort:
    constexpr cudf::size_type MIN_AVG_LIST_SIZE_FOR_RADIXSORT{100};

    if ((child.size() / offsets.size()) < MIN_AVG_LIST_SIZE_FOR_RADIXSORT) {
      auto child_table = segmented_sort_by_key(table_view{{child}},
                                               table_view{{child}},
                                               offsets,
                                               {column_order},
                                               {null_precedence},
                                               stream,
                                               mr);
      return std::move(child_table->release().front());
    }

    auto output =
      cudf::detail::allocate_like(child, child.size(), mask_allocation_policy::NEVER, stream, mr);
    mutable_column_view mutable_output_view = output->mutable_view();

    auto keys = [&]() {
      if (child.nullable()) {
        rmm::device_uvector<T> keys(child.size(), stream);
        auto const null_replace_T = null_precedence == null_order::AFTER
                                      ? std::numeric_limits<T>::max()
                                      : std::numeric_limits<T>::min();

        auto device_child = column_device_view::create(child, stream);
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

    if (column_order == order::ASCENDING)
      SortPairsAscending(child.nullable() ? keys.data() : child.begin<T>(),
                         mutable_output_view.begin<T>(),
                         mutable_indices_view.begin<size_type>(),
                         mutable_indices_view.begin<size_type>(),
                         child.size(),
                         offsets.size() - 1,
                         offsets.begin<size_type>(),
                         offsets.begin<size_type>() + 1,
                         stream);
    else
      SortPairsDescending(child.nullable() ? keys.data() : child.begin<T>(),
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

std::unique_ptr<column> sort_lists(lists_column_view const& input,
                                   order column_order,
                                   null_order null_precedence,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) return empty_like(input.parent());
  auto output_offset = make_numeric_column(
    input.offsets().type(), input.size() + 1, mask_state::UNALLOCATED, stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    input.offsets_begin(),
                    input.offsets_end(),
                    output_offset->mutable_view().begin<size_type>(),
                    [first = input.offsets_begin()] __device__(auto offset_index) {
                      return offset_index - *first;
                    });
  // for numeric columns, calls Faster segmented radix sort path
  // for non-numeric columns, calls segmented_sort_by_key.
  auto output_child = type_dispatcher<dispatch_storage_type>(input.child().type(),
                                                             SegmentedSortColumn{},
                                                             input.get_sliced_child(stream),
                                                             output_offset->view(),
                                                             column_order,
                                                             null_precedence,
                                                             stream,
                                                             mr);

  auto null_mask = cudf::detail::copy_bitmask(input.parent(), stream, mr);

  // Assemble list column & return
  return make_lists_column(input.size(),
                           std::move(output_offset),
                           std::move(output_child),
                           input.null_count(),
                           std::move(null_mask),
                           stream,
                           mr);
}

std::unique_ptr<column> stable_sort_lists(lists_column_view const& input,
                                          order column_order,
                                          null_order null_precedence,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) { return empty_like(input.parent()); }

  auto output_offset = make_numeric_column(
    input.offsets().type(), input.size() + 1, mask_state::UNALLOCATED, stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    input.offsets_begin(),
                    input.offsets_end(),
                    output_offset->mutable_view().template begin<size_type>(),
                    [first = input.offsets_begin()] __device__(auto offset_index) {
                      return offset_index - *first;
                    });

  auto const child              = input.get_sliced_child(stream);
  auto const sorted_child_table = stable_segmented_sort_by_key(table_view{{child}},
                                                               table_view{{child}},
                                                               output_offset->view(),
                                                               {column_order},
                                                               {null_precedence},
                                                               stream,
                                                               mr);

  return make_lists_column(input.size(),
                           std::move(output_offset),
                           std::move(sorted_child_table->release().front()),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr),
                           stream,
                           mr);
}
}  // namespace detail

std::unique_ptr<column> sort_lists(lists_column_view const& input,
                                   order column_order,
                                   null_order null_precedence,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::sort_lists(input, column_order, null_precedence, cudf::default_stream_value, mr);
}

std::unique_ptr<column> stable_sort_lists(lists_column_view const& input,
                                          order column_order,
                                          null_order null_precedence,
                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::stable_sort_lists(
    input, column_order, null_precedence, cudf::default_stream_value, mr);
}

}  // namespace lists
}  // namespace cudf
