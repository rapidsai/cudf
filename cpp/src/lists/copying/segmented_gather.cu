/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cudf/detail/copy_range.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/lists/detail/gather.cuh>
#include <cudf/utilities/default_stream.hpp>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace lists {
namespace detail {

std::unique_ptr<column> segmented_gather(lists_column_view const& value_column,
                                         lists_column_view const& gather_map,
                                         out_of_bounds_policy bounds_policy,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(is_index_type(gather_map.child().type()),
               "Gather map should be list column of index type");
  CUDF_EXPECTS(!gather_map.has_nulls(), "Gather map contains nulls");
  CUDF_EXPECTS(value_column.size() == gather_map.size(),
               "Gather map and list column should be same size");

  auto const gather_map_sliced_child = gather_map.get_sliced_child(stream);
  auto const gather_map_size         = gather_map_sliced_child.size();
  auto const gather_index_begin      = gather_map.offsets_begin() + 1;
  auto const gather_index_end        = gather_map.offsets_end();
  auto const value_offsets           = value_column.offsets_begin();
  auto const value_device_view       = column_device_view::create(value_column.parent(), stream);
  auto const map_begin =
    cudf::detail::indexalator_factory::make_input_iterator(gather_map_sliced_child);
  auto const out_of_bounds = [] __device__(auto const index, auto const list_size) {
    return index >= list_size || (index < 0 && -index > list_size);
  };

  // Calculate Flattened gather indices  (value_offset[row]+sub_index
  auto transformer = [values_lists_view = *value_device_view,
                      value_offsets,
                      map_begin,
                      gather_index_begin,
                      gather_index_end,
                      bounds_policy,
                      out_of_bounds] __device__(size_type index) -> size_type {
    // Get each row's offset. (Each row is a list).
    auto offset_idx =
      thrust::upper_bound(
        thrust::seq, gather_index_begin, gather_index_end, gather_index_begin[-1] + index) -
      gather_index_begin;
    // Get each sub_index in list in each row of gather_map.
    auto sub_index    = map_begin[index];
    auto list_is_null = values_lists_view.is_null(offset_idx);
    auto list_size = list_is_null ? 0 : (value_offsets[offset_idx + 1] - value_offsets[offset_idx]);
    auto wrapped_sub_index  = sub_index < 0 ? sub_index + list_size : sub_index;
    auto constexpr null_idx = cuda::std::numeric_limits<cudf::size_type>::max();
    // Add sub_index to value_column offsets, to get gather indices of child of value_column
    return (bounds_policy == out_of_bounds_policy::NULLIFY && out_of_bounds(sub_index, list_size))
             ? null_idx
             : value_offsets[offset_idx] + wrapped_sub_index - value_offsets[0];
  };
  auto child_gather_index_begin = cudf::detail::make_counting_transform_iterator(0, transformer);

  // Call gather on child of value_column
  auto child_table = cudf::detail::gather(table_view({value_column.get_sliced_child(stream)}),
                                          child_gather_index_begin,
                                          child_gather_index_begin + gather_map_size,
                                          bounds_policy,
                                          stream,
                                          mr);
  auto child       = std::move(child_table->release().front());

  // Create list offsets from gather_map.
  auto output_offset = cudf::detail::allocate_like(
    gather_map.offsets(), gather_map.size() + 1, mask_allocation_policy::RETAIN, stream, mr);
  auto output_offset_view = output_offset->mutable_view();
  cudf::detail::copy_range_in_place(gather_map.offsets(),
                                    output_offset_view,
                                    gather_map.offset(),
                                    gather_map.offset() + output_offset_view.size(),
                                    0,
                                    stream);
  // Assemble list column & return
  auto null_mask       = cudf::detail::copy_bitmask(value_column.parent(), stream, mr);
  size_type null_count = value_column.null_count();
  return make_lists_column(gather_map.size(),
                           std::move(output_offset),
                           std::move(child),
                           null_count,
                           std::move(null_mask),
                           stream,
                           mr);
}

}  // namespace detail

std::unique_ptr<column> segmented_gather(lists_column_view const& source_column,
                                         lists_column_view const& gather_map_list,
                                         out_of_bounds_policy bounds_policy,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_gather(
    source_column, gather_map_list, bounds_policy, cudf::get_default_stream(), mr);
}

}  // namespace lists
}  // namespace cudf
