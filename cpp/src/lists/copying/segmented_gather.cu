/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/detail/gather.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/lists/detail/gather.cuh>

#include <rmm/cuda_stream_view.hpp>
#include "cudf/stream_compaction.hpp"
#include "thrust/count.h"
#include "thrust/iterator/counting_iterator.h"

#include <thrust/binary_search.h>

namespace cudf {
namespace lists {
namespace detail {

std::unique_ptr<column> segmented_gather(column_view const& list_column,
                                         column_view const& gather_map_list,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  auto const value_column = lists_column_view{list_column};
  auto const gather_map   = lists_column_view{gather_map_list};
  CUDF_EXPECTS(is_index_type(gather_map.child().type()), "Gather map should be list of index type");
  CUDF_EXPECTS(gather_map.has_nulls() == false, "gather_map contains nulls");
  std::cout << value_column.size() << "==" << gather_map.size() << "\n";
  CUDF_EXPECTS(value_column.size() == gather_map.size(),
               "Gather map and list column should be same size");
  auto const gather_map_size = gather_map.get_sliced_child(stream).size();
  auto value_offsets         = value_column.offsets().begin<size_type>();
  auto gather_offsets        = gather_map.offsets().begin<size_type>();

  // Flattened gather indices
  auto child_gather_index =
    make_numeric_column(data_type{type_to_id<size_type>()}, gather_map_size);
  auto child_gather_index_begin = child_gather_index->mutable_view().begin<size_type>();
  auto child_gather_index_end   = child_gather_index->mutable_view().end<size_type>();

  thrust::upper_bound(rmm::exec_policy(stream),
                      gather_map.offsets().begin<size_type>() + 1,
                      gather_map.offsets().end<size_type>(),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(gather_map_size),
                      child_gather_index_begin);

  thrust::device_ptr<size_type> pt(child_gather_index_begin);
#define DEBUG_SEG_GATHER 1
  if (DEBUG_SEG_GATHER) {
    printf("\nv1:");
    for (auto i : thrust::host_vector<size_type>(pt, pt + gather_map_size)) printf("%d,", i);
    printf("\n");
  }

  auto map_begin =
    cudf::detail::indexalator_factory::make_input_iterator(gather_map.get_sliced_child(stream));
  thrust::transform(rmm::exec_policy(stream),
                    child_gather_index_begin,
                    child_gather_index_end,
                    map_begin,
                    child_gather_index_begin,
                    [value_offsets, gather_offsets] __device__(size_type offset_idx,
                                                               size_type sub_index) -> size_type {
                      auto list_size = value_offsets[offset_idx + 1] - value_offsets[offset_idx];
                      auto wrapped_sub_index = (sub_index % list_size + list_size) % list_size;
                      return value_offsets[offset_idx] + wrapped_sub_index;
                    });

  if (DEBUG_SEG_GATHER) {
    printf("\nv3:");
    for (auto i : thrust::host_vector<size_type>(pt, pt + gather_map_size)) printf("%d,", i);
    printf("\n");
  }

  // Call gather on child of value_column
  auto child_table = cudf::detail::gather(table_view({value_column.get_sliced_child(stream)}),
                                          *child_gather_index,
                                          out_of_bounds_policy::DONT_CHECK,
                                          cudf::detail::negative_index_policy::NOT_ALLOWED,
                                          stream,
                                          mr);
  auto child       = std::move(child_table->release().front());

  // Create list offsets from gather_map.
  auto output_offset =
    cudf::allocate_like(gather_map.offsets(), mask_allocation_policy::RETAIN, mr);
  auto output_offset_view = output_offset->mutable_view();
  cudf::copy_range_in_place(
    gather_map.offsets(), output_offset_view, 0, gather_map.offsets().size(), 0);
  // Assemble list column & return
  auto null_mask       = cudf::detail::copy_bitmask(list_column, stream, mr);
  size_type null_count = gather_map.null_count();
  return make_lists_column(gather_map.size(),
                           std::move(output_offset),
                           std::move(child),
                           null_count,
                           std::move(null_mask));
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
