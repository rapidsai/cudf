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
#include <cudf/detail/gather.cuh>
#include <cudf/lists/drop_list_duplicates.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {

/**
 * @copydoc cudf::lists::drop_list_duplicates
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> drop_list_duplicates(lists_column_view const& lists_column,
                                             duplicate_keep_option keep,
                                             null_equality nulls_equal,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  if (lists_column.is_empty()) return cudf::empty_like(lists_column.parent());

    /*
     * Given input = { {1, 1, 2, 1, 3}, {4}, {5, 6, 6, 6, 5} }
     *  - if keep is KEEP_FIRST, the output will be { {1, 2, 3}, {4}, {5, 6} }
     *  - if keep is KEEP_LAST, the output will be  { {2, 1, 3}, {4}, {6, 5} }
     *  - if keep is KEEP_NONE, the output will be  { {2, 3}, {4}, {} }
     *
     * 1. Generate ordered indices for each list element
     *   ordered_indices = { {0, 1, 2, 3, 4}, {5}, {6, 7, 8, 9, 10} }
     *
     * 2. Call sort_list on the lists and indices using the list entries as keys
     *   sorted_lists = { {1, 1, 1, 2, 3}, {4}, {5, 5, 6, 6, 6} }, and
     *   sorted_indices = { {0, 1, 3, 2, 4}, {5}, {6, 10, 7, 8, 9} }
     *
     * 3. Remove list indices if the list entries are duplicated
     *  - with keep is KEEP_FIRST: sorted_unique_indices = { {0, 2, 4}, {5}, {6, 7} }
     *  - with keep is KEEP_LAST:  sorted_unique_indices = { {3, 2, 4}, {5}, {10, 9} }
     *  - with keep is KEEP_NONE:  sorted_unique_indices = { {2, 4}, {5}, {} }
     *
     * 4. Call sort_lists on the sorted_unique_indices to obtain the final list indices
     *  - with keep is KEEP_FIRST: sorted_unique_indices = { {0, 2, 4}, {5}, {6, 7} }
     *  - with keep is KEEP_LAST:  sorted_unique_indices = { {2, 3, 4}, {5}, {9, 10} }
     *  - with keep is KEEP_NONE:  sorted_unique_indices = { {2, 4}, {5}, {} }
     *
     * 5. Gather list entries using the sorted_unique_indices as gather map
     *   (remember to deal with null elements)
     *
     *  Corner cases:
     *   - null entries in a list: depending on the nulls_equal policy, if it is set to EQUAL then
     * only one null entry is kept. Which null entry to keep---it is specified by the keep policy.
     *   - null rows: just return null rows, nothing changes.
     *   - NaN entries in a list: NaNs should be treated as equal, thus only one NaN value is kept.
     *  Again, which value to keep depends on the keep policy.
     *   - Nested types are not supported---the function should throw logic_error.
     */

#if 0
  auto const offsets_column = lists_column.offsets();

  // create a column_view with attributes of the parent and data from the offsets
  column_view annotated_offsets(data_type{type_id::INT32},
                                lists_column.size() + 1,
                                offsets_column.data<int32_t>(),
                                lists_column.null_mask(),
                                lists_column.null_count(),
                                lists_column.offset());

  // create a gather map for extracting elements from the child column
  auto gather_map = make_fixed_width_column(
    data_type{type_id::INT32}, annotated_offsets.size() - 1, mask_state::UNALLOCATED, stream);
  auto d_gather_map       = gather_map->mutable_view().data<int32_t>();
  auto const child_column = lists_column.child();

  // build the gather map using the offsets and the provided index
  auto const d_column = column_device_view::create(annotated_offsets, stream);
  if (index < 0)
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(gather_map->size()),
                      d_gather_map,
                      map_index_fn<false>{*d_column, index, child_column.size()});
  else
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(gather_map->size()),
                      d_gather_map,
                      map_index_fn<true>{*d_column, index, child_column.size()});

  // Gather only the unique entries
  auto result = cudf::detail::gather(table_view({child_column}),
                                     d_gather_map,
                                     d_gather_map + gather_map->size(),
                                     out_of_bounds_policy::NULLIFY,  // nullify-out-of-bounds
                                     stream,
                                     mr)
                  ->release();

  // Set zero size for the null_mask if there is no null element
  if (result.front()->null_count() == 0)
    result.front()->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);

  return std::unique_ptr<column>(std::move(result.front()));
#endif

  return empty_like(lists_column.parent());
}

}  // namespace detail

/**
 * @copydoc cudf::lists::drop_list_duplicates
 */
std::unique_ptr<column> drop_list_duplicates(lists_column_view const& lists_column,
                                             duplicate_keep_option keep,
                                             null_equality nulls_equal,
                                             rmm::mr::device_memory_resource* mr)
{
  return detail::drop_list_duplicates(
    lists_column, keep, nulls_equal, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
