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
#include <cudf/detail/gather.cuh>
#include <cudf/lists/extract.hpp>

namespace cudf {
namespace lists {
namespace detail {

/**
 * @copydoc cudf::lists::extract_list_element
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> extract_list_element(lists_column_view lists_column,
                                             size_type index,
                                             cudaStream_t stream,
                                             rmm::mr::device_memory_resource* mr)
{
  if (lists_column.size() == 0) return empty_like(lists_column.parent());
  auto offsets_column = lists_column.offsets();

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
  auto d_gather_map = gather_map->mutable_view().data<int32_t>();
  auto child_column = lists_column.child();
  CUDF_EXPECTS(child_column.type().id() != type_id::LIST,
               "Nested lists not yet supported in extract_list_element");

  // build the gather map using the offsets and the provided index
  auto child_size = child_column.size();  // used for out-of-bounds condition
  auto d_column   = column_device_view::create(annotated_offsets, stream);
  auto d_offsets  = *d_column;
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(gather_map->size()),
                    d_gather_map,
                    [d_offsets, index, child_size] __device__(auto idx) {
                      if (d_offsets.is_null(idx)) return child_size;
                      auto offset = d_offsets.element<int32_t>(idx);
                      auto length = d_offsets.element<int32_t>(idx + 1) - offset;
                      return index < length ? index + offset : child_size;
                    });

  // call gather on the child column
  auto result = cudf::detail::gather(table_view({child_column}),
                                     d_gather_map,
                                     d_gather_map + gather_map->size(),
                                     true,  // nullify-out-of-bounds
                                     mr,
                                     stream)
                  ->release();
  return std::unique_ptr<column>(std::move(result.front()));
}

}  // namespace detail

/**
 * @copydoc cudf::lists::extract_list_element
 */
std::unique_ptr<column> extract_list_element(lists_column_view const& lists_column,
                                             size_type index,
                                             rmm::mr::device_memory_resource* mr)
{
  return detail::extract_list_element(lists_column, index, 0, mr);
}

}  // namespace lists
}  // namespace cudf
