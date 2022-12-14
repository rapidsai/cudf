/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "utilities.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/reverse.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf::lists {
namespace detail {

std::unique_ptr<column> reverse(lists_column_view const& input,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) { return cudf::empty_like(input.parent()); }

  auto const child = input.get_sliced_child(stream);

  // The labels are also a map from each list element to its corresponding zero-based list index.
  auto const labels = generate_labels(input, child.size(), stream);

  // The offsets of the output lists column.
  auto out_offsets = get_normalized_offsets(input, stream, mr);

  // Build a gather map to copy the output list elements from the input list elements.
  auto gather_map = rmm::device_uvector<size_type>(child.size(), stream);

  // Build a segmented reversed order for the child column.
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::counting_iterator<size_type>(0),
                     child.size(),
                     [list_offsets = out_offsets->view().begin<offset_type>(),
                      list_indices = labels->view().begin<size_type>(),
                      gather_map   = gather_map.begin()] __device__(auto const idx) {
                       auto const list_idx     = list_indices[idx];
                       auto const begin_offset = list_offsets[list_idx];
                       auto const end_offset   = list_offsets[list_idx + 1];

                       // Reverse the order of elements within each list.
                       gather_map[idx] = begin_offset + (end_offset - idx - 1);
                     });

  auto child_segmented_reversed =
    cudf::detail::gather(table_view{{child}},
                         device_span<size_type const>{gather_map.data(), gather_map.size()},
                         out_of_bounds_policy::DONT_CHECK,
                         cudf::detail::negative_index_policy::NOT_ALLOWED,
                         stream,
                         mr);

  return cudf::make_lists_column(input.size(),
                                 std::move(out_offsets),
                                 std::move(child_segmented_reversed->release().front()),
                                 input.null_count(),
                                 cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                 stream,
                                 mr);
}

}  // namespace detail

std::unique_ptr<column> reverse(lists_column_view const& input, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::reverse(input, cudf::get_default_stream(), mr);
}

}  // namespace cudf::lists
