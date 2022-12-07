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
#include <cudf/utilities/error.hpp>

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
  if (input.is_empty()) { return empty_like(input.parent()); }

  auto const child = input.get_sliced_child(stream);

  // The labels are also zero-based list indices.
  auto const labels = generate_labels(input, child.size(), stream);

  // Build a gather map to copy the output list elements.
  auto gather_map = rmm::device_uvector<size_type>(child.size(), stream);
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::counting_iterator<size_type>(0),
                     child.size(),
                     [d_offsets           = input.offsets_begin(),
                      list_indices        = labels->view().begin<size_type>(),
                      new_element_indices = gather_map.begin()] __device__(auto const idx) {
                       // The first offset value, used for zero-normalizing offsets.
                       auto const first_offset  = *d_offsets;
                       auto const list_idx      = list_indices[idx];
                       auto const begin_offset  = d_offsets[list_idx] - first_offset;
                       auto const end_offset    = d_offsets[list_idx + 1] - first_offset;
                       new_element_indices[idx] = begin_offset + (end_offset - idx - 1);
                     });

  auto reversed_child_table =
    cudf::detail::gather(table_view{{child}},
                         device_span<size_type const>{gather_map.data(), gather_map.size()},
                         out_of_bounds_policy::DONT_CHECK,
                         cudf::detail::negative_index_policy::NOT_ALLOWED,
                         stream,
                         mr);
  auto out_child   = std::move(reversed_child_table->release().front());
  auto out_offsets = get_normalized_offsets(input, stream, mr);

  return cudf::make_lists_column(input.size(),
                                 std::move(out_offsets),
                                 std::move(out_child),
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
