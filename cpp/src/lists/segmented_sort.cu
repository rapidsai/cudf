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
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>

#include <cub/device/device_segmented_radix_sort.cuh>

namespace cudf {
namespace lists {
namespace detail {

struct SegmentedSortColumn {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& child,
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
  auto output_child = type_dispatcher(input.child().type(),
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
                           std::move(null_mask));
}
}  // namespace detail

std::unique_ptr<column> sort_lists(lists_column_view const& input,
                                   order column_order,
                                   null_order null_precedence,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::sort_lists(input, column_order, null_precedence, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
