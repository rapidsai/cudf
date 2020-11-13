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
#include <cudf/detail/copy_range.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/lists/lists_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <iostream>

namespace cudf {
namespace lists {
namespace detail {

// New lists column from a subset of a lists_column_view
std::unique_ptr<cudf::column> copy_slice(lists_column_view const& lists,
                                         size_type start,
                                         size_type end,
                                         cudaStream_t stream,
                                         rmm::mr::device_memory_resource* mr)
{
  if (lists.is_empty()) { return cudf::empty_like(lists.parent()); }
  if (end < 0 || end > lists.size()) end = lists.size();
  CUDF_EXPECTS(((start >= 0) && (start < end)), "Invalid slice range.");
  auto lists_count   = end - start;
  auto offsets_count = lists_count + 1;  // num_offsets always 1 more than num_lists

  // Account for the offset of the view:
  start += lists.offset();
  end += lists.offset();

  // Offsets at the beginning and end of the slice:
  auto offsets_data = lists.offsets().data<cudf::size_type>();
  auto start_offset = cudf::detail::get_value<size_type>(lists.offsets(), start, stream);
  auto end_offset   = cudf::detail::get_value<size_type>(lists.offsets(), end, stream);

  rmm::device_uvector<cudf::size_type> out_offsets(offsets_count, stream);
  auto execpol = rmm::exec_policy(stream);

  // Compute the offsets column of the result:
  thrust::transform(
    execpol->on(stream),
    offsets_data + start,
    offsets_data + end + 1,  // size of offsets column is 1 greater than slice length
    out_offsets.data(),
    [start_offset] __device__(cudf::size_type i) { return i - start_offset; });
  auto offsets = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32}, offsets_count, out_offsets.release());

  // Compute the child column of the result.
  // If the child of this lists column is itself a lists column, we call copy_slice() on it.
  // Otherwise, it is a column of the leaf type, so we call slice() on it and copy the resulting
  // view into a cudf::column:
  auto child =
    (lists.child().type() == cudf::data_type{type_id::LIST})
      ? copy_slice(lists_column_view(lists.child()), start_offset, end_offset, stream, mr)
      : std::make_unique<cudf::column>(
          cudf::detail::slice(lists.child(), {start_offset, end_offset}, stream).front());

  // Compute the null mask of the result:
  auto null_mask =
    cudf::detail::copy_bitmask(lists.null_mask(), start, end, rmm::cuda_stream_view{stream}, mr);

  return make_lists_column(lists_count,
                           std::move(offsets),
                           std::move(child),
                           cudf::UNKNOWN_NULL_COUNT,
                           std::move(null_mask));
}
}  // namespace detail
}  // namespace lists
}  // namespace cudf
