/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/strings/detail/copying.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/functional>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

std::unique_ptr<cudf::column> copy_slice(strings_column_view const& input,
                                         size_type start,
                                         size_type end,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }
  CUDF_EXPECTS(((start >= 0) && (start < end)), "Invalid start parameter value.");
  auto const strings_count  = end - start;
  auto const offsets_offset = start + input.offset();

  // slice the offsets child column
  auto offsets_column = std::make_unique<cudf::column>(
    cudf::detail::slice(
      input.offsets(), {offsets_offset, offsets_offset + strings_count + 1}, stream)
      .front(),
    stream,
    mr);
  auto const chars_offset =
    offsets_offset == 0 ? 0L : get_offset_value(offsets_column->view(), 0, stream);
  if (chars_offset > 0) {
    // adjust the individual offset values only if needed
    auto d_offsets =
      cudf::detail::offsetalator_factory::make_output_iterator(offsets_column->mutable_view());
    auto input_offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), offsets_offset);
    thrust::transform(rmm::exec_policy(stream),
                      input_offsets,
                      input_offsets + offsets_column->size(),
                      d_offsets,
                      cuda::proclaim_return_type<int64_t>(
                        [chars_offset] __device__(auto offset) { return offset - chars_offset; }));
  }

  // slice the chars child column
  auto const data_size =
    static_cast<std::size_t>(get_offset_value(offsets_column->view(), strings_count, stream));
  auto chars_buffer =
    rmm::device_buffer{input.chars_begin(stream) + chars_offset, data_size, stream, mr};

  // slice the null mask
  auto null_mask = cudf::detail::copy_bitmask(
    input.null_mask(), offsets_offset, offsets_offset + strings_count, stream, mr);

  auto null_count = cudf::detail::null_count(
    static_cast<bitmask_type const*>(null_mask.data()), 0, strings_count, stream);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_buffer),
                             null_count,
                             std::move(null_mask));
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
