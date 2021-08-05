/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/copying.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/strings/detail/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace strings {
namespace detail {

std::unique_ptr<cudf::column> copy_slice(strings_column_view const& strings,
                                         size_type start,
                                         size_type end,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  if (strings.is_empty()) return make_empty_column(data_type{type_id::STRING});
  if (end < 0 || end > strings.size()) end = strings.size();
  CUDF_EXPECTS(((start >= 0) && (start < end)), "Invalid start parameter value.");
  auto const strings_count  = end - start;
  auto const offsets_offset = start + strings.offset();

  // slice the offsets child column
  auto offsets_column = std::make_unique<cudf::column>(
    cudf::slice(strings.offsets(), {offsets_offset, offsets_offset + strings_count + 1}).front(),
    stream,
    mr);
  auto const chars_offset =
    offsets_offset == 0 ? 0 : cudf::detail::get_value<int32_t>(offsets_column->view(), 0, stream);
  if (chars_offset > 0) {
    // adjust the individual offset values only if needed
    auto d_offsets = offsets_column->mutable_view();
    thrust::transform(rmm::exec_policy(stream),
                      d_offsets.begin<int32_t>(),
                      d_offsets.end<int32_t>(),
                      d_offsets.begin<int32_t>(),
                      [chars_offset] __device__(auto offset) { return offset - chars_offset; });
  }

  // slice the chars child column
  auto const data_size =
    cudf::detail::get_value<int32_t>(offsets_column->view(), strings_count, stream);
  auto chars_column = std::make_unique<cudf::column>(
    cudf::slice(strings.chars(), {chars_offset, chars_offset + data_size}).front(), stream, mr);

  // slice the null mask
  auto null_mask = cudf::detail::copy_bitmask(
    strings.null_mask(), offsets_offset, offsets_offset + strings_count, stream, mr);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             UNKNOWN_NULL_COUNT,
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
