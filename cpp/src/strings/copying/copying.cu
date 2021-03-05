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
#include <cudf/detail/gather.cuh>
#include <cudf/detail/get_value.cuh>
#include <cudf/strings/copying.hpp>
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
  if (strings.is_empty()) return make_empty_strings_column(stream, mr);
  if (end < 0 || end > strings.size()) end = strings.size();
  CUDF_EXPECTS(((start >= 0) && (start < end)), "Invalid start parameter value.");
  size_type const strings_count = end - start;
  if (start == 0 && strings.offset() == 0) {
    // sliced at the beginning and copying everything, so no need to gather
    auto offsets_column = std::make_unique<cudf::column>(
      cudf::slice(strings.offsets(), {0, strings_count + 1}).front(), stream, mr);
    auto data_size =
      cudf::detail::get_value<int32_t>(offsets_column->view(), strings_count, stream);
    auto chars_column = std::make_unique<cudf::column>(
      cudf::slice(strings.chars(), {0, data_size}).front(), stream, mr);
    auto null_mask = cudf::detail::copy_bitmask(strings.null_mask(), 0, strings_count, stream, mr);
    return make_strings_column(strings_count,
                               std::move(offsets_column),
                               std::move(chars_column),
                               UNKNOWN_NULL_COUNT,
                               std::move(null_mask),
                               stream,
                               mr);
  }

  // do the full gather instead
  // TODO: it may be faster to just copy sliced child columns and then fixup the offset values
  auto sliced_table = cudf::detail::gather(table_view{{strings.parent()}},
                                           thrust::counting_iterator<size_type>(start),
                                           thrust::counting_iterator<size_type>(end),
                                           cudf::out_of_bounds_policy::DONT_CHECK,
                                           stream,
                                           mr)
                        ->release();
  std::unique_ptr<column> output_column(std::move(sliced_table.front()));
  if (output_column->null_count() == 0)
    output_column->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);
  return output_column;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
