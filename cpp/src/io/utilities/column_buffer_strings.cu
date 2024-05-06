/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "column_buffer.hpp"

#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf::io::detail {

std::unique_ptr<column> cudf::io::detail::inline_column_buffer::make_string_column_impl(
  rmm::cuda_stream_view stream)
{
  // if the size of _string_data is over the threshold for 64bit size_type, _data will contain
  // sizes rather than offsets. need special handling for that case.
  auto const threshold = static_cast<size_t>(strings::detail::get_offset64_threshold());
  if (_string_data.size() > threshold) {
    if (not strings::detail::is_large_strings_enabled()) {
      CUDF_FAIL("String column exceeds the column size limit", std::overflow_error);
    }
    // create new offsets
    auto const offsets_ptr = static_cast<size_type*>(_data.data());
    auto offsets_col       = make_numeric_column(
      data_type{type_id::INT64}, size + 1, mask_state::UNALLOCATED, stream, _mr);
    auto d_offsets64 = offsets_col->mutable_view().template data<int64_t>();
    // it's safe to call with size + 1 because _data is also sized that large
    cudf::detail::sizes_to_offsets(offsets_ptr, offsets_ptr + size + 1, d_offsets64, stream);
    return make_strings_column(
      size, std::move(offsets_col), std::move(_string_data), null_count(), std::move(_null_mask));
  } else {
    // no need for copies, just transfer ownership of the data_buffers to the columns
    auto offsets_col = std::make_unique<column>(
      data_type{type_to_id<size_type>()}, size + 1, std::move(_data), rmm::device_buffer{}, 0);
    return make_strings_column(
      size, std::move(offsets_col), std::move(_string_data), null_count(), std::move(_null_mask));
  }
}

}  // namespace cudf::io::detail
