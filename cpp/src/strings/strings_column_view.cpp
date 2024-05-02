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

#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
//
strings_column_view::strings_column_view(column_view strings_column) : column_view(strings_column)
{
  CUDF_EXPECTS(type().id() == type_id::STRING, "strings_column_view only supports strings");
}

column_view strings_column_view::parent() const { return static_cast<column_view>(*this); }

column_view strings_column_view::offsets() const
{
  CUDF_EXPECTS(num_children() > 0, "strings column has no children");
  return child(offsets_column_index);
}

int64_t strings_column_view::chars_size(rmm::cuda_stream_view stream) const noexcept
{
  if (size() == 0) { return 0L; }
  return cudf::strings::detail::get_offset_value(offsets(), offsets().size() - 1, stream);
}

strings_column_view::chars_iterator strings_column_view::chars_begin(rmm::cuda_stream_view) const
{
  return head<char>();
}

strings_column_view::chars_iterator strings_column_view::chars_end(
  rmm::cuda_stream_view stream) const
{
  return chars_begin(stream) + chars_size(stream);
}

}  // namespace cudf
