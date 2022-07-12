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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

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

strings_column_view::offset_iterator strings_column_view::offsets_begin() const
{
  return offsets().begin<offset_type>() + offset();
}

strings_column_view::offset_iterator strings_column_view::offsets_end() const
{
  return offsets_begin() + size() + 1;
}

column_view strings_column_view::chars() const
{
  CUDF_EXPECTS(num_children() > 0, "strings column has no children");
  return child(chars_column_index);
}

size_type strings_column_view::chars_size() const noexcept
{
  if (size() == 0) return 0;
  return chars().size();
}

strings_column_view::chars_iterator strings_column_view::chars_begin() const
{
  return chars().begin<char>();
}

strings_column_view::chars_iterator strings_column_view::chars_end() const
{
  return chars_begin() + chars_size();
}

}  // namespace cudf
