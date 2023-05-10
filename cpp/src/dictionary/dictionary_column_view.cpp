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

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {
//
dictionary_column_view::dictionary_column_view(column_view const& dictionary_column)
  : column_view(dictionary_column)
{
  CUDF_EXPECTS(type().id() == type_id::DICTIONARY32,
               "dictionary_column_view only supports DICTIONARY type");
  if (size() > 0) CUDF_EXPECTS(num_children() == 2, "dictionary column has no children");
}

column_view dictionary_column_view::parent() const noexcept
{
  return static_cast<column_view>(*this);
}

column_view dictionary_column_view::indices() const noexcept { return child(0); }

column_view dictionary_column_view::get_indices_annotated() const noexcept
{
  return column_view(
    indices().type(), size(), indices().head(), null_mask(), null_count(), offset());
}

column_view dictionary_column_view::keys() const noexcept { return child(1); }

size_type dictionary_column_view::keys_size() const noexcept
{
  return (size() == 0) ? 0 : keys().size();
}

data_type dictionary_column_view::keys_type() const noexcept
{
  return (size() == 0) ? data_type{type_id::EMPTY} : keys().type();
}

}  // namespace cudf
