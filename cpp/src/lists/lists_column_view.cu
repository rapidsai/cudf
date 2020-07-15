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

#include <cudf/column/column_device_view.cuh>
#include <cudf/lists/list_view.cuh>
#include <cudf/lists/lists_column_view.hpp>

namespace cudf {

lists_column_view::lists_column_view(column_view const& lists_column) : column_view(lists_column)
{
  CUDF_EXPECTS(type().id() == type_id::LIST, "lists_column_view only supports lists");
}

column_view lists_column_view::parent() const { return static_cast<column_view>(*this); }

column_view lists_column_view::offsets() const
{
  CUDF_EXPECTS(num_children() == 2, "lists column has an incorrect number of children");
  return column_view::child(offsets_column_index);
}

column_view lists_column_view::child() const
{
  CUDF_EXPECTS(num_children() == 2, "lists column has an incorrect number of children");
  return column_view::child(child_column_index);
}

}  // namespace cudf
