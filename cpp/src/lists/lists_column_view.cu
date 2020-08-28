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
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
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

column_view lists_column_view::get_sliced_child(cudaStream_t stream) const
{
  // if I have a positive offset, I need to slice my child
  if (offset() > 0) {
    // theoretically this function could always do this step and be correct, but get_value<>
    // actually hits the gpu so it's best to avoid it if possible.
    size_type child_offset_start = cudf::detail::get_value<size_type>(offsets(), offset(), stream);
    size_type child_offset_end =
      cudf::detail::get_value<size_type>(offsets(), offset() + size(), stream);
    return cudf::detail::slice(child(), {child_offset_start, child_offset_end}, stream).front();
  }

  // if I don't have a positive offset, but I am shorter than my offsets() would otherwise indicate,
  // I need to do a split and return the front.
  if (size() < offsets().size() - 1) {
    size_type child_offset = cudf::detail::get_value<size_type>(offsets(), size(), stream);
    return cudf::detail::slice(child(), {0, child_offset}, stream).front();
  }

  // otherwise just return the child directly
  return child();
}

}  // namespace cudf
