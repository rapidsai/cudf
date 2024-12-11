/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/null_mask.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {

structs_column_view::structs_column_view(column_view const& rhs) : column_view{rhs}
{
  CUDF_EXPECTS(type().id() == type_id::STRUCT, "structs_column_view only supports struct columns");
}

column_view structs_column_view::parent() const { return *this; }

column_view structs_column_view::get_sliced_child(int index, rmm::cuda_stream_view stream) const
{
  std::vector<column_view> children;
  children.reserve(child(index).num_children());
  for (size_type i = 0; i < child(index).num_children(); i++) {
    children.push_back(child(index).child(i));
  }

  return column_view{
    child(index).type(),
    size(),
    child(index).head<uint8_t>(),
    child(index).null_mask(),
    child(index).null_count()
      ? cudf::detail::null_count(child(index).null_mask(), offset(), offset() + size(), stream)
      : 0,
    offset(),
    children};
}

}  // namespace cudf
