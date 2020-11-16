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

#include <thrust/iterator/counting_iterator.h>

#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace structs {
namespace detail {

/**
 * @copydoc cudf::structs::detail::extract_ordered_struct_children
 *
 */
std::vector<std::vector<column_view>> extract_ordered_struct_children(
  std::vector<column_view> const& struct_cols)
{
  auto const num_children = struct_cols[0].num_children();
  auto const num_cols     = static_cast<size_type>(struct_cols.size());

  std::vector<std::vector<column_view>> result;
  result.reserve(num_children);

  for (size_type child_index = 0; child_index < num_children; child_index++) {
    std::vector<column_view> children;
    children.reserve(num_cols);
    for (size_type col_index = 0; col_index < num_cols; col_index++) {
      structs_column_view scv(struct_cols[col_index]);

      // all inputs must have the same # of children and they must all be of the
      // same type.
      CUDF_EXPECTS(struct_cols[0].num_children() == scv.num_children(),
                   "Mismatch in number of children during struct concatenate");
      CUDF_EXPECTS(struct_cols[0].child(child_index).type() == scv.child(child_index).type(),
                   "Mismatch in child types during struct concatenate");
      children.push_back(scv.get_sliced_child(child_index));
    }

    result.push_back(std::move(children));
  }

  return result;
}

}  // namespace detail
}  // namespace structs
}  // namespace cudf
