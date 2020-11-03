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
  // all inputs must be structs
  CUDF_EXPECTS(
    std::all_of(struct_cols.begin(),
                struct_cols.end(),
                [](column_view const& col) { return col.type().id() == type_id::STRUCT; }),
    "Encountered non-struct input to extract_ordered_struct_children");

  std::vector<std::vector<column_view>> result;
  result.reserve(struct_cols[0].num_children());

  auto child_index = thrust::make_counting_iterator(0);
  std::transform(child_index,
                 child_index + struct_cols[0].num_children(),
                 std::back_inserter(result),
                 [&struct_cols](int i) {
                   std::vector<column_view> children;

                   // extract the i'th child from each input column
                   auto col_index = thrust::make_counting_iterator(0);
                   std::transform(
                     col_index,
                     col_index + struct_cols.size(),
                     std::back_inserter(children),
                     [&struct_cols, i](int col_index) {
                       structs_column_view scv(struct_cols[col_index]);

                       // all inputs must have the same # of children and they must all be of the
                       // same type.
                       CUDF_EXPECTS(struct_cols[0].num_children() == scv.num_children(),
                                    "Mismatch in number of children during struct concatenate");
                       CUDF_EXPECTS(struct_cols[0].child(i).type() == scv.child(i).type(),
                                    "Mismatch in child types during struct concatenate");
                       return scv.get_sliced_child(i);
                     });
                   return std::move(children);
                 });

  return std::move(result);
}

}  // namespace detail
}  // namespace structs
}  // namespace cudf
