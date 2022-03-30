/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/table/table_view.hpp>

namespace cudf::detail {

struct linked_column_view;

using LinkedColPtr    = std::shared_ptr<linked_column_view>;
using LinkedColVector = std::vector<LinkedColPtr>;

/**
 * @brief column_view with the added member pointer to the parent of this column.
 *
 */
struct linked_column_view : public column_view_base {
  linked_column_view(column_view const& col) : linked_column_view(nullptr, col) {}

  linked_column_view(linked_column_view* parent, column_view const& col)
    : column_view_base(col), parent(parent)
  {
    children.reserve(col.num_children());
    std::transform(
      col.child_begin(), col.child_end(), std::back_inserter(children), [&](column_view const& c) {
        return std::make_shared<linked_column_view>(this, c);
      });
  }

  operator column_view() const
  {
    std::vector<column_view> column_view_children;
    column_view_children.reserve(children.size());
    std::transform(children.begin(),
                   children.end(),
                   std::back_inserter(column_view_children),
                   [](auto const& c) { return *c; });
    return column_view(this->type(),
                       this->size(),
                       this->head(),
                       this->null_mask(),
                       UNKNOWN_NULL_COUNT,
                       this->offset(),
                       std::move(column_view_children));
  }

  linked_column_view* parent;  //!< Pointer to parent of this column. Nullptr if root
  LinkedColVector children;
};

/**
 * @brief Converts all column_views of a table into linked_column_views
 *
 * @param table table of columns to convert
 * @return Vector of converted linked_column_views
 */
inline LinkedColVector table_to_linked_columns(table_view const& table)
{
  LinkedColVector result;
  result.reserve(table.num_columns());
  std::transform(table.begin(), table.end(), std::back_inserter(result), [&](column_view const& c) {
    return std::make_shared<linked_column_view>(c);
  });

  return result;
}

}  // namespace cudf::detail