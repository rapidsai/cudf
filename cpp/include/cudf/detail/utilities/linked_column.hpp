/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <memory>
#include <vector>

namespace cudf::detail {

struct linked_column_view;

using LinkedColPtr    = std::shared_ptr<linked_column_view>;
using LinkedColVector = std::vector<LinkedColPtr>;

/**
 * @brief A column_view class with pointer to parent's column_view
 */
struct linked_column_view : public column_view_base {
  linked_column_view(linked_column_view const&)            = delete;
  linked_column_view& operator=(linked_column_view const&) = delete;

  /**
   * @brief Construct from column_view
   *
   * @param col column_view to wrap
   */
  linked_column_view(column_view const& col);

  /**
   * @brief Construct from column_view with it's parent
   *
   * @param parent Pointer to the column_view's parent column_view
   * @param col column_view to wrap
   */
  linked_column_view(linked_column_view* parent, column_view const& col);

  /**
   * @brief Conversion operator to cast this instance to it's column_view
   */
  operator column_view() const;

  linked_column_view* parent;  ///< Pointer to parent of this column; nullptr if root
  LinkedColVector children;    ///< Vector of children of this instance
};

/**
 * @brief Converts all column_views of a table into linked_column_views
 *
 * @param table table of columns to convert
 * @return Vector of converted linked_column_views
 */
LinkedColVector table_to_linked_columns(table_view const& table);

}  // namespace cudf::detail
