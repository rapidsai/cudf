/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <utilities/error_utils.hpp>

namespace cudf {
namespace exp {

// Copy the columns from another table
table::table(table const& other) : _num_rows{other.num_rows()} {
  _columns.reserve(other._columns.size());
  for (auto const& c : other._columns) {
    _columns.emplace_back(std::make_unique<column>(*c));
  }
}

// Move the contents of a vector `unique_ptr<column>`
table::table(std::vector<std::unique_ptr<column>>&& columns)
    : _columns{std::move(columns)} {
  CUDF_EXPECTS(columns.size() > 0, "Invalid number of columns");
  _num_rows = columns[0]->size();
  for (auto const& c : _columns) {
    CUDF_EXPECTS(c->size() == num_rows(), "Column size mismatch.");
  }
}

// Copy the contents of a `table_view`
table::table(table_view view, cudaStream_t stream,
             rmm::mr::device_memory_resource* mr) {
  _columns.reserve(view.num_columns());
  for (auto const& c : view) {
    _columns.emplace_back(std::make_unique<column>(c, stream, mr));
  }
}

// Create immutable view
table_view table::view() const {
  std::vector<column_view> views;
  views.reserve(_columns.size());
  for (auto const& c : _columns) {
    views.push_back(c->view());
  }
  return table_view{views};
}

// Create mutable view
mutable_table_view table::mutable_view() {
  std::vector<mutable_column_view> views;
  views.reserve(_columns.size());
  for (auto const& c : _columns) {
    views.push_back(c->mutable_view());
  }
  return mutable_table_view{views};
}

// Release ownership of columns
std::vector<std::unique_ptr<column>> table::release() {
  _num_rows = 0;
  return std::move(_columns);
}

}  // namespace exp
}  // namespace cudf
