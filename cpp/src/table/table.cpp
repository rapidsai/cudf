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
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace experimental {

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
  if(num_columns() > 0) {
    for (auto const& c : _columns) {
      CUDF_EXPECTS(c, "Unexpected null column");
      CUDF_EXPECTS(c->size() == _columns.front()->size(), "Column size mismatch.");
    }
    _num_rows = _columns.front()->size();
  } else {
    _num_rows = 0;
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

// Returns a table_view with set of specified columns
table_view table::select(std::vector<cudf::size_type> const& column_indices) const {
    std::vector<column_view> columns;
    for (auto index : column_indices) {
      columns.push_back(_columns.at(index)->view());
    }
    return table_view(columns);
}

std::unique_ptr<table>
concatenate(std::vector<table_view> const& tables_to_concat,
            rmm::mr::device_memory_resource *mr, cudaStream_t stream) {
  if (tables_to_concat.size() == 0) { return std::make_unique<table>(); }

  size_type number_of_cols = tables_to_concat.front().num_columns();
  CUDF_EXPECTS(std::all_of(tables_to_concat.begin(), tables_to_concat.end(),
        [number_of_cols](auto const& t) { return t.num_columns() == number_of_cols; }),
      "Mismatch in table number of columns to concatenate.");

  std::vector<std::unique_ptr<column>> concat_columns;
  for (size_type i = 0; i < number_of_cols; ++i) {
    std::vector<column_view> cols;
    for (auto &t : tables_to_concat) {
      cols.emplace_back(t.column(i));
    }
    concat_columns.emplace_back(concatenate(cols, mr, stream));
  }
  return std::make_unique<table>(std::move(concat_columns));
}

}  // namespace experimental
}  // namespace cudf
