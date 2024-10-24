/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {

// Copy the columns from another table
table::table(table const& other, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
  : _num_rows{other.num_rows()}
{
  CUDF_FUNC_RANGE();
  _columns.reserve(other._columns.size());
  for (auto const& c : other._columns) {
    _columns.emplace_back(std::make_unique<column>(*c, stream, mr));
  }
}

// Move the contents of a vector `unique_ptr<column>`
table::table(std::vector<std::unique_ptr<column>>&& columns) : _columns{std::move(columns)}
{
  if (num_columns() > 0) {
    for (auto const& c : _columns) {
      CUDF_EXPECTS(c, "Unexpected null column");
      CUDF_EXPECTS(c->size() == _columns.front()->size(),
                   "Column size mismatch: " + std::to_string(c->size()) +
                     " != " + std::to_string(_columns.front()->size()));
    }
    _num_rows = _columns.front()->size();
  } else {
    _num_rows = 0;
  }
}

// Copy the contents of a `table_view`
table::table(table_view view, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
  : _num_rows{view.num_rows()}
{
  CUDF_FUNC_RANGE();
  _columns.reserve(view.num_columns());
  for (auto const& c : view) {
    _columns.emplace_back(std::make_unique<column>(c, stream, mr));
  }
}

// Create immutable view
table_view table::view() const
{
  std::vector<column_view> views;
  views.reserve(_columns.size());
  for (auto const& c : _columns) {
    views.push_back(c->view());
  }
  return table_view{views};
}

// Create mutable view
mutable_table_view table::mutable_view()
{
  std::vector<mutable_column_view> views;
  views.reserve(_columns.size());
  for (auto const& c : _columns) {
    views.push_back(c->mutable_view());
  }
  return mutable_table_view{views};
}

// Release ownership of columns
std::vector<std::unique_ptr<column>> table::release()
{
  _num_rows = 0;
  return std::move(_columns);
}

}  // namespace cudf
