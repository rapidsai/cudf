/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cassert>
#include <algorithm>
#include <vector>

namespace cudf {

namespace detail {

template <typename ColumnView>
table_view_base<ColumnView>::table_view_base(
    std::vector<ColumnView> const& cols)
    : _columns{cols} {
  if(num_columns() > 0)
  {
    std::for_each(_columns.begin(), _columns.end(), [this](ColumnView col) {
      CUDF_EXPECTS(col.size() == _columns.front().size(), "Column size mismatch.");
    });
    _num_rows = _columns.front().size();
  } else {
    _num_rows = 0;
  }
}

template <typename ColumnView>
ColumnView const& table_view_base<ColumnView>::column(
    size_type column_index) const noexcept {
  assert(column_index >= 0);
  assert(column_index < _columns.size());
  return _columns[column_index];
}

// Explicit instantiation for a table of `column_view`s
template class table_view_base<column_view>;

// Explicit instantiation for a table of `mutable_column_view`s
template class table_view_base<mutable_column_view>;
}  // namespace detail

// Convert mutable view to immutable view
mutable_table_view::operator table_view() {
  return table_view{{begin(), end()}};
}

}  // namespace cudf
