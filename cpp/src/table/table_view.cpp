/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cassert>
#include <vector>

namespace cudf {
namespace detail {
template <typename ColumnView>
table_view_base<ColumnView>::table_view_base(std::vector<ColumnView> const& cols) : _columns{cols}
{
  if (num_columns() > 0) {
    std::for_each(_columns.begin(), _columns.end(), [this](ColumnView col) {
      CUDF_EXPECTS(col.size() == _columns.front().size(), "Column size mismatch.");
    });
    _num_rows = _columns.front().size();
  } else {
    _num_rows = 0;
  }
}

template <typename ViewType>
auto concatenate_column_views(std::vector<ViewType> const& views)
{
  using ColumnView = typename ViewType::ColumnView;
  std::vector<ColumnView> concat_cols;
  for (auto& view : views) {
    concat_cols.insert(concat_cols.end(), view.begin(), view.end());
  }
  return concat_cols;
}

template <typename ColumnView>
ColumnView const& table_view_base<ColumnView>::column(size_type column_index) const
{
  return _columns.at(column_index);
}

// Explicit instantiation for a table of `column_view`s
template class table_view_base<column_view>;

// Explicit instantiation for a table of `mutable_column_view`s
template class table_view_base<mutable_column_view>;
}  // namespace detail

// Returns a table_view with set of specified columns
table_view table_view::select(std::vector<size_type> const& column_indices) const
{
  return select(column_indices.begin(), column_indices.end());
}

// Convert mutable view to immutable view
mutable_table_view::operator table_view()
{
  std::vector<column_view> cols{begin(), end()};
  return table_view{cols};
}

table_view::table_view(std::vector<table_view> const& views)
  : table_view{concatenate_column_views(views)}
{
}

mutable_table_view::mutable_table_view(std::vector<mutable_table_view> const& views)
  : mutable_table_view{concatenate_column_views(views)}
{
}

table_view scatter_columns(table_view const& source,
                           std::vector<size_type> const& map,
                           table_view const& target)
{
  std::vector<cudf::column_view> updated_columns(target.begin(), target.end());
  // scatter(updated_table.begin(),updated_table.end(),indices.begin(),updated_columns.begin());
  for (size_type idx = 0; idx < source.num_columns(); ++idx)
    updated_columns[map[idx]] = source.column(idx);
  return table_view{updated_columns};
}

std::vector<column_view> get_nullable_columns(table_view const& table)
{
  std::vector<column_view> result;
  for (auto const& col : table) {
    if (col.nullable()) { result.push_back(col); }
    for (auto it = col.child_begin(); it != col.child_end(); ++it) {
      auto const& child = *it;
      if (child.size() == col.size()) {
        auto const child_result = get_nullable_columns(table_view{{child}});
        result.insert(result.end(), child_result.begin(), child_result.end());
      }
    }
  }
  return result;
}

namespace detail {

template <typename TableView>
bool is_relationally_comparable(TableView const& lhs, TableView const& rhs)
{
  return std::all_of(thrust::counting_iterator<size_type>(0),
                     thrust::counting_iterator<size_type>(lhs.num_columns()),
                     [lhs, rhs](auto const i) {
                       return lhs.column(i).type() == rhs.column(i).type() and
                              cudf::is_relationally_comparable(lhs.column(i).type());
                     });
}

// Explicit template instantiation for a table of immutable views
template bool is_relationally_comparable<table_view>(table_view const& lhs, table_view const& rhs);

// Explicit template instantiation for a table of mutable views
template bool is_relationally_comparable<mutable_table_view>(mutable_table_view const& lhs,
                                                             mutable_table_view const& rhs);

}  // namespace detail
}  // namespace cudf
