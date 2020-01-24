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
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <algorithm>
#include <vector>

/**---------------------------------------------------------------------------*
 * @file table_view.hpp
 * @brief A `(mutable_)table_view` is a set of `(mutable_)column_view`s of equal
 * size.
 *
 * A `(mutable_)table_view` is non-owning and trivially copyable and should be
 * passed by value.
 *---------------------------------------------------------------------------**/

namespace cudf {
namespace detail {
/**---------------------------------------------------------------------------*
 * @brief Base class for a table of `ColumnView`s
 *
 * This class should not be used directly. Instead:
 * -`table_view` should be used for a table of `column_view`s
 * -`mutable_table_view` should be used for a table of `mutable_column_view`s
 *
 * All public constructors and member functions of `table_view_base` are
 * available in both `table_view` and `mutable_table_view`.
 *
 * @tparam ColumnView The type of column view the table contains
 *---------------------------------------------------------------------------**/
template <typename ColumnView>
class table_view_base {
  static_assert(std::is_same<ColumnView, column_view>::value or
                    std::is_same<ColumnView, mutable_column_view>::value,
                "table_view_base can only be instantiated with column_view or "
                "column_view_base.");

 private:
  std::vector<ColumnView> _columns{};  ///< ColumnViews to columns of equal size
  size_type _num_rows{};  ///< The number of elements in every column

 public:
  using iterator = decltype(std::begin(_columns));
  using const_iterator = decltype(std::cbegin(_columns));

  /**---------------------------------------------------------------------------*
   * @brief Construct a table from a vector of column views
   *
   * @note Because a `std::vector` is constructible from a
   * `std::initializer_list`, this constructor also supports the following
   * usage:
   * ```
   * column_view c0, c1, c2;
   * ...
   * table_view t{{c0,c1,c2}}; // Creates a `table` from c0, c1, c2
   * ```
   *
   * @throws cudf::logic_error If all views do not have the same size
   *
   * @param cols The vector of columns to construct the table from
   *---------------------------------------------------------------------------**/
  explicit table_view_base(std::vector<ColumnView> const& cols);

  /**---------------------------------------------------------------------------*
   * @brief Returns an iterator to the first view in the `table`.
   *---------------------------------------------------------------------------**/
  iterator begin() noexcept { return std::begin(_columns); }

  /**---------------------------------------------------------------------------*
   * @brief Returns an iterator to the first view in the `table`.
   *---------------------------------------------------------------------------**/
  const_iterator begin() const noexcept { return std::begin(_columns); }

  /**---------------------------------------------------------------------------*
   * @brief Returns an iterator one past the last column view in the `table`.
   *
   * `end()` acts as a place holder. Attempting to dereference it results in
   * undefined behavior.
   *---------------------------------------------------------------------------**/
  iterator end() noexcept { return std::end(_columns); }

  /**---------------------------------------------------------------------------*
   * @brief Returns an iterator one past the last column view in the `table`.
   *
   * `end()` acts as a place holder. Attempting to dereference it results in
   * undefined behavior.
   *---------------------------------------------------------------------------**/
  const_iterator end() const noexcept { return std::end(_columns); }

  /**---------------------------------------------------------------------------*
   * @brief Returns a reference to the view of the specified column
   *
   * @throws std::out_of_range
   * If `column_index` is out of the range [0, num_columns)
   *
   * @param column_index The index of the desired column
   * @return A reference to the desired column
   *---------------------------------------------------------------------------**/
  ColumnView const& column(size_type column_index) const;

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of columns
   *---------------------------------------------------------------------------**/
  size_type num_columns() const noexcept { return _columns.size(); }

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of rows
   *---------------------------------------------------------------------------**/
  size_type num_rows() const noexcept { return _num_rows; }

  table_view_base() = default;

  ~table_view_base() = default;

  table_view_base(table_view_base const&) = default;

  table_view_base(table_view_base&&) = default;
  table_view_base& operator=(table_view_base const&) = default;
  table_view_base& operator=(table_view_base&&) = default;
};
}  // namespace detail

/**---------------------------------------------------------------------------*
 * @brief A set of `column_view`s of the same size.
 *
 * All public member functions and constructors are inherited from
 *`table_view_base<column_view>`.
 *---------------------------------------------------------------------------**/
class table_view : public detail::table_view_base<column_view> {
  using detail::table_view_base<column_view>::table_view_base;

public:
  using ColumnView = column_view;

  table_view() = default;

  /**---------------------------------------------------------------------------*
   * @brief Construct a table from a vector of table views
   *
   * @note Because a `std::vector` is constructible from a
   * `std::initializer_list`, this constructor also supports the following
   * usage:
   * ```
   * table_view t0, t1, t2;
   * ...
   * table_view t{{t0,t1,t2}}; // Creates a `table` from the columns of
   * t0, t1, t2
   * ```
   *
   * @throws cudf::logic_error
   * If number of rows mismatch
   *
   * @param views The vector of table views to construct the table from
   *---------------------------------------------------------------------------**/
  table_view(std::vector<table_view> const &views);

  /**---------------------------------------------------------------------------*
   * @brief Returns a table_view with set of specified columns.
   *
   * @throws std::out_of_range
   * If any element in `column_indices` is outside [0, num_columns())
   *
   * @param column_indices Indices of columns in the table
   * @return A table_view consisting of columns from the original table
   * specified by the elements of `column_indices`
   *---------------------------------------------------------------------------**/
  table_view select(std::vector<size_type> const& column_indices) const;
};

/**---------------------------------------------------------------------------*
 * @brief A set of `mutable_column_view`s of the same size.
 *
 * All public member functions and constructors are inherited from
 *`table_view_base<mutable_column_view>`.
 *---------------------------------------------------------------------------**/
class mutable_table_view : public detail::table_view_base<mutable_column_view> {
  using detail::table_view_base<mutable_column_view>::table_view_base;

public:
  using ColumnView = mutable_column_view;

  mutable_table_view() = default;

  mutable_column_view& column(size_type column_index) const {
    return const_cast<mutable_column_view&>(table_view_base::column(column_index));
  }
  /**---------------------------------------------------------------------------*
   * @brief Creates an immutable `table_view` of the columns
   *---------------------------------------------------------------------------**/
  operator table_view();

  /**---------------------------------------------------------------------------*
   * @brief Construct a table from a vector of table views
   *
   * @note Because a `std::vector` is constructible from a
   * `std::initializer_list`, this constructor also supports the following
   * usage:
   * ```
   * table_view t0, t1, t2;
   * ...
   * table_view t{{t0,t1,t2}}; // Creates a `table` from the columns of
   * t0, t1, t2
   * ```
   *
   * @throws cudf::logic_error
   * If number of rows mismatch
   *
   * @param views The vector of table views to construct the table from
   *---------------------------------------------------------------------------**/
  mutable_table_view(std::vector<mutable_table_view> const &views);
};

inline bool has_nulls(table_view view) {
  return std::any_of(view.begin(), view.end(),
                     [](column_view col) { return col.has_nulls(); });
}

/**---------------------------------------------------------------------------*
   * @brief Checks if two `table_view`s have columns of same types
   *
   * @param lhs left-side table_view operand
   * @param rhs right-side table_view operand
   * @return boolean comparison result
   *---------------------------------------------------------------------------**/
inline bool have_same_types(table_view const& lhs, table_view const& rhs) {
  return std::equal(lhs.begin(), lhs.end(),
                    rhs.begin(), rhs.end(),
                    [](column_view const& lcol, column_view const& rcol){
                      return (lcol.type() == rcol.type());
                    });
}
}  // namespace cudf
