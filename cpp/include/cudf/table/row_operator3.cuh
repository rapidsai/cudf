/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/sorting.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/equal.h>
#include <thrust/swap.h>
#include <thrust/transform_reduce.h>

#include <limits>

namespace cudf {
namespace experimental {

/**
 * @brief Computes whether one row is lexicographically *less* than another row.
 *
 * Lexicographic ordering is determined by:
 * - Two rows are compared element by element.
 * - The first mismatching element defines which row is lexicographically less
 * or greater than the other.
 *
 * Lexicographic ordering is exactly equivalent to doing an alphabetical sort of
 * two words, for example, `aac` would be *less* than (or precede) `abb`. The
 * second letter in both words is the first non-equal letter, and `a < b`, thus
 * `aac < abb`.
 *
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 */
template <typename Nullate>
class row_lexicographic_comparator {
 public:
  /**
   * @brief Construct a function object for performing a lexicographic
   * comparison between the rows of two tables.
   *
   * @throws cudf::logic_error if `lhs.num_columns() != rhs.num_columns()`
   * @throws cudf::logic_error if column types of `lhs` and `rhs` are not comparable.
   *
   * @param lhs The first table
   * @param rhs The second table (may be the same table as `lhs`)
   * @param has_nulls Indicates if either input table contains columns with nulls.
   * @param column_order Optional, device array the same length as a row that
   * indicates the desired ascending/descending order of each column in a row.
   * If `nullptr`, it is assumed all columns are sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row
   * and indicates how null values compare to all other for every column. If
   * it is nullptr, then null precedence would be `null_order::BEFORE` for all
   * columns.
   */
  row_lexicographic_comparator(Nullate has_nulls,
                               table_device_view lhs,
                               table_device_view rhs,
                               order const* column_order         = nullptr,
                               null_order const* null_precedence = nullptr)
    : _lhs{lhs},
      _rhs{rhs},
      _nulls{has_nulls},
      _column_order{column_order},
      _null_precedence{null_precedence}
  {
    CUDF_EXPECTS(_lhs.num_columns() == _rhs.num_columns(), "Mismatched number of columns.");
    // CUDF_EXPECTS(detail::is_relationally_comparable(_lhs, _rhs),
    //              "Attempted to compare elements of uncomparable types.");
  }

  /**
   * @brief Checks whether the row at `lhs_index` in the `lhs` table compares
   * lexicographically less than the row at `rhs_index` in the `rhs` table.
   *
   * @param lhs_index The index of row in the `lhs` table to examine
   * @param rhs_index The index of the row in the `rhs` table to examine
   * @return `true` if row from the `lhs` table compares less than row in the
   * `rhs` table
   */
  __device__ bool operator()(size_type lhs_index, size_type rhs_index) const noexcept
  {
    for (size_type i = 0; i < _lhs.num_columns(); ++i) {
      bool ascending = (_column_order == nullptr) or (_column_order[i] == order::ASCENDING);

      weak_ordering state{weak_ordering::EQUIVALENT};
      null_order null_precedence =
        _null_precedence == nullptr ? null_order::BEFORE : _null_precedence[i];

      column_device_view lcol = _lhs.column(i);
      column_device_view rcol = _rhs.column(i);
      while (lcol.type().id() == type_id::STRUCT) {
        bool const lhs_is_null{lcol.is_null(lhs_index)};
        bool const rhs_is_null{rcol.is_null(rhs_index)};

        if (lhs_is_null or rhs_is_null) {  // atleast one is null
          state = null_compare(lhs_is_null, rhs_is_null, null_precedence);
          if (state != weak_ordering::EQUIVALENT) break;
        }

        lcol = lcol.children()[0];
        rcol = rcol.children()[0];
      }

      if (state == weak_ordering::EQUIVALENT) {
        auto comparator = element_relational_comparator{_nulls, lcol, rcol, null_precedence};
        state           = cudf::type_dispatcher(lcol.type(), comparator, lhs_index, rhs_index);
      }

      if (state == weak_ordering::EQUIVALENT) { continue; }

      return state == (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
    }
    return false;
  }

 private:
  table_device_view _lhs;
  table_device_view _rhs;
  Nullate _nulls{};
  null_order const* _null_precedence{};
  order const* _column_order{};
};  // class row_lexicographic_comparator

}  // namespace experimental
}  // namespace cudf
