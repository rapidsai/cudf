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

#include <cudf/column/column_device_view.cuh>
#include <cudf/sorting.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <utilities/release_assert.cuh>

#include <thrust/equal.h>
#include <thrust/swap.h>

namespace cudf {
namespace exp {

/**---------------------------------------------------------------------------*
 * @brief Performs an equality comparison between two elements in two columns.
 *
 * @tparam has_nulls Indicates the potential for null values in either column.
 *---------------------------------------------------------------------------**/
template <bool has_nulls = true>
class element_equality_comparator {
 public:
  /**---------------------------------------------------------------------------*
   * @brief Construct type-dispatched function object for comparing equality
   * between two elements.
   *
   * @note `lhs` and `rhs` may be the same.
   *
   * @param lhs The column containing the first element
   * @param rhs The column containg the second element (may be the same as lhs)
   * @param nulls_are_equal Indicates if two null elements are treated as
   *equivalent
   *---------------------------------------------------------------------------**/
  __host__ __device__ element_equality_comparator(column_device_view lhs,
                                                  column_device_view rhs,
                                                  bool nulls_are_equal = true)
      : lhs{lhs}, rhs{rhs}, nulls_are_equal{nulls_are_equal} {}

  /**---------------------------------------------------------------------------*
   * @brief Compares the specified elements for equality.
   *
   * @param lhs_element_index The index of the first element
   * @param rhs_element_index The index of the second element
   *---------------------------------------------------------------------------**/
  template <typename Element>
  __device__ bool operator()(size_type lhs_element_index,
                             size_type rhs_element_index) const noexcept {
    if (has_nulls) {
      bool const lhs_is_null{lhs.nullable() and lhs.is_null(lhs_element_index)};
      bool const rhs_is_null{rhs.nullable() and rhs.is_null(rhs_element_index)};
      if (lhs_is_null and rhs_is_null) {
        return nulls_are_equal;
      } else if (lhs_is_null != rhs_is_null) {
        return false;
      }
    }
    return lhs.element<Element>(lhs_element_index) ==
           rhs.element<Element>(rhs_element_index);
  }

 private:
  column_device_view lhs;
  column_device_view rhs;
  bool nulls_are_equal;
};

template <bool has_nulls = true>
class row_equality_comparator {
 public:
  row_equality_comparator(table_device_view lhs, table_device_view rhs,
                          bool nulls_are_equal = true)
      : lhs{lhs}, rhs{rhs}, nulls_are_equal{nulls_are_equal} {
    CUDF_EXPECTS(lhs.num_columns() == rhs.num_columns(),
                 "Mismatched number of columns.");
  }

  __device__ bool operator()(size_type lhs_row_index, size_type rhs_row_index) {
    auto equal_elements = [=](column_device_view l, column_device_view r) {
      return cudf::exp::type_dispatcher(
          l.type(),
          element_equality_comparator<has_nulls>{l, r, nulls_are_equal},
          lhs_row_index, rhs_row_index);
    };

    return thrust::equal(thrust::seq, lhs.begin(), lhs.end(), rhs.begin(),
                         equal_elements);
  }

 private:
  table_device_view lhs;
  table_device_view rhs;
  bool nulls_are_equal;
};

/**---------------------------------------------------------------------------*
 * @brief Result type of the `element_relational_comparator` function object.
 *
 * Indicates how two elements `a` and `b` compare with one and another.
 *
 * Equivalence is defined as `not (a<b) and not (b<a)`. Elements that are are
 * EQUIVALENT may not necessarily be *equal*.
 *
 *---------------------------------------------------------------------------**/
enum class weak_ordering {
  LESS,        ///< Indicates `a` is less than (ordered before) `b`
  EQUIVALENT,  ///< Indicates `a` is ordered neither before nor after `b`
  GREATER      ///< Indicates `a` is greater than (ordered after) `b`
};

/**---------------------------------------------------------------------------*
 * @brief Performs a relational comparison between two elements in two columns.
 *
 * @tparam has_nulls Indicates the potential for null values in either column.
 *---------------------------------------------------------------------------**/
template <bool has_nulls = true>
class element_relational_comparator {
 public:
  /**---------------------------------------------------------------------------*
   * @brief Construct type-dispatched function object for performing a
   * relational comparison between two elements.
   *
   * @note `lhs` and `rhs` may be the same.
   *
   * @param lhs The column containing the first element
   * @param rhs The column containg the second element (may be the same as lhs)
   * @param null_precedence Indicates how null values are ordered with other
   * values
   *---------------------------------------------------------------------------**/
  __host__ __device__ element_relational_comparator(column_device_view lhs,
                                                    column_device_view rhs,
                                                    null_order null_precedence)
      : lhs{lhs}, rhs{rhs}, null_precedence{null_precedence} {}

  /**---------------------------------------------------------------------------*
   * @brief Performs a relational comparison between the specified elements
   *
   * @param lhs_element_index The index of the first element
   * @param rhs_element_index The index of the second element
   * @param null_precedence Indicates how null values are ordered with other
   * values
   * @return weak_ordering Indicates the relationship between the elements in
   * the `lhs` and `rhs` columns.
   *---------------------------------------------------------------------------**/
  template <typename Element, std::enable_if_t<cudf::is_relationally_comparable<
                                  Element, Element>()>* = nullptr>
  __device__ weak_ordering operator()(size_type lhs_element_index,
                                      size_type rhs_element_index) const
      noexcept {
    if (has_nulls) {
      bool const lhs_is_null{lhs.nullable() and lhs.is_null(lhs_element_index)};
      bool const rhs_is_null{rhs.nullable() and rhs.is_null(rhs_element_index)};

      if (lhs_is_null and rhs_is_null) {  // null <? null
        return weak_ordering::EQUIVALENT;
      } else if (lhs_is_null) {  // null <? x
        return (null_precedence == null_order::BEFORE) ? weak_ordering::LESS
                                                       : weak_ordering::GREATER;
      } else if (rhs_is_null) {  // x <? null
        return (null_precedence == null_order::AFTER) ? weak_ordering::LESS
                                                      : weak_ordering::GREATER;
      }
    }

    Element const lhs_element = lhs.element<Element>(lhs_element_index);
    Element const rhs_element = rhs.element<Element>(rhs_element_index);

    if (lhs_element < rhs_element) {
      return weak_ordering::LESS;
    } else if (rhs_element < lhs_element) {
      return weak_ordering::GREATER;
    }
    return weak_ordering::EQUIVALENT;
  }

  template <typename Element,
            std::enable_if_t<not cudf::is_relationally_comparable<
                Element, Element>()>* = nullptr>
  __device__ weak_ordering operator()(size_type lhs_element_index,
                                      size_type rhs_element_index) {
    release_assert(false &&
                   "Attempted to compare elements of uncomparable types.");
  }

 private:
  column_device_view lhs;
  column_device_view rhs;
  null_order null_precedence;
};

/**---------------------------------------------------------------------------*
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
 * @tparam has_nulls Indicates the potential for null values in either row.
 *---------------------------------------------------------------------------**/
template <bool has_nulls = true>
class row_lexicographic_comparator {
 public:
  /**---------------------------------------------------------------------------*
   * @brief Construct a function object for performing a lexicographic
   * comparison between the rows of two tables.
   *
   * @throws cudf::logic_error if `lhs.num_columns() != rhs.num_columns()`
   *
   * @param lhs The first table
   * @param rhs The second table (may be the same table as `lhs`)
   * @param null_precedence Indicates how null values compare to all other
   *values.
   * @param column_order Optional, device array the same length as a row that
   * indicates the desired ascending/descending order of each column in a row.
   * If `nullptr`, it is assumed all columns are sorted in ascending order.
   *---------------------------------------------------------------------------**/
  row_lexicographic_comparator(table_device_view lhs, table_device_view rhs,
                               null_order null_precedence = null_order::BEFORE,
                               order* column_order = nullptr)
      : _lhs{lhs},
        _rhs{rhs},
        _null_precedence{null_precedence},
        _column_order{column_order} {
    // Add check for types to be the same.
    CUDF_EXPECTS(_lhs.num_columns() == _rhs.num_columns(),
                 "Mismatched number of columns.");
  }

  /**---------------------------------------------------------------------------*
   * @brief Checks whether the row at `lhs_index` in the `lhs` table compares
   * lexicographically less than the row at `rhs_index` in the `rhs` table.
   *
   * @param lhs_index The index of row in the `lhs` table to examine
   * @param rhs_index The index of the row in the `rhs` table to examine
   * @return `true` if row from the `lhs` table compares less than row in the
   * `rhs` table
   *---------------------------------------------------------------------------**/
  __device__ bool operator()(size_type lhs_index, size_type rhs_index) const
      noexcept {
    for (size_type i = 0; i < _lhs.num_columns(); ++i) {
      bool ascending =
          (_column_order == nullptr) or (_column_order[i] == order::ASCENDING);

      weak_ordering state{weak_ordering::EQUIVALENT};

      if (not ascending) {
        thrust::swap(lhs_index, rhs_index);
      }

      auto comparator = element_relational_comparator<has_nulls>{
          _lhs.column(i), _rhs.column(i), _null_precedence};

      state = cudf::exp::type_dispatcher(_lhs.column(i).type(), comparator,
                                         lhs_index, rhs_index);

      if (state == weak_ordering::EQUIVALENT) {
        continue;
      }

      return (state == weak_ordering::LESS) ? true : false;
    }
    return false;
  }

 private:
  table_device_view _lhs;
  table_device_view _rhs;
  null_order _null_precedence{null_order::BEFORE};
  order const* _column_order{};
};  // class row_lexicographic_comparator

}  // namespace exp
}  // namespace cudf
