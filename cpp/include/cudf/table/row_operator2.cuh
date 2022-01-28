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
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/equal.h>
#include <thrust/swap.h>
#include <thrust/transform_reduce.h>

#include <limits>

namespace cudf {

/**
 * @brief Result type of the `element_relational_comparator2` function object.
 *
 * Indicates how two elements `a` and `b` compare with one and another.
 *
 * Equivalence is defined as `not (a<b) and not (b<a)`. Elements that are
 * EQUIVALENT may not necessarily be *equal*.
 */
enum class weak_ordering2 {
  LESS,        ///< Indicates `a` is less than (ordered before) `b`
  EQUIVALENT,  ///< Indicates `a` is ordered neither before nor after `b`
  GREATER      ///< Indicates `a` is greater than (ordered after) `b`
};

namespace detail {
/**
 * @brief Compare the elements ordering with respect to `lhs`.
 *
 * @param[in] lhs first element
 * @param[in] rhs second element
 * @return weak_ordering2 Indicates the relationship between the elements in
 * the `lhs` and `rhs` columns.
 */
template <typename Element>
__device__ weak_ordering2 compare_elements2(Element lhs, Element rhs)
{
  if (lhs < rhs) {
    return weak_ordering2::LESS;
  } else if (rhs < lhs) {
    return weak_ordering2::GREATER;
  }
  return weak_ordering2::EQUIVALENT;
}
}  // namespace detail

/*
 * @brief A specialization for floating-point `Element` type relational comparison
 * to derive the order of the elements with respect to `lhs`. Specialization is to
 * handle `nan` in the order shown below.
 * `[-Inf, -ve, 0, -0, +ve, +Inf, NaN, NaN, null] (for null_order::AFTER)`
 * `[null, -Inf, -ve, 0, -0, +ve, +Inf, NaN, NaN] (for null_order::BEFORE)`
 *
 * @param[in] lhs first element
 * @param[in] rhs second element
 * @return weak_ordering2 Indicates the relationship between the elements in
 * the `lhs` and `rhs` columns.
 */
template <typename Element, std::enable_if_t<std::is_floating_point<Element>::value>* = nullptr>
__device__ weak_ordering2 relational_compare2(Element lhs, Element rhs)
{
  if (isnan(lhs) and isnan(rhs)) {
    return weak_ordering2::EQUIVALENT;
  } else if (isnan(rhs)) {
    return weak_ordering2::LESS;
  } else if (isnan(lhs)) {
    return weak_ordering2::GREATER;
  }

  return detail::compare_elements2(lhs, rhs);
}

/**
 * @brief Compare the nulls according to null order.
 *
 * @param lhs_is_null boolean representing if lhs is null
 * @param rhs_is_null boolean representing if lhs is null
 * @param null_precedence null order
 * @return Indicates the relationship between null in lhs and rhs columns.
 */
inline __device__ auto null_compare2(bool lhs_is_null, bool rhs_is_null, null_order null_precedence)
{
  if (lhs_is_null and rhs_is_null) {  // null <? null
    return weak_ordering2::EQUIVALENT;
  } else if (lhs_is_null) {  // null <? x
    return (null_precedence == null_order::BEFORE) ? weak_ordering2::LESS : weak_ordering2::GREATER;
  } else if (rhs_is_null) {  // x <? null
    return (null_precedence == null_order::AFTER) ? weak_ordering2::LESS : weak_ordering2::GREATER;
  }
  return weak_ordering2::EQUIVALENT;
}

/**
 * @brief A specialization for non-floating-point `Element` type relational
 * comparison to derive the order of the elements with respect to `lhs`.
 *
 * @param[in] lhs first element
 * @param[in] rhs second element
 * @return weak_ordering2 Indicates the relationship between the elements in
 * the `lhs` and `rhs` columns.
 */
template <typename Element, std::enable_if_t<not std::is_floating_point<Element>::value>* = nullptr>
__device__ weak_ordering2 relational_compare2(Element lhs, Element rhs)
{
  return detail::compare_elements2(lhs, rhs);
}

/**
 * @brief A specialization for floating-point `Element` type to check if
 * `lhs` is equivalent to `rhs`. `nan == nan`.
 *
 * @param[in] lhs first element
 * @param[in] rhs second element
 * @return bool `true` if `lhs` == `rhs` else `false`.
 */
template <typename Element, std::enable_if_t<std::is_floating_point<Element>::value>* = nullptr>
__device__ bool equality_compare2(Element lhs, Element rhs)
{
  if (isnan(lhs) and isnan(rhs)) { return true; }
  return lhs == rhs;
}

/**
 * @brief A specialization for non-floating-point `Element` type to check if
 * `lhs` is equivalent to `rhs`.
 *
 * @param[in] lhs first element
 * @param[in] rhs second element
 * @return bool `true` if `lhs` == `rhs` else `false`.
 */
template <typename Element, std::enable_if_t<not std::is_floating_point<Element>::value>* = nullptr>
__device__ bool equality_compare2(Element const lhs, Element const rhs)
{
  return lhs == rhs;
}

/**
 * @brief Performs an equality comparison between two elements in two columns.
 *
 * @tparam has_nulls Indicates the potential for null values in either column.
 */
template <bool has_nulls = true>
class element_equality_comparator2 {
 public:
  /**
   * @brief Construct type-dispatched function object for comparing equality
   * between two elements.
   *
   * @note `lhs` and `rhs` may be the same.
   *
   * @param lhs The column containing the first element
   * @param rhs The column containing the second element (may be the same as lhs)
   * @param nulls_are_equal Indicates if two null elements are treated as equivalent
   */
  __host__ __device__ element_equality_comparator2(column_device_view lhs,
                                                   column_device_view rhs,
                                                   bool nulls_are_equal = true)
    : lhs{lhs}, rhs{rhs}, nulls_are_equal{nulls_are_equal}
  {
  }

  /**
   * @brief Compares the specified elements for equality.
   *
   * @param lhs_element_index The index of the first element
   * @param rhs_element_index The index of the second element
   *
   */
  template <typename Element,
            std::enable_if_t<cudf::is_equality_comparable<Element, Element>()>* = nullptr>
  __device__ bool operator()(size_type lhs_element_index,
                             size_type rhs_element_index) const noexcept
  {
    if (has_nulls) {
      bool const lhs_is_null{lhs.is_null(lhs_element_index)};
      bool const rhs_is_null{rhs.is_null(rhs_element_index)};
      if (lhs_is_null and rhs_is_null) {
        return nulls_are_equal;
      } else if (lhs_is_null != rhs_is_null) {
        return false;
      }
    }

    return equality_compare2(lhs.element<Element>(lhs_element_index),
                             rhs.element<Element>(rhs_element_index));
  }

  template <typename Element,
            std::enable_if_t<not cudf::is_equality_comparable<Element, Element>()>* = nullptr>
  __device__ bool operator()(size_type lhs_element_index, size_type rhs_element_index)
  {
    cudf_assert(false && "Attempted to compare elements of uncomparable types.");
    return false;
  }

 private:
  column_device_view lhs;
  column_device_view rhs;
  bool nulls_are_equal;
};

template <bool has_nulls = true>
class row_equality_comparator2 {
 public:
  row_equality_comparator2(table_device_view lhs,
                           table_device_view rhs,
                           bool nulls_are_equal = true)
    : lhs{lhs}, rhs{rhs}, nulls_are_equal{nulls_are_equal}
  {
    CUDF_EXPECTS(lhs.num_columns() == rhs.num_columns(), "Mismatched number of columns.");
  }

  __device__ bool operator()(size_type lhs_row_index, size_type rhs_row_index) const noexcept
  {
    auto equal_elements = [=](column_device_view l, column_device_view r) {
      return cudf::type_dispatcher(l.type(),
                                   element_equality_comparator2<has_nulls>{l, r, nulls_are_equal},
                                   lhs_row_index,
                                   rhs_row_index);
    };

    return thrust::equal(thrust::seq, lhs.begin(), lhs.end(), rhs.begin(), equal_elements);
  }

 private:
  table_device_view lhs;
  table_device_view rhs;
  bool nulls_are_equal;
};

/**
 * @brief Performs a relational comparison between two elements in two columns.
 *
 * @tparam has_nulls Indicates the potential for null values in either column.
 */
template <bool has_nulls = true>
class element_relational_comparator2 {
 public:
  /**
   * @brief Construct type-dispatched function object for performing a
   * relational comparison between two elements.
   *
   * @note `lhs` and `rhs` may be the same.
   *
   * @param lhs The column containing the first element
   * @param rhs The column containing the second element (may be the same as lhs)
   * @param null_precedence Indicates how null values are ordered with other
   * values
   */
  __host__ __device__ element_relational_comparator2(column_device_view lhs,
                                                     column_device_view rhs,
                                                     null_order null_precedence)
    : lhs{lhs}, rhs{rhs}, null_precedence{null_precedence}
  {
  }

  /**
   * @brief Performs a relational comparison between the specified elements
   *
   * @param lhs_element_index The index of the first element
   * @param rhs_element_index The index of the second element
   * @param null_precedence Indicates how null values are ordered with other
   * values
   * @return weak_ordering2 Indicates the relationship between the elements in
   * the `lhs` and `rhs` columns.
   */
  template <typename Element,
            std::enable_if_t<cudf::is_relationally_comparable<Element, Element>()>* = nullptr>
  __device__ weak_ordering2 operator()(size_type lhs_element_index,
                                       size_type rhs_element_index) const noexcept
  {
    if (has_nulls) {
      bool const lhs_is_null{lhs.is_null(lhs_element_index)};
      bool const rhs_is_null{rhs.is_null(rhs_element_index)};

      if (lhs_is_null or rhs_is_null) {  // atleast one is null
        return null_compare2(lhs_is_null, rhs_is_null, null_precedence);
      }
    }

    return relational_compare2(lhs.element<Element>(lhs_element_index),
                               rhs.element<Element>(rhs_element_index));
  }

  template <typename Element,
            std::enable_if_t<not cudf::is_relationally_comparable<Element, Element>()>* = nullptr>
  __device__ weak_ordering2 operator()(size_type lhs_element_index, size_type rhs_element_index)
  {
    cudf_assert(false && "Attempted to compare elements of uncomparable types.");
    return weak_ordering2::LESS;
  }

 private:
  column_device_view lhs;
  column_device_view rhs;
  null_order null_precedence;
};

template <typename T>
struct device_stack {
  __device__ device_stack(T* stack_storage, int capacity)
    : stack(stack_storage), capacity(capacity), size(0)
  {
  }
  __device__ void push(T const& val)
  {
    cudf_assert(size < capacity and "Stack overflow");
    stack[size++] = val;
  }
  __device__ T pop()
  {
    cudf_assert(size > 0 and "Stack underflow");
    return stack[--size];
  }
  __device__ T top()
  {
    cudf_assert(size > 0 and "Stack underflow");
    return stack[size - 1];
  }
  __device__ bool empty() { return size == 0; }

 private:
  T* stack;
  int capacity;
  int size;
};

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
 * @tparam has_nulls Indicates the potential for null values in either row.
 */
template <bool has_nulls = true>
class row_lexicographic_comparator2 {
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
   * @param column_order Optional, device array the same length as a row that
   * indicates the desired ascending/descending order of each column in a row.
   * If `nullptr`, it is assumed all columns are sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row
   * and indicates how null values compare to all other for every column. If
   * it is nullptr, then null precedence would be `null_order::BEFORE` for all
   * columns.
   */
  row_lexicographic_comparator2(table_device_view lhs,
                                table_device_view rhs,
                                order const* column_order         = nullptr,
                                null_order const* null_precedence = nullptr)
    : _lhs{lhs}, _rhs{rhs}, _column_order{column_order}, _null_precedence{null_precedence}
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
    using stack_value_type =
      thrust::tuple<column_device_view const*, column_device_view const*, size_t>;
    stack_value_type stack_storage[10];

    for (size_type i = 0; i < _lhs.num_columns(); ++i) {
      device_stack<stack_value_type> stack(stack_storage, 9);
      bool ascending = (_column_order == nullptr) or (_column_order[i] == order::ASCENDING);

      weak_ordering2 state{weak_ordering2::EQUIVALENT};
      null_order null_precedence =
        _null_precedence == nullptr ? null_order::BEFORE : _null_precedence[i];

      column_device_view const* lcol = _lhs.begin() + i;
      column_device_view const* rcol = _rhs.begin() + i;
      size_t curr_child              = 0;

      while (true) {
        bool const lhs_is_null{lcol->is_null(lhs_index)};
        bool const rhs_is_null{rcol->is_null(rhs_index)};

        if (lhs_is_null or rhs_is_null) {  // atleast one is null
          state = null_compare2(lhs_is_null, rhs_is_null, null_precedence);
          if (state != weak_ordering2::EQUIVALENT) break;
        } else if (lcol->type().id() != type_id::STRUCT) {
          auto comparator =
            element_relational_comparator2<has_nulls>{*lcol, *rcol, null_precedence};
          state = cudf::type_dispatcher(lcol->type(), comparator, lhs_index, rhs_index);
          if (state != weak_ordering2::EQUIVALENT) break;
        }

        // Reaching here means the nullability was same and we need to continue comparing
        if (lcol->type().id() == type_id::STRUCT) {
          stack.push({lcol, rcol, 0});
        } else {
          // unwind stack until we reach a struct level with children still left to compare
          bool completed_comparison = false;
          do {
            if (stack.empty()) {
              completed_comparison = true;
              break;
            }
            thrust::tie(lcol, rcol, curr_child) = stack.pop();
          } while (lcol->num_child_columns() <= curr_child + 1);
          if (completed_comparison) { break; }
          stack.push({lcol, rcol, curr_child + 1});
          // break;
        }

        // The top of the stack now is where we have to continue comparing from
        thrust::tie(lcol, rcol, curr_child) = stack.top();

        lcol = &lcol->children()[curr_child];
        rcol = &rcol->children()[curr_child];
      }

      if (state == weak_ordering2::EQUIVALENT) { continue; }

      return state == (ascending ? weak_ordering2::LESS : weak_ordering2::GREATER);
    }
    return false;
  }

 private:
  table_device_view _lhs;
  table_device_view _rhs;
  null_order const* _null_precedence{};
  order const* _column_order{};
};  // class row_lexicographic_comparator2

}  // namespace cudf
