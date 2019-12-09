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
#include <cudf/detail/utilities/release_assert.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>

#include <thrust/equal.h>
#include <thrust/swap.h>
#include <thrust/transform_reduce.h>

namespace cudf {
namespace experimental {

/**---------------------------------------------------------------------------*
 * @brief Result type of the `element_relational_comparator` function object.
 *
 * Indicates how two elements `a` and `b` compare with one and another.
 *
 * Equivalence is defined as `not (a<b) and not (b<a)`. Elements that are
 * EQUIVALENT may not necessarily be *equal*.
 *
 *---------------------------------------------------------------------------**/
enum class weak_ordering {
  LESS,        ///< Indicates `a` is less than (ordered before) `b`
  EQUIVALENT,  ///< Indicates `a` is ordered neither before nor after `b`
  GREATER      ///< Indicates `a` is greater than (ordered after) `b`
};

namespace detail {
/**---------------------------------------------------------------------------*
* @brief Compare the elements ordering with respect to `lhs`.
*
* @param[in] lhs first element
* @param[in] rhs second element
* @return weak_ordering Indicates the relationship between the elements in
* the `lhs` and `rhs` columns.
*---------------------------------------------------------------------------**/
template <typename Element>
__device__ weak_ordering compare_elements(Element lhs, Element rhs)
{
    if(lhs < rhs) {
        return weak_ordering::LESS;
    } else if(rhs < lhs) {
        return weak_ordering::GREATER;
    }
    return weak_ordering::EQUIVALENT;
}
}
/**---------------------------------------------------------------------------*
* @brief A specialization for floating-point `Element` type rerlational comparison
* to derive the order of the elements with respect to `lhs`. Specialization is to
* handle `nan` in the order shown below.
* `[-Inf, -ve, 0, -0, +ve, +Inf, NaN, NaN, null] (for null_order::AFTER)`
* `[null, -Inf, -ve, 0, -0, +ve, +Inf, NaN, NaN] (for null_order::BEFORE)`
*
* @param[in] lhs first element
* @param[in] rhs second element
* @return weak_ordering Indicates the relationship between the elements in
* the `lhs` and `rhs` columns.
*---------------------------------------------------------------------------**/
template <typename Element,
            std::enable_if_t<std::is_floating_point<Element>::value>* = nullptr>
__device__ weak_ordering relational_compare(Element lhs, Element rhs) {

    if(isnan(lhs) and isnan(rhs)) {
        return weak_ordering::EQUIVALENT;
    } else if(isnan(rhs)) {
        return weak_ordering::LESS;
    } else if(isnan(lhs)) {
        return weak_ordering::GREATER;
    }

    return detail::compare_elements(lhs, rhs);
}

/**---------------------------------------------------------------------------*
* @brief A specialization for non-floating-point `Element` type relational
* comparison to derive the order of the elements with respect to `lhs`.
*
* @param[in] lhs first element
* @param[in] rhs second element
* @return weak_ordering Indicates the relationship between the elements in
* the `lhs` and `rhs` columns.
*---------------------------------------------------------------------------**/
template <typename Element,
            std::enable_if_t<not std::is_floating_point<Element>::value>* = nullptr>
__device__ weak_ordering relational_compare(Element lhs, Element rhs) {
    return detail::compare_elements(lhs, rhs);
}

/**---------------------------------------------------------------------------*
* @brief A specialization for floating-point `Element` type to check if
* `lhs` is equivalent to `rhs`. `nan == nan`.
*
* @param[in] lhs first element
* @param[in] rhs second element
* @return bool `true` if `lhs` == `rhs` else `false`.
*---------------------------------------------------------------------------**/
template <typename Element,
            std::enable_if_t<std::is_floating_point<Element>::value>* = nullptr>
__device__ bool equality_compare(Element lhs, Element rhs) {
    if (isnan(lhs) and isnan(rhs)) {
        return true;
    }
    return lhs == rhs;
}

/**---------------------------------------------------------------------------*
* @brief A specialization for non-floating-point `Element` type to check if
* `lhs` is equivalent to `rhs`.
*
* @param[in] lhs first element
* @param[in] rhs second element
* @return bool `true` if `lhs` == `rhs` else `false`.
*---------------------------------------------------------------------------**/
template <typename Element,
            std::enable_if_t<not std::is_floating_point<Element>::value>* = nullptr>
__device__ bool equality_compare(Element const lhs, Element const rhs) {
    return lhs == rhs;
}

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

    return equality_compare(lhs.element<Element>(lhs_element_index),
                            rhs.element<Element>(rhs_element_index));
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

  __device__ bool operator()(size_type lhs_row_index, size_type rhs_row_index) const noexcept{
    auto equal_elements = [=](column_device_view l, column_device_view r) {
      return cudf::experimental::type_dispatcher(
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

    return relational_compare(lhs.element<Element>(lhs_element_index),
                              rhs.element<Element>(rhs_element_index));
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
   * @param column_order Optional, device array the same length as a row that
   * indicates the desired ascending/descending order of each column in a row.
   * If `nullptr`, it is assumed all columns are sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row
   * and indicates how null values compare to all other for every column. If
   * it is nullptr, then null precedence would be `null_order::BEFORE` for all
   * columns.
   *---------------------------------------------------------------------------**/
  row_lexicographic_comparator(table_device_view lhs, table_device_view rhs,
                               order const* column_order = nullptr,
                               null_order const* null_precedence = nullptr)
      : _lhs{lhs},
        _rhs{rhs},
        _column_order{column_order},
        _null_precedence{null_precedence} {
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
      null_order null_precedence = _null_precedence == nullptr ?
                                     null_order::BEFORE: _null_precedence[i];

      auto comparator = element_relational_comparator<has_nulls>{
          _lhs.column(i), _rhs.column(i), null_precedence};

      state = cudf::experimental::type_dispatcher(_lhs.column(i).type(), comparator,
                                         lhs_index, rhs_index);

      if (state == weak_ordering::EQUIVALENT) {
        continue;
      }

      return state == (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
    }
    return false;
  }

 private:
  table_device_view _lhs;
  table_device_view _rhs;
  null_order const*  _null_precedence{};
  order const* _column_order{};
};  // class row_lexicographic_comparator

/**---------------------------------------------------------------------------*
 * @brief Computes the hash value of an element in the given column.
 *
 * @tparam hash_function Hash functor to use for hashing elements.
 * @tparam has_nulls Indicates the potential for null values in the column.
 *---------------------------------------------------------------------------**/
template <template <typename> class hash_function, bool has_nulls = true>
class element_hasher {
 public:
  template <typename T>
  __device__ inline hash_value_type operator()(
      column_device_view col, size_type row_index) {
    if (has_nulls && col.is_null(row_index)) {
      return std::numeric_limits<hash_value_type>::max();
    }

    return hash_function<T>{}(col.element<T>(row_index));
  }
};

/**---------------------------------------------------------------------------*
 * @brief Computes the hash value of a row in the given table.
 *
 * @tparam hash_function Hash functor to use for hashing elements.
 * @tparam has_nulls Indicates the potential for null values in the table.
 *---------------------------------------------------------------------------**/
template <template <typename> class hash_function, bool has_nulls = true>
class row_hasher {
 public:
  row_hasher() = delete;
  row_hasher(table_device_view t) : _table{t} {}

  __device__ auto operator()(size_type row_index) const {
    auto hash_combiner = [](hash_value_type lhs, hash_value_type rhs) {
      return hash_function<hash_value_type>{}.hash_combine(lhs, rhs);
    };

    // Hashes an element in a column
    auto hasher = [=](size_type column_index) {
      return cudf::experimental::type_dispatcher(
          _table.column(column_index).type(),
          element_hasher<hash_function, has_nulls>{},
          _table.column(column_index), row_index);
    };

    // Hash each element and combine all the hash values together
    return thrust::transform_reduce(
        thrust::seq, thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(_table.num_columns()), hasher,
        hash_value_type{0}, hash_combiner);
  }

 private:
  table_device_view _table;
};

/**---------------------------------------------------------------------------*
 * @brief Computes the hash value of a row in the given table, combined with an
 * initial hash value for each column.
 *
 * @tparam hash_function Hash functor to use for hashing elements.
 * @tparam has_nulls Indicates the potential for null values in the table.
 *---------------------------------------------------------------------------**/
template <template <typename> class hash_function, bool has_nulls = true>
class row_hasher_initial_values {
 public:
  row_hasher_initial_values() = delete;
  row_hasher_initial_values(table_device_view t, hash_value_type *initial_hash)
      : _table{t}, _initial_hash(initial_hash) {}

  __device__ auto operator()(size_type row_index) const {
    auto hash_combiner = [](hash_value_type lhs, hash_value_type rhs) {
      return hash_function<hash_value_type>{}.hash_combine(lhs, rhs);
    };

    // Hashes an element in a column and combines with an initial value
    auto hasher = [=](size_type column_index) {
      auto hash_value = cudf::experimental::type_dispatcher(
          _table.column(column_index).type(),
          element_hasher<hash_function, has_nulls>{},
          _table.column(column_index), row_index);

      return hash_combiner(_initial_hash[column_index], hash_value);
    };

    // Hash each element and combine all the hash values together
    return thrust::transform_reduce(
        thrust::seq, thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(_table.num_columns()), hasher,
        hash_value_type{0}, hash_combiner);
  }

 private:
  table_device_view _table;
  hash_value_type *_initial_hash;
};

}  // namespace experimental
}  // namespace cudf
