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

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/limits>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

namespace CUDF_EXPORT cudf {

/**
 * @brief Result type of the `element_relational_comparator` function object.
 *
 * Indicates how two elements `a` and `b` compare with one and another.
 *
 * Equivalence is defined as `not (a<b) and not (b<a)`. Elements that are
 * EQUIVALENT may not necessarily be *equal*.
 */
enum class weak_ordering {
  LESS,        ///< Indicates `a` is less than (ordered before) `b`
  EQUIVALENT,  ///< Indicates `a` is ordered neither before nor after `b`
  GREATER      ///< Indicates `a` is greater than (ordered after) `b`
};

namespace detail {
/**
 * @brief Compare the elements ordering with respect to `lhs`.
 *
 * @param lhs first element
 * @param rhs second element
 * @return Indicates the relationship between the elements in
 * the `lhs` and `rhs` columns.
 */
template <typename Element>
__device__ weak_ordering compare_elements(Element lhs, Element rhs)
{
  if (lhs < rhs) {
    return weak_ordering::LESS;
  } else if (rhs < lhs) {
    return weak_ordering::GREATER;
  }
  return weak_ordering::EQUIVALENT;
}
}  // namespace detail

/**
 * @brief A specialization for floating-point `Element` type relational comparison
 * to derive the order of the elements with respect to `lhs`.
 *
 * This specialization handles `nan` in the following order:
 * `[-Inf, -ve, 0, -0, +ve, +Inf, NaN, NaN, null] (for null_order::AFTER)`
 * `[null, -Inf, -ve, 0, -0, +ve, +Inf, NaN, NaN] (for null_order::BEFORE)`
 *
 */
template <typename Element, std::enable_if_t<std::is_floating_point_v<Element>>* = nullptr>
__device__ weak_ordering relational_compare(Element lhs, Element rhs)
{
  if (isnan(lhs) and isnan(rhs)) {
    return weak_ordering::EQUIVALENT;
  } else if (isnan(rhs)) {
    return weak_ordering::LESS;
  } else if (isnan(lhs)) {
    return weak_ordering::GREATER;
  }

  return detail::compare_elements(lhs, rhs);
}

/**
 * @brief Compare the nulls according to null order.
 *
 * @param lhs_is_null boolean representing if lhs is null
 * @param rhs_is_null boolean representing if lhs is null
 * @param null_precedence null order
 * @return Indicates the relationship between null in lhs and rhs columns.
 */
inline __device__ auto null_compare(bool lhs_is_null, bool rhs_is_null, null_order null_precedence)
{
  if (lhs_is_null and rhs_is_null) {  // null <? null
    return weak_ordering::EQUIVALENT;
  } else if (lhs_is_null) {  // null <? x
    return (null_precedence == null_order::BEFORE) ? weak_ordering::LESS : weak_ordering::GREATER;
  } else if (rhs_is_null) {  // x <? null
    return (null_precedence == null_order::AFTER) ? weak_ordering::LESS : weak_ordering::GREATER;
  }
  return weak_ordering::EQUIVALENT;
}

/**
 * @brief A specialization for non-floating-point `Element` type relational
 * comparison to derive the order of the elements with respect to `lhs`.
 *
 * @param lhs The first element
 * @param rhs The second element
 * @return Indicates the relationship between the elements in the `lhs` and `rhs` columns
 */
template <typename Element, std::enable_if_t<not std::is_floating_point_v<Element>>* = nullptr>
__device__ weak_ordering relational_compare(Element lhs, Element rhs)
{
  return detail::compare_elements(lhs, rhs);
}

/**
 * @brief A specialization for floating-point `Element` type to check if
 * `lhs` is equivalent to `rhs`. `nan == nan`.
 *
 * @param lhs first element
 * @param rhs second element
 * @return `true` if `lhs` == `rhs` else `false`.
 */
template <typename Element, std::enable_if_t<std::is_floating_point_v<Element>>* = nullptr>
__device__ bool equality_compare(Element lhs, Element rhs)
{
  if (isnan(lhs) and isnan(rhs)) { return true; }
  return lhs == rhs;
}

/**
 * @brief A specialization for non-floating-point `Element` type to check if
 * `lhs` is equivalent to `rhs`.
 *
 * @param lhs first element
 * @param rhs second element
 * @return `true` if `lhs` == `rhs` else `false`.
 */
template <typename Element, std::enable_if_t<not std::is_floating_point_v<Element>>* = nullptr>
__device__ bool equality_compare(Element const lhs, Element const rhs)
{
  return lhs == rhs;
}

/**
 * @brief Performs an equality comparison between two elements in two columns.
 *
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 */
template <typename Nullate>
class element_equality_comparator {
 public:
  /**
   * @brief Construct type-dispatched function object for comparing equality
   * between two elements.
   *
   * @note `lhs` and `rhs` may be the same.
   *
   * @param has_nulls Indicates if either input column contains nulls.
   * @param lhs The column containing the first element
   * @param rhs The column containing the second element (may be the same as lhs)
   * @param nulls_are_equal Indicates if two null elements are treated as equivalent
   */
  __host__ __device__
  element_equality_comparator(Nullate has_nulls,
                              column_device_view lhs,
                              column_device_view rhs,
                              null_equality nulls_are_equal = null_equality::EQUAL)
    : lhs{lhs}, rhs{rhs}, nulls{has_nulls}, nulls_are_equal{nulls_are_equal}
  {
  }

  /**
   * @brief Compares the specified elements for equality.
   *
   * @param lhs_element_index The index of the first element
   * @param rhs_element_index The index of the second element
   * @return True if both lhs and rhs element are both nulls and `nulls_are_equal` is true, or equal
   */
  template <typename Element,
            std::enable_if_t<cudf::is_equality_comparable<Element, Element>()>* = nullptr>
  __device__ bool operator()(size_type lhs_element_index,
                             size_type rhs_element_index) const noexcept
  {
    if (nulls) {
      bool const lhs_is_null{lhs.is_null(lhs_element_index)};
      bool const rhs_is_null{rhs.is_null(rhs_element_index)};
      if (lhs_is_null and rhs_is_null) {
        return nulls_are_equal == null_equality::EQUAL;
      } else if (lhs_is_null != rhs_is_null) {
        return false;
      }
    }

    return equality_compare(lhs.element<Element>(lhs_element_index),
                            rhs.element<Element>(rhs_element_index));
  }

  // @cond
  template <typename Element,
            std::enable_if_t<not cudf::is_equality_comparable<Element, Element>()>* = nullptr>
  __device__ bool operator()(size_type lhs_element_index, size_type rhs_element_index)
  {
    CUDF_UNREACHABLE("Attempted to compare elements of uncomparable types.");
  }
  // @endcond

 private:
  column_device_view lhs;
  column_device_view rhs;
  Nullate nulls;
  null_equality nulls_are_equal;
};

/**
 * @brief Performs a relational comparison between two elements in two tables.
 *
 * @tparam Nullate A cudf::nullate type describing how to check for nulls
 */
template <typename Nullate>
class row_equality_comparator {
 public:
  /**
   * @brief Construct a new row equality comparator object
   *
   * @param has_nulls Indicates if either input column contains nulls
   * @param lhs The column containing the first element
   * @param rhs The column containing the second element (may be the same as lhs)
   * @param nulls_are_equal Indicates if two null elements are treated as equivalent
   */
  row_equality_comparator(Nullate has_nulls,
                          table_device_view lhs,
                          table_device_view rhs,
                          null_equality nulls_are_equal = null_equality::EQUAL)
    : lhs{lhs}, rhs{rhs}, nulls{has_nulls}, nulls_are_equal{nulls_are_equal}
  {
    CUDF_EXPECTS(lhs.num_columns() == rhs.num_columns(), "Mismatched number of columns.");
  }

  /**
   * @brief Compares the specified rows for equality.
   *
   * @param lhs_row_index The index of the first row to compare (in the lhs table)
   * @param rhs_row_index The index of the second row to compare (in the rhs table)
   * @return true if both rows are equal, otherwise false
   */
  __device__ bool operator()(size_type lhs_row_index, size_type rhs_row_index) const noexcept
  {
    auto equal_elements = [=](column_device_view l, column_device_view r) {
      return cudf::type_dispatcher(l.type(),
                                   element_equality_comparator{nulls, l, r, nulls_are_equal},
                                   lhs_row_index,
                                   rhs_row_index);
    };

    return thrust::equal(thrust::seq, lhs.begin(), lhs.end(), rhs.begin(), equal_elements);
  }

 private:
  table_device_view lhs;
  table_device_view rhs;
  Nullate nulls;
  null_equality nulls_are_equal;
};

/**
 * @brief Performs a relational comparison between two elements in two columns.
 *
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 */
template <typename Nullate>
class element_relational_comparator {
 public:
  /**
   * @brief Construct type-dispatched function object for performing a
   * relational comparison between two elements.
   *
   * @note `lhs` and `rhs` may be the same.
   *
   * @param lhs The column containing the first element
   * @param rhs The column containing the second element (may be the same as lhs)
   * @param has_nulls Indicates if either input column contains nulls.
   * @param null_precedence Indicates how null values are ordered with other values
   */
  __host__ __device__ element_relational_comparator(Nullate has_nulls,
                                                    column_device_view lhs,
                                                    column_device_view rhs,
                                                    null_order null_precedence)
    : lhs{lhs}, rhs{rhs}, nulls{has_nulls}, null_precedence{null_precedence}
  {
  }

  /**
   * @brief Construct type-dispatched function object for performing a relational comparison between
   * two elements in two columns.
   *
   * @param has_nulls Indicates if either input column contains nulls
   * @param lhs The column containing the first element
   * @param rhs The column containing the second element (may be the same as lhs)
   */
  __host__ __device__ element_relational_comparator(Nullate has_nulls,
                                                    column_device_view lhs,
                                                    column_device_view rhs)
    : lhs{lhs}, rhs{rhs}, nulls{has_nulls}
  {
  }

  /**
   * @brief Performs a relational comparison between the specified elements
   *
   * @param lhs_element_index The index of the first element
   * @param rhs_element_index The index of the second element
   * @return Indicates the relationship between the elements in
   * the `lhs` and `rhs` columns.
   */
  template <typename Element,
            std::enable_if_t<cudf::is_relationally_comparable<Element, Element>()>* = nullptr>
  __device__ weak_ordering operator()(size_type lhs_element_index,
                                      size_type rhs_element_index) const noexcept
  {
    if (nulls) {
      bool const lhs_is_null{lhs.is_null(lhs_element_index)};
      bool const rhs_is_null{rhs.is_null(rhs_element_index)};

      if (lhs_is_null or rhs_is_null) {  // at least one is null
        return null_compare(lhs_is_null, rhs_is_null, null_precedence);
      }
    }

    return relational_compare(lhs.element<Element>(lhs_element_index),
                              rhs.element<Element>(rhs_element_index));
  }

  // @cond
  template <typename Element,
            std::enable_if_t<not cudf::is_relationally_comparable<Element, Element>()>* = nullptr>
  __device__ weak_ordering operator()(size_type lhs_element_index, size_type rhs_element_index)
  {
    CUDF_UNREACHABLE("Attempted to compare elements of uncomparable types.");
  }
  // @endcond

 private:
  column_device_view lhs;
  column_device_view rhs;
  Nullate nulls;
  null_order null_precedence{};
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
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 */
template <typename Nullate>
class row_lexicographic_comparator {
 public:
  /**
   * @brief Construct a function object for performing a lexicographic
   * comparison between the rows of two tables.
   *
   * Behavior is undefined if called with incomparable column types.
   *
   * @throws cudf::logic_error if `lhs.num_columns() != rhs.num_columns()`
   *
   * @param has_nulls Indicates if either input table contains columns with nulls.
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
  }

  /**
   * @brief Checks whether the row at `lhs_index` in the `lhs` table compares
   * lexicographically less than the row at `rhs_index` in the `rhs` table.
   *
   * @param lhs_index The index of the row in the `lhs` table to examine
   * @param rhs_index The index of the row in the `rhs` table to examine
   * @return `true` if row from the `lhs` table compares less than row in the
   * `rhs` table
   */
  __device__ bool operator()(size_type lhs_index, size_type rhs_index) const noexcept
  {
    for (size_type i = 0; i < _lhs.num_columns(); ++i) {
      bool ascending = (_column_order == nullptr) or (_column_order[i] == order::ASCENDING);

      null_order null_precedence =
        _null_precedence == nullptr ? null_order::BEFORE : _null_precedence[i];

      auto comparator =
        element_relational_comparator{_nulls, _lhs.column(i), _rhs.column(i), null_precedence};

      weak_ordering state =
        cudf::type_dispatcher(_lhs.column(i).type(), comparator, lhs_index, rhs_index);

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

/**
 * @brief Computes the hash value of an element in the given column.
 *
 * @tparam hash_function Hash functor to use for hashing elements.
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 */
template <template <typename> class hash_function, typename Nullate>
class element_hasher {
 public:
  /**
   * @brief Returns the hash value of the given element in the given column.
   *
   * @tparam T The type of the element to hash
   * @param col The column to hash
   * @param row_index The index of the row to hash
   * @return The hash value of the given element
   */
  template <typename T, CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  __device__ hash_value_type operator()(column_device_view col, size_type row_index) const
  {
    if (has_nulls && col.is_null(row_index)) {
      return cuda::std::numeric_limits<hash_value_type>::max();
    }
    return hash_function<T>{}(col.element<T>(row_index));
  }

  /**
   * @brief Returns the hash value of the given element in the given column.
   *
   * @tparam T The type of the element to hash
   * @param col The column to hash
   * @param row_index The index of the row to hash
   * @return The hash value of the given element
   */
  template <typename T, CUDF_ENABLE_IF(not column_device_view::has_element_accessor<T>())>
  __device__ hash_value_type operator()(column_device_view col, size_type row_index) const
  {
    CUDF_UNREACHABLE("Unsupported type in hash.");
  }

  Nullate has_nulls;  ///< A cudf::nullate type describing how to check for nulls.
};

/**
 * @brief Function object for computing the hash value of a row in a column.
 *
 * @tparam hash_function Hash functor to use for hashing elements
 * @tparam Nullate A cudf::nullate type describing how to check for nulls
 */
template <template <typename> class hash_function, typename Nullate>
class element_hasher_with_seed {
 public:
  /**
   * @brief Constructs a function object for hashing an element in the given column
   *
   * @param has_nulls Indicates if either input column contains nulls
   * @param seed The seed to use for the hash function
   */
  __device__ element_hasher_with_seed(Nullate has_nulls, uint32_t seed)
    : _seed{seed}, _has_nulls{has_nulls}
  {
  }

  /**
   * @brief Constructs a function object for hashing an element in the given column
   *
   * @param has_nulls Indicates if either input column contains nulls
   * @param seed The seed to use for the hash function
   * @param null_hash The hash value to use for null elements
   */
  __device__ element_hasher_with_seed(Nullate has_nulls, uint32_t seed, hash_value_type null_hash)
    : _seed{seed}, _null_hash{null_hash}, _has_nulls{has_nulls}
  {
  }

  /**
   * @brief Returns the hash value of the given element in the given column.
   *
   * @tparam T The type of the element to hash
   * @param col The column to hash
   * @param row_index The index of the row to hash
   * @return The hash value of the given element
   */
  template <typename T, CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  __device__ hash_value_type operator()(column_device_view col, size_type row_index) const
  {
    if (_has_nulls && col.is_null(row_index)) { return _null_hash; }
    return hash_function<T>{_seed}(col.element<T>(row_index));
  }

  /**
   * @brief Returns the hash value of the given element in the given column.
   *
   * @tparam T The type of the element to hash
   * @param col The column to hash
   * @param row_index The index of the row to hash
   * @return The hash value of the given element
   */
  template <typename T, CUDF_ENABLE_IF(not column_device_view::has_element_accessor<T>())>
  __device__ hash_value_type operator()(column_device_view col, size_type row_index) const
  {
    CUDF_UNREACHABLE("Unsupported type in hash.");
  }

 private:
  uint32_t _seed{DEFAULT_HASH_SEED};
  hash_value_type _null_hash{cuda::std::numeric_limits<hash_value_type>::max()};
  Nullate _has_nulls;
};

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * @tparam hash_function Hash functor to use for hashing elements.
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 */
template <template <typename> class hash_function, typename Nullate>
class row_hasher {
 public:
  row_hasher() = delete;

  /**
   * @brief Constructs a row_hasher object.
   *
   * @param has_nulls Indicates if either input table contains nulls
   * @param t A table_device_view to hash
   */
  CUDF_HOST_DEVICE row_hasher(Nullate has_nulls, table_device_view t)
    : _table{t}, _has_nulls{has_nulls}
  {
  }
  /**
   * @brief Constructs a row_hasher object with a seed value.
   *
   * @param has_nulls Indicates if either input table contains nulls
   * @param t A table_device_view to hash
   * @param seed A seed value to use for hashing
   */
  CUDF_HOST_DEVICE row_hasher(Nullate has_nulls, table_device_view t, uint32_t seed)
    : _table{t}, _seed(seed), _has_nulls{has_nulls}
  {
  }

  /**
   * @brief Computes the hash value of the row at `row_index` in the `table`
   *
   * @param row_index The index of the row in the `table` to hash
   * @return The hash value of the row at `row_index` in the `table`
   */
  __device__ auto operator()(size_type row_index) const
  {
    // Hash the first column w/ the seed
    auto const initial_hash = cudf::hashing::detail::hash_combine(
      hash_value_type{0},
      type_dispatcher<dispatch_storage_type>(
        _table.column(0).type(),
        element_hasher_with_seed<hash_function, Nullate>{_has_nulls, _seed},
        _table.column(0),
        row_index));

    // Hashes an element in a column
    auto hasher = [=](size_type column_index) {
      return cudf::type_dispatcher<dispatch_storage_type>(
        _table.column(column_index).type(),
        element_hasher<hash_function, Nullate>{_has_nulls},
        _table.column(column_index),
        row_index);
    };

    // Hash each element and combine all the hash values together
    return thrust::transform_reduce(
      thrust::seq,
      // note that this starts at 1 and not 0 now since we already hashed the first column
      thrust::make_counting_iterator(1),
      thrust::make_counting_iterator(_table.num_columns()),
      hasher,
      initial_hash,
      [](hash_value_type lhs, hash_value_type rhs) {
        return cudf::hashing::detail::hash_combine(lhs, rhs);
      });
  }

 private:
  table_device_view _table;
  Nullate _has_nulls;
  uint32_t _seed{DEFAULT_HASH_SEED};
};

}  // namespace CUDF_EXPORT cudf
