/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace row::primitive {
/**
 * @brief Performs an equality comparison between two elements in two columns.
 */
class element_equality_comparator {
 public:
  /**
   * @brief Construct type-dispatched function object for comparing equality
   * between two elements.
   *
   * @note `lhs` and `rhs` may be the same.
   *
   * @param lhs The column containing the first element
   * @param rhs The column containing the second element (may be the same as lhs)
   */
  __host__ __device__ element_equality_comparator(column_device_view lhs, column_device_view rhs)
    : lhs{lhs}, rhs{rhs}
  {
  }

  /**
   * @brief Compares the specified elements for equality.
   *
   * @param lhs_element_index The index of the first element
   * @param rhs_element_index The index of the second element
   * @return True if lhs and rhs element are equal
   */
  template <typename Element, CUDF_ENABLE_IF(cudf::is_equality_comparable<Element, Element>())>
  __device__ bool operator()(size_type lhs_element_index,
                             size_type rhs_element_index) const noexcept
  {
    return cudf::equality_compare(lhs.element<Element>(lhs_element_index),
                                  rhs.element<Element>(rhs_element_index));
  }

  // @cond
  template <typename Element, CUDF_ENABLE_IF(not cudf::is_equality_comparable<Element, Element>())>
  __device__ bool operator()(size_type lhs_element_index, size_type rhs_element_index)
  {
    CUDF_UNREACHABLE("Attempted to compare elements of uncomparable types.");
  }
  // @endcond

 private:
  column_device_view lhs;
  column_device_view rhs;
};

/**
 * @brief Performs a relational comparison between two elements in two tables.
 */
class row_equality_comparator {
 public:
  /**
   * @brief Construct a new row equality comparator object
   *
   * @param lhs The column containing the first element
   * @param rhs The column containing the second element (may be the same as lhs)
   */
  row_equality_comparator(table_device_view lhs, table_device_view rhs) : lhs{lhs}, rhs{rhs}
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
    return cudf::type_dispatcher<dispatch_storage_type>(
      lhs.begin()->type(),
      element_equality_comparator{*lhs.begin(), *rhs.begin()},
      lhs_row_index,
      rhs_row_index);
  }

 private:
  table_device_view lhs;
  table_device_view rhs;
};

/**
 * @brief Performs a relational comparison between two elements in two columns.
 */
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
   */
  __host__ __device__ element_relational_comparator(column_device_view lhs, column_device_view rhs)
    : lhs{lhs}, rhs{rhs}
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
  template <typename Element, CUDF_ENABLE_IF(cudf::is_relationally_comparable<Element, Element>())>
  __device__ weak_ordering operator()(size_type lhs_element_index,
                                      size_type rhs_element_index) const noexcept
  {
    return cudf::relational_compare(lhs.element<Element>(lhs_element_index),
                                    rhs.element<Element>(rhs_element_index));
  }

  // @cond
  template <typename Element,
            CUDF_ENABLE_IF(not cudf::is_relationally_comparable<Element, Element>())>
  __device__ weak_ordering operator()(size_type lhs_element_index, size_type rhs_element_index)
  {
    CUDF_UNREACHABLE("Attempted to compare elements of uncomparable types.");
  }
  // @endcond

 private:
  column_device_view lhs;
  column_device_view rhs;
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
 */
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
   * @param lhs The first table
   * @param rhs The second table (may be the same table as `lhs`)
   * @param column_order Optional, device array the same length as a row that
   * indicates the desired ascending/descending order of each column in a row.
   * If `nullptr`, it is assumed all columns are sorted in ascending order.
   */
  row_lexicographic_comparator(table_device_view lhs,
                               table_device_view rhs,
                               order const* column_order = nullptr)
    : _lhs{lhs}, _rhs{rhs}, _column_order{column_order}
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

      auto comparator = element_relational_comparator{_lhs.column(i), _rhs.column(i)};

      weak_ordering state = cudf::type_dispatcher<dispatch_storage_type>(
        _lhs.column(i).type(), comparator, lhs_index, rhs_index);

      if (state == weak_ordering::EQUIVALENT) { continue; }

      return state == (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
    }
    return false;
  }

 private:
  table_device_view _lhs;
  table_device_view _rhs;
  order const* _column_order{};
};  // class row_lexicographic_comparator

/**
 * @brief Function object for computing the hash value of a row in a column.
 *
 * @tparam Hash Hash functor to use for hashing elements
 */
template <template <typename> class Hash>
class element_hasher {
 public:
  /**
   * @brief Constructs a function object for hashing an element in the given column
   *
   * @param seed The seed to use for the hash function
   */
  __device__ element_hasher(uint32_t seed) : _seed{seed} {}

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
    return Hash<T>{_seed}(col.element<T>(row_index));
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
  uint32_t _seed;
};

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * @tparam Hash Hash functor to use for hashing elements.
 */
template <template <typename> class Hash>
class row_hasher {
 public:
  row_hasher() = delete;

  /**
   * @brief Constructs a row_hasher object with a seed value.
   *
   * @param t A table_device_view to hash
   * @param seed A seed value to use for hashing
   */
  CUDF_HOST_DEVICE row_hasher(table_device_view t, uint32_t seed = 0) : _table{t}, _seed{seed} {}

  /**
   * @brief Computes the hash value of the row at `row_index` in the `table`
   *
   * @param row_index The index of the row in the `table` to hash
   * @return The hash value of the row at `row_index` in the `table`
   */
  __device__ auto operator()(size_type row_index) const
  {
    return cudf::type_dispatcher<dispatch_storage_type>(
      _table.column(0).type(), element_hasher<Hash>{_seed}, _table.column(0), row_index);
  }

 private:
  table_device_view _table;
  uint32_t _seed;
};

}  // namespace row::primitive
}  // namespace CUDF_EXPORT cudf
