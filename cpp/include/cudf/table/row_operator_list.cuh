/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
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

template <cudf::type_id t>
struct non_nested_id_to_type {
  using type = std::conditional_t<cudf::is_nested(data_type(t)), void, id_to_type<t>>;
};

namespace row {

namespace equality_hashing {

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
   * @return True if lhs and rhs are equal or if both lhs and rhs are null and nulls are configured
   * to be considered equal (`nulls_are_equal` == `null_equality::EQUAL`)
   */
  template <typename Element, CUDF_ENABLE_IF(cudf::is_equality_comparable<Element, Element>())>
  __device__ bool operator()(size_type const lhs_element_index,
                             size_type const rhs_element_index) const noexcept
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

  template <typename Element,
            CUDF_ENABLE_IF(not cudf::is_equality_comparable<Element, Element>() and
                           not cudf::is_nested<Element>())>
  __device__ bool operator()(size_type const lhs_element_index, size_type const rhs_element_index)
  {
    // TODO: make this CUDF_UNREACHABLE
    cudf_assert(false && "Attempted to compare elements of uncomparable types.");
    return false;
  }

  template <typename Element, CUDF_ENABLE_IF(cudf::is_nested<Element>())>
  __device__ bool operator()(size_type const lhs_element_index,
                             size_type const rhs_element_index) const noexcept
  {
    column_device_view lcol = lhs;
    column_device_view rcol = rhs;
    int l_start_off         = lhs_element_index;
    int r_start_off         = rhs_element_index;
    int l_end_off           = lhs_element_index + 1;
    int r_end_off           = rhs_element_index + 1;
    while (is_nested(lcol.type())) {
      if (nulls) {
        for (int i = l_start_off, j = r_start_off; i < l_end_off; ++i, ++j) {
          bool const lhs_is_null{lcol.is_null(i)};
          bool const rhs_is_null{rcol.is_null(j)};

          if (lhs_is_null and rhs_is_null) {
            if (nulls_are_equal == null_equality::UNEQUAL) { return false; }
          } else if (lhs_is_null != rhs_is_null) {
            return false;
          }
        }
      }
      if (lcol.type().id() == type_id::STRUCT) {
        lcol = lcol.child(0);
        rcol = rcol.child(0);
      } else if (lcol.type().id() == type_id::LIST) {
        auto l_list_col = detail::lists_column_device_view(lcol);
        auto r_list_col = detail::lists_column_device_view(rcol);
        for (int i = l_start_off, j = r_start_off; i < l_end_off; ++i, ++j) {
          if (l_list_col.offset_at(i + 1) - l_list_col.offset_at(i) !=
              r_list_col.offset_at(j + 1) - r_list_col.offset_at(j))
            return false;
        }
        lcol        = l_list_col.child();
        rcol        = r_list_col.child();
        l_start_off = l_list_col.offset_at(l_start_off);
        r_start_off = r_list_col.offset_at(r_start_off);
        l_end_off   = l_list_col.offset_at(l_end_off);
        r_end_off   = r_list_col.offset_at(r_end_off);
        if (l_end_off - l_start_off != r_end_off - r_start_off) { return false; }
      }
    }

    for (int i = l_start_off, j = r_start_off; i < l_end_off; ++i, ++j) {
      bool equal = type_dispatcher<non_nested_id_to_type>(
        lcol.type(), element_equality_comparator{nulls, lcol, rcol, nulls_are_equal}, i, j);
      if (not equal) { return false; }
    }
    return true;
  }

 private:
  column_device_view const lhs;
  column_device_view const rhs;
  Nullate const nulls;
  null_equality const nulls_are_equal;
};

template <typename Nullate>
class device_row_comparator {
  friend class self_eq_comparator;

  /**
   * @brief Construct a function object for performing equality comparison between the rows of two
   * tables.
   *
   * @param has_nulls Indicates if either input table contains columns with nulls.
   * @param lhs The first table
   * @param rhs The second table (may be the same table as `lhs`)
   * @param nulls_are_equal Indicates if two null elements are treated as equivalent
   */
  device_row_comparator(Nullate has_nulls,
                        table_device_view lhs,
                        table_device_view rhs,
                        null_equality nulls_are_equal = null_equality::EQUAL)
    : lhs{lhs}, rhs{rhs}, nulls{has_nulls}, nulls_are_equal{nulls_are_equal}
  {
  }

 public:
  /**
   * @brief Checks whether the row at `lhs_index` in the `lhs` table is equal to the row at
   * `rhs_index` in the `rhs` table.
   *
   * @param lhs_index The index of row in the `lhs` table to examine
   * @param rhs_index The index of the row in the `rhs` table to examine
   * @return `true` if row from the `lhs` table is equal to the row in the `rhs` table
   */
  __device__ bool operator()(size_type const lhs_index, size_type const rhs_index) const noexcept
  {
    auto equal_elements = [=](column_device_view l, column_device_view r) {
      return cudf::type_dispatcher(
        l.type(), element_equality_comparator{nulls, l, r, nulls_are_equal}, lhs_index, rhs_index);
    };

    return thrust::equal(thrust::seq, lhs.begin(), lhs.end(), rhs.begin(), equal_elements);
  }

 private:
  table_device_view const lhs;
  table_device_view const rhs;
  Nullate const nulls;
  null_equality const nulls_are_equal;
};

struct preprocessed_table {
  /**
   * @brief Preprocess table for use with row equality comparison or row hashing
   *
   * Sets up the table for use with row equality comparison or row hashing. The resulting
   * preprocessed table can be passed to the constructor of `equality_hashing::self_comparator` to
   * avoid preprocessing again.
   *
   * @param table The table to preprocess
   * @param stream The cuda stream to use while preprocessing.
   */
  static std::shared_ptr<preprocessed_table> create(table_view const& table,
                                                    rmm::cuda_stream_view stream);

 private:
  friend class self_eq_comparator;

  using table_device_view_owner =
    std::invoke_result_t<decltype(table_device_view::create), table_view, rmm::cuda_stream_view>;

  preprocessed_table(table_device_view_owner&& table,
                     std::vector<rmm::device_buffer>&& null_buffers)
    : _t(std::move(table)), _null_buffers(std::move(null_buffers))
  {
  }

  /**
   * @brief Implicit conversion operator to a `table_device_view` of the preprocessed table.
   *
   * @return table_device_view
   */
  operator table_device_view() { return *_t; }

 private:
  table_device_view_owner _t;
  std::vector<rmm::device_buffer> _null_buffers;
};

class self_eq_comparator {
 public:
  /**
   * @brief Construct an owning object for performing equality comparisons between two rows of the
   * same table.
   *
   * @param t The table to compare
   * @param stream The stream to construct this object on. Not the stream that will be used for
   * comparisons using this object.
   */
  self_eq_comparator(table_view const& t, rmm::cuda_stream_view stream)
    : d_t(preprocessed_table::create(t, stream))
  {
  }

  /**
   * @brief Construct an owning object for performing equality comparisons between two rows of the
   * same table.
   *
   * This constructor allows independently constructing a `preprocessed_table` and sharing it among
   * multiple comparators.
   *
   * @param t A table preprocessed for equality comparison
   */
  self_eq_comparator(std::shared_ptr<preprocessed_table> t) : d_t{std::move(t)} {}

  /**
   * @brief Get the comparison operator to use on the device
   *
   * Returns a binary callable, `F`, with signature `bool F(size_t, size_t)`.
   *
   * `F(i,j)` returns true if and only if row `i` compares equal to row `j`.
   *
   * @tparam Nullate Optional, A cudf::nullate type describing how to check for nulls.
   */
  template <typename Nullate>
  device_row_comparator<Nullate> device_comparator(Nullate nullate = {}) const
  {
    return device_row_comparator(nullate, *d_t, *d_t);
  }

 private:
  std::shared_ptr<preprocessed_table> d_t;
};

}  // namespace equality_hashing
}  // namespace row
}  // namespace experimental
}  // namespace cudf
