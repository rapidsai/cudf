/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/detail/row_operator/lexicographic_common.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf::detail::row::primitive {

/**
 * @brief Performs a lexicographic comparison between rows of a numeric table.
 *
 * Uses the same reduced type map as primitive row equality so that comparisons of
 * numeric rows do not instantiate comparators for strings, dictionaries, or nested types.
 * NaNs compare equivalent to other NaNs and greater than all other non-null values.
 *
 * The table and ordering metadata must remain valid for the device use of this comparator.
 */
class row_lexicographic_comparator {
 public:
  /**
   * @brief Constructs a comparator for rows in the same numeric table.
   *
   * @param has_nulls Indicates if the input contains nulls
   * @param table Device view of a table whose columns must all be numeric
   * @param column_order Per-column sort order, or an empty span for all ascending
   * @param null_precedence Per-column null order, or an empty span for all nulls before
   */
  row_lexicographic_comparator(nullate::DYNAMIC has_nulls,
                               table_device_view table,
                               device_span<order const> column_order,
                               device_span<null_order const> null_precedence)
    : _has_nulls{has_nulls},
      _table{table},
      _column_order{column_order},
      _null_precedence{null_precedence}
  {
  }

  /**
   * @brief Compares two rows in lexicographic order.
   *
   * @param lhs_index Index of the first row
   * @param rhs_index Index of the second row
   * @return Weak ordering of the first row relative to the second row
   */
  __device__ weak_ordering operator()(size_type lhs_index, size_type rhs_index) const noexcept
  {
    for (size_type i = 0; i < _table.num_columns(); ++i) {
      auto const& col     = _table.column(i);
      auto state          = weak_ordering::EQUIVALENT;
      bool compare_values = true;
      if (_has_nulls) {
        bool const lhs_is_null = col.is_null(lhs_index);
        bool const rhs_is_null = col.is_null(rhs_index);
        if (lhs_is_null or rhs_is_null) {
          auto const null_precedence =
            _null_precedence.empty() ? null_order::BEFORE : _null_precedence[i];
          state          = null_compare(lhs_is_null, rhs_is_null, null_precedence);
          compare_values = false;
        }
      }

      if (compare_values) {
        state = cudf::type_dispatcher<dispatch_primitive_type>(
          col.type(), element_comparator{}, col, lhs_index, rhs_index);
      }

      if (state == weak_ordering::EQUIVALENT) { continue; }

      bool const ascending = _column_order.empty() || _column_order[i] == order::ASCENDING;
      return ascending
               ? state
               : (state == weak_ordering::LESS ? weak_ordering::GREATER : weak_ordering::LESS);
    }
    return weak_ordering::EQUIVALENT;
  }

 private:
  struct element_comparator {
    template <typename Element>
    __device__ weak_ordering operator()(column_device_view const& col,
                                        size_type lhs_index,
                                        size_type rhs_index) const noexcept
      requires(cudf::is_numeric<Element>())
    {
      return lexicographic::sorting_physical_element_comparator{}(col.element<Element>(lhs_index),
                                                                  col.element<Element>(rhs_index));
    }

    template <typename Element>
    __device__ weak_ordering operator()(column_device_view const&,
                                        size_type,
                                        size_type) const noexcept
      requires(not cudf::is_numeric<Element>())
    {
      CUDF_UNREACHABLE("Primitive lexicographic comparison requires numeric columns.");
    }
  };

  nullate::DYNAMIC _has_nulls;
  table_device_view _table;
  device_span<order const> _column_order;
  device_span<null_order const> _null_precedence;
};

}  // namespace cudf::detail::row::primitive
