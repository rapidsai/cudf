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


/** --------------------------------------------------------------------------*
 * @brief provides column input iterator with nulls replaced with a specified value
 * @file iterator.cuh
 * 
 * The column input iterator is designed to be used as an input
 * iterator for thrust and cub.
 *
 * Usage:
 * auto iter = make_null_replacement_iterator(column, null_value);
 * 
 * The column input iterator returns only a scalar value of data at [id] or 
 * the null_replacement value passed while creating the iterator.
 * For non-null column, use 
 * auto iter = column.begin<Element>();
 * 
 * -------------------------------------------------------------------------**/

#pragma once

#include <cudf/column/column_device_view.cuh>

namespace cudf {
namespace experimental {
namespace detail {

/** -------------------------------------------------------------------------*
 * @brief value accessor of column with null bitmask
 * A unary functor returns scalar value at `id`.
 * `operator() (cudf::size_type id)` computes `element` and valid flag at `id`
 * This functor is only allowed for nullable columns.
 *
 * the return value for element `i` will return `column[i]`
 * if it is valid, or `null_replacement` if it is null.
 *
 * @throws `cudf::logic_error` if the column is not nullable.
 * @throws `cudf::logic_error` if column datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the column
 * -------------------------------------------------------------------------**/
template <typename Element>
struct null_replaced_value_accessor
{
  column_device_view const col;      ///< column view of column in device
  Element const null_replacement{};  ///< value returned when element is null

  /** -------------------------------------------------------------------------*
   * @brief constructor
   * @param[in] _col column device view of cudf column
   * @param[in] null_replacement The value to return for null elements
   * -------------------------------------------------------------------------**/
  null_replaced_value_accessor(column_device_view const& _col, Element null_val)
      : col{_col}, null_replacement{null_val}
  {
    CUDF_EXPECTS(data_type(experimental::type_to_id<Element>()) == col.type(),
                 "the data type mismatch");
    // verify valid is non-null, otherwise, is_valid_nocheck() will crash
    CUDF_EXPECTS(_col.nullable(), "Unexpected non-nullable column.");
  }

  CUDA_DEVICE_CALLABLE
  Element operator()(cudf::size_type i) const {
    return col.is_valid_nocheck(i) ? col.element<Element>(i) : null_replacement;
  }
};

/** -------------------------------------------------------------------------*
 * @brief validity accessor of column with null bitmask
 * A unary functor returns validity at `id`.
 * `operator() (cudf::size_type id)` computes validity flag at `id`
 * This functor is only allowed for nullable columns.
 *
 * @throws `cudf::logic_error` if the column is not nullable.
 * -------------------------------------------------------------------------**/
struct validity_accessor {
  column_device_view const col;

  /** -------------------------------------------------------------------------*
   * @brief constructor
   * @param[in] _col column device view of cudf column
   * -------------------------------------------------------------------------**/
  validity_accessor(column_device_view const& _col)
    : col{_col}
  {
    // verify valid is non-null, otherwise, is_valid() will crash
    CUDF_EXPECTS(_col.nullable(), "Unexpected non-nullable column.");
  }

  CUDA_DEVICE_CALLABLE
  bool operator()(cudf::size_type i) const {
    return col.is_valid_nocheck(i);
  }
};

/**
 * @brief Constructs an iterator over a column's values that replaces null
 * elements with a specified value.
 *
 * Dereferencing the returned iterator for element `i` will return `column[i]`
 * if it is valid, or `null_replacement` if it is null.
 * This iterator is only allowed for nullable columns.
 *
 * @throws `cudf::logic_error` if the column is not nullable.
 * @throws `cudf::logic_error` if column datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the column
 * @param column The column to iterate
 * @param null_replacement The value to return for null elements
 * @return auto Iterator that returns valid column elements, or a null
 * replacement value for null elements.
 */
template <typename Element>
auto make_null_replacement_iterator(column_device_view const& column,
                                    Element const null_replacement = Element{0})
{
  return thrust::make_transform_iterator(
      thrust::counting_iterator<cudf::size_type>{0},
      null_replaced_value_accessor<Element>{column, null_replacement});
}

/**
 * @brief Constructs an iterator over a column's validities.
 *
 * Dereferencing the returned iterator for element `i` will return the validity
 * of `column[i]`
 * This iterator is only allowed for nullable columns.
 *
 * @throws `cudf::logic_error` if the column is not nullable.
 *
 * @param column The column to iterate
 * @return auto Iterator that returns validities of column elements.
 */
auto inline make_validity_iterator(column_device_view const& column)
{
  return thrust::make_transform_iterator(
      thrust::counting_iterator<cudf::size_type>{0},
      validity_accessor{column});
}

} //namespace detail
} //namespace experimental
} //namespace cudf
