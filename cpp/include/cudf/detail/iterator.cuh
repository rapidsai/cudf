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

#include <cudf/cudf.h>
#include <cudf/column/column_device_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cudf {

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
 *
 * @tparam Element The type of elements in the column
 * -------------------------------------------------------------------------**/
template <typename Element>
struct column_device_view::null_replaced_value_accessor
{
  column_device_view const col;         ///< column view of column in device
  Element const null_replacement{};     ///< value returned when element is null

/** -------------------------------------------------------------------------*
 * @brief constructor
 * @param[in] _col column device view of cudf column
 * @param[in] null_replacement The value to return for null elements
 * -------------------------------------------------------------------------**/
  null_replaced_value_accessor(column_device_view const& _col, Element null_val)
    : col{_col}, null_replacement{null_val}
  {
    // verify valid is non-null, otherwise, is_valid() will crash
    CUDF_EXPECTS(_col.nullable(), "Unexpected non-nullable column.");
  }

  CUDA_DEVICE_CALLABLE
  Element operator()(cudf::size_type i) const {
    return col.is_valid_nocheck(i) ?
      col.element<Element>(i) : null_replacement;
  }
};

/** -------------------------------------------------------------------------*
 * @brief value accessor of column without null bitmask
 * A unary functor returns scalar value at `id`.
 * `operator() (cudf::size_type id)` computes `element`
 * This functor is only allowed for non-nullable columns.
 *
 * the return value for element `i` will return `column[i]`
 *
 * @throws `cudf::logic_error` if the column is nullable.
 *
 * @tparam Element The type of elements in the column
 * -------------------------------------------------------------------------**/

template <typename Element>
struct column_device_view::value_accessor
{
  column_device_view const col;         ///< column view of column in device

/** -------------------------------------------------------------------------*
 * @brief constructor
 * @param[in] _col column device view of cudf column
 * -------------------------------------------------------------------------**/
  value_accessor(column_device_view const& _col)
    : col{_col}
  {
    // verify valid is null
    CUDF_EXPECTS(!_col.nullable(), "Unexpected nullable column.");
  }

  CUDA_DEVICE_CALLABLE
  Element operator()(cudf::size_type i) const {
    return col.element<Element>(i);
  }
};

template <typename T>
column_device_view::const_iterator<T> column_device_view::begin() const {
  CUDF_EXPECTS(data_type(experimental::type_to_id<T>()) == type(), "the data type mismatch");
  return column_device_view::const_iterator<T>{
      column_device_view::count_it{0},
      column_device_view::value_accessor<T>{*this}};
}
template <typename T>
column_device_view::const_iterator<T> column_device_view::end() const {
  CUDF_EXPECTS(data_type(experimental::type_to_id<T>()) == type(), "the data type mismatch");
  return column_device_view::const_iterator<T>{
      column_device_view::count_it{size()},
      column_device_view::value_accessor<T>{*this}};
}

template <typename T>
column_device_view::const_null_iterator<T> column_device_view::nbegin(const T& null_val) const {
  CUDF_EXPECTS(data_type(experimental::type_to_id<T>()) == type(), "the data type mismatch");
  return column_device_view::const_null_iterator<T>{
      column_device_view::count_it{0},
      column_device_view::null_replaced_value_accessor<T>{*this, null_val}};
}

template <typename T>
column_device_view::const_null_iterator<T> column_device_view::nend(const T& null_val) const {
  CUDF_EXPECTS(data_type(experimental::type_to_id<T>()) == type(), "the data type mismatch");
  return column_device_view::const_null_iterator<T>{
      column_device_view::count_it{size()},
      column_device_view::null_replaced_value_accessor<T>{*this, null_val}};
}

namespace experimental {
namespace detail {
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
  CUDF_EXPECTS(data_type(experimental::type_to_id<Element>()) == column.type(), "the data type mismatch");

  return thrust::make_transform_iterator(
      thrust::counting_iterator<cudf::size_type>{0},
      column_device_view::null_replaced_value_accessor<Element>{column, null_replacement});
}


/**
 * @brief Constructs an iterator over a column's values
 *
 * Dereferencing the returned iterator for element `i` will return `column[i]`
 * This iterator is only allowed for non-nullable columns.
 *
 * @throws `cudf::logic_error` if the column is nullable.
 * @throws `cudf::logic_error` if column datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the column
 * @param column The column to iterate
 * @return auto Iterator that returns valid column elements
 */
template <typename Element>
auto make_no_null_iterator(column_device_view const& column)
{
  CUDF_EXPECTS(data_type(experimental::type_to_id<Element>()) == column.type(), "the data type mismatch");

  return thrust::make_transform_iterator(
      thrust::counting_iterator<cudf::size_type>{0},
      column_device_view::value_accessor<Element>{column});
}

} //namespace detail
} //namespace experimental
} //namespace cudf
