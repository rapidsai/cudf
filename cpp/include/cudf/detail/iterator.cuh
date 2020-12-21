/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

/**
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
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>

namespace cudf {
namespace detail {
/**
 * @brief value accessor of column with null bitmask
 * A unary functor returns scalar value at `id`.
 * `operator() (cudf::size_type id)` computes `element` and valid flag at `id`
 * This functor is only allowed for nullable columns.
 *
 * the return value for element `i` will return `column[i]`
 * if it is valid, or `null_replacement` if it is null.
 *
 * @throws cudf::logic_error if the column is not nullable.
 * @throws cudf::logic_error if column datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the column
 */
template <typename Element>
struct null_replaced_value_accessor {
  column_device_view const col;      ///< column view of column in device
  Element const null_replacement{};  ///< value returned when element is null

  /**
   * @brief constructor
   * @param[in] _col column device view of cudf column
   * @param[in] null_replacement The value to return for null elements
   */
  null_replaced_value_accessor(column_device_view const& _col, Element null_val)
    : col{_col}, null_replacement{null_val}
  {
    CUDF_EXPECTS(data_type(type_to_id<Element>()) == col.type(), "the data type mismatch");
    // verify valid is non-null, otherwise, is_valid_nocheck() will crash
    CUDF_EXPECTS(_col.nullable(), "Unexpected non-nullable column.");
  }

  CUDA_DEVICE_CALLABLE
  Element operator()(cudf::size_type i) const
  {
    return col.is_valid_nocheck(i) ? col.element<Element>(i) : null_replacement;
  }
};

/**
 * @brief validity accessor of column with null bitmask
 * A unary functor returns validity at `id`.
 * `operator() (cudf::size_type id)` computes validity flag at `id`
 * This functor is only allowed for nullable columns.
 *
 * @throws cudf::logic_error if the column is not nullable.
 */
struct validity_accessor {
  column_device_view const col;

  /**
   * @brief constructor
   * @param[in] _col column device view of cudf column
   */
  validity_accessor(column_device_view const& _col) : col{_col}
  {
    // verify valid is non-null, otherwise, is_valid() will crash
    CUDF_EXPECTS(_col.nullable(), "Unexpected non-nullable column.");
  }

  CUDA_DEVICE_CALLABLE
  bool operator()(cudf::size_type i) const { return col.is_valid_nocheck(i); }
};

/**
 * @brief Constructs an iterator over a column's values that replaces null
 * elements with a specified value.
 *
 * Dereferencing the returned iterator for element `i` will return `column[i]`
 * if it is valid, or `null_replacement` if it is null.
 * This iterator is only allowed for nullable columns.
 *
 * @throws cudf::logic_error if the column is not nullable.
 * @throws cudf::logic_error if column datatype and Element type mismatch.
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
 * @brief Constructs a pair iterator over a column's values and its validity.
 *
 * Dereferencing the returned iterator returns a `thrust::pair<Element, bool>`.
 *
 * If an element at position `i` is valid (or `has_nulls == false`), then for `p = *(iter + i)`,
 * `p.first` contains the value of the element at `i` and `p.second == true`.
 *
 * Else, if the element at `i` is null, then the value of `p.first` is undefined and `p.second ==
 * false`. `pair(column[i], validity)`. `validity` is `true` if `has_nulls=false`. `validity` is
 * validity of the element at `i` if `has_nulls=true` and the column is nullable.
 *
 * @throws cudf::logic_error if the column is nullable.
 * @throws cudf::logic_error if column datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the column
 * @tparam has_nulls boolean indicating to treat the column is nullable
 * @param column The column to iterate
 * @return auto Iterator that returns valid column elements, and validity of the
 * element in a pair
 */
template <typename Element, bool has_nulls = false>
auto make_pair_iterator(column_device_view const& column)
{
  return column.pair_begin<Element, has_nulls>();
}

/**
 * @brief Constructs an iterator over a column's validities.
 *
 * Dereferencing the returned iterator for element `i` will return the validity
 * of `column[i]`
 * This iterator is only allowed for nullable columns.
 *
 * @throws cudf::logic_error if the column is not nullable.
 *
 * @param column The column to iterate
 * @return auto Iterator that returns validities of column elements.
 */
auto inline make_validity_iterator(column_device_view const& column)
{
  return thrust::make_transform_iterator(thrust::counting_iterator<cudf::size_type>{0},
                                         validity_accessor{column});
}

/**
 * @brief value accessor for scalar with valid data.
 * The unary functor returns data of Element type of the scalar.
 *
 * @throws `cudf::logic_error` if scalar datatype and Element type mismatch.
 *
 * @tparam Element The type of return type of functor
 */
template <typename Element>
struct scalar_value_accessor {
  using ScalarType       = scalar_type_t<Element>;
  using ScalarDeviceType = scalar_device_type_t<Element>;
  ScalarDeviceType const dscalar;  ///< scalar device view

  scalar_value_accessor(scalar const& scalar_value)
    : dscalar(get_scalar_device_view(static_cast<ScalarType&>(const_cast<scalar&>(scalar_value))))
  {
    CUDF_EXPECTS(type_id_matches_device_storage_type<Element>(scalar_value.type().id()),
                 "the data type mismatch");
  }

  /**
   * @brief returns the value of the scalar.
   *
   * @throw `cudf::logic_error` if this function is called in host.
   *
   * @return value of the scalar.
   */
  CUDA_DEVICE_CALLABLE
  const Element operator()(size_type) const
  {
#if defined(__CUDA_ARCH__)
    return dscalar.value();
#else
    CUDF_FAIL("unsupported device scalar iterator operation");
#endif
  }
};

/**
 * @brief Constructs a constant device iterator over a scalar's value.
 *
 * Dereferencing the returned iterator returns a `Element`.
 *
 * For `p = *(iter + i)`, `p` is the value stored in the scalar.
 *
 * The behavior is undefined if the scalar is destroyed before iterator dereferencing.
 *
 * @throws cudf::logic_error if scalar datatype and Element type mismatch.
 * @throws cudf::logic_error if scalar is null.
 * @throws cudf::logic_error if the returned iterator is dereferenced in host
 *
 * @tparam Element The type of element in the scalar
 * @param scalar_value The scalar to iterate
 * @return auto Iterator that returns scalar value
 */
template <typename Element>
auto inline make_scalar_iterator(scalar const& scalar_value)
{
  CUDF_EXPECTS(data_type(type_to_id<Element>()) == scalar_value.type(), "the data type mismatch");
  CUDF_EXPECTS(scalar_value.is_valid(), "the scalar value must be valid");
  return thrust::make_transform_iterator(thrust::make_constant_iterator<size_type>(0),
                                         scalar_value_accessor<Element>{scalar_value});
}

/**
 * @brief pair accessor for scalar.
 * The unary functor returns a pair of data of Element type and bool validity of the scalar.
 *
 * @throws `cudf::logic_error` if scalar datatype and Element type mismatch.
 *
 * @tparam Element The type of return type of functor
 */
template <typename Element>
struct scalar_pair_accessor : public scalar_value_accessor<Element> {
  using super_t    = scalar_value_accessor<Element>;
  using value_type = thrust::pair<Element, bool>;
  scalar_pair_accessor(scalar const& scalar_value) : scalar_value_accessor<Element>(scalar_value) {}

  /**
   * @brief returns a pair with value and validity of the scalar.
   *
   * @throw `cudf::logic_error` if this function is called in host.
   *
   * @return a pair with value and validity of the scalar.
   */
  CUDA_HOST_DEVICE_CALLABLE
  const value_type operator()(size_type) const
  {
#if defined(__CUDA_ARCH__)
    return {Element(super_t::dscalar.value()), super_t::dscalar.is_valid()};
#else
    CUDF_FAIL("unsupported device scalar iterator operation");
#endif
  }
};

/**
 * @brief Constructs a constant device pair iterator over a scalar's value and its validity.
 *
 * Dereferencing the returned iterator returns a `thrust::pair<Element, bool>`.
 *
 * If scalar is valid, then for `p = *(iter + i)`, `p.first` contains
 * the value of the scalar and `p.second == true`.
 *
 * Else, if the scalar is null, then the value of `p.first` is undefined and `p.second == false`.
 *
 * The behavior is undefined if the scalar is destroyed before iterator dereferencing.
 *
 * @throws cudf::logic_error if scalar datatype and Element type mismatch.
 * @throws cudf::logic_error if the returned iterator is dereferenced in host
 *
 * @tparam Element The type of elements in the scalar
 * @tparam bool unused. This template parameter exists to enforce same
 * template interface as @ref make_pair_iterator(column_device_view const&).
 * @param scalar_value The scalar to iterate
 * @return auto Iterator that returns scalar, and validity of the scalar in a pair
 */
template <typename Element, bool = false>
auto inline make_pair_iterator(scalar const& scalar_value)
{
  CUDF_EXPECTS(type_id_matches_device_storage_type<Element>(scalar_value.type().id()),
               "the data type mismatch");
  return thrust::make_transform_iterator(thrust::make_constant_iterator<size_type>(0),
                                         scalar_pair_accessor<Element>{scalar_value});
}

}  // namespace detail
}  // namespace cudf
