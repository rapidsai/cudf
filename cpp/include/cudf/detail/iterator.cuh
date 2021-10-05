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

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <utility>

namespace cudf {
namespace detail {
/**
 * @brief Convenience wrapper for creating a `thrust::transform_iterator` over a
 * `thrust::counting_iterator`.
 *
 * Example:
 * @code{.cpp}
 * // Returns square of the value of the counting iterator
 * auto iter = make_counting_transform_iterator(0, [](auto i){ return (i * i);});
 * iter[0] == 0
 * iter[1] == 1
 * iter[2] == 4
 * ...
 * iter[n] == n * n
 * @endcode
 *
 * @param start The starting value of the counting iterator
 * @param f The unary function to apply to the counting iterator.
 * @return A transform iterator that applies `f` to a counting iterator
 */
template <typename UnaryFunction>
inline auto make_counting_transform_iterator(cudf::size_type start, UnaryFunction f)
{
  return thrust::make_transform_iterator(thrust::make_counting_iterator(start), f);
}

/**
 * @brief Value accessor of column that may have a null bitmask.
 *
 * This unary functor returns scalar value at `id`.
 * The `operator()(cudf::size_type id)` computes the `element` and valid flag at `id`.
 *
 * The return value for element `i` will return `column[i]`
 * if it is valid, or `null_replacement` if it is null.
 *
 * @tparam Element The type of elements in the column
 */
template <typename Element>
struct null_replaced_value_accessor {
  column_device_view const col;      ///< column view of column in device
  Element const null_replacement{};  ///< value returned when element is null
  bool const has_nulls;              ///< true if col has null elements

  /**
   * @brief Creates an accessor for a null-replacement iterator.
   *
   * @throws cudf::logic_error if `col` type does not match Element type.
   * @throws cudf::logic_error if `has_nulls` is true but `col` does not have a validity mask.
   *
   * @param[in] col column device view of cudf column
   * @param[in] null_replacement The value to return for null elements
   * @param[in] has_nulls Must be set to true if `col` has nulls.
   */
  null_replaced_value_accessor(column_device_view const& col,
                               Element null_val,
                               bool has_nulls = true)
    : col{col}, null_replacement{null_val}, has_nulls{has_nulls}
  {
    CUDF_EXPECTS(type_id_matches_device_storage_type<Element>(col.type().id()),
                 "the data type mismatch");
    if (has_nulls) CUDF_EXPECTS(col.nullable(), "column with nulls must have a validity bitmask");
  }

  CUDA_DEVICE_CALLABLE
  Element operator()(cudf::size_type i) const
  {
    return has_nulls && col.is_null_nocheck(i) ? null_replacement : col.element<Element>(i);
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
 * This iterator is only allowed for both nullable and non-nullable columns.
 *
 * @throws cudf::logic_error if the column is not nullable.
 * @throws cudf::logic_error if column datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the column
 * @param column The column to iterate
 * @param null_replacement The value to return for null elements
 * @param has_nulls Must be set to true if `column` has nulls.
 * @return Iterator that returns valid column elements, or a null
 * replacement value for null elements.
 */
template <typename Element>
auto make_null_replacement_iterator(column_device_view const& column,
                                    Element const null_replacement = Element{0},
                                    bool has_nulls                 = true)
{
  return make_counting_transform_iterator(
    0, null_replaced_value_accessor<Element>{column, null_replacement, has_nulls});
}

/**
 * @brief Constructs an optional iterator over a column's values and its validity.
 *
 * Dereferencing the returned iterator returns a `thrust::optional<Element>`.
 *
 * When the element of an iterator contextually converted to bool, the conversion returns true
 * if the object contains a value and false if it does not contain a value.
 *
 * make_optional_iterator with mode `DYNAMIC` defers the assumption of nullability to
 * runtime, with the user stating on construction of the iterator if column has nulls.
 * `DYNAMIC` mode is nice when an algorithm is going to execute on multiple
 * iterators and you don't want to compile all the combinations of iterator types
 *
 * Example:
 *
 * \code{.cpp}
 * template<typename T>
 * void some_function(cudf::column_view<T> const& col_view){
 *    auto d_col = cudf::column_device_view::create(col_view);
 *    // Create a `DYNAMIC` optional iterator
 *    auto optional_iterator = cudf::detail::make_optional_iterator<T>(d_col,
 *                                                cudf::contains_nulls::DYNAMIC{},
 *                                                col_view.has_nulls());
 * }
 * \endcode
 *
 * @throws cudf::logic_error if the column is not nullable, and `DYNAMIC` mode used and
 *         the user has stated nulls exist
 * @throws cudf::logic_error if column datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the column
 * @param column The column to iterate
 * @return Iterator that returns valid column elements and the validity of the
 * element in a thrust::optional
 */
template <typename Element>
auto make_optional_iterator(column_device_view const& column,
                            contains_nulls::DYNAMIC,
                            bool has_nulls)
{
  return column.optional_begin<Element>(contains_nulls::DYNAMIC{}, has_nulls);
}

/**
 * @brief Constructs an optional iterator over a column's values and its validity.
 *
 * Dereferencing the returned iterator returns a `thrust::optional<Element>`.
 *
 * When the element of an iterator contextually converted to bool, the conversion returns true
 * if the object contains a value and false if it does not contain a value.
 *
 * make_optional_iterator with mode `YES` means that the column supports nulls and
 * potentially has null values, therefore the optional might not contain a value
 *
 * Example:
 *
 * \code{.cpp}
 * template<typename T, bool has_nulls>
 * void some_function(cudf::column_view<T> const& col_view){
 *    auto d_col = cudf::column_device_view::create(col_view);
 *    if constexpr(has_nulls) {
 *      auto optional_iterator = cudf::detail::make_optional_iterator<T>(d_col,
 *                                                  cudf::contains_nulls::YES{});
 *      //use optional_iterator
 *    } else {
 *      auto optional_iterator = cudf::detail::make_optional_iterator<T>(d_col,
 *                                                  cudf::contains_nulls::NO{});
 *      //use optional_iterator
 *    }
 * }
 * \endcode
 *
 * @throws cudf::logic_error if the column is not nullable, and `YES` mode used
 * @throws cudf::logic_error if column datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the column
 * @param column The column to iterate
 * @return Iterator that returns column elements and the validity of the
 * element as a thrust::optional
 */
template <typename Element>
auto make_optional_iterator(column_device_view const& column, contains_nulls::YES)
{
  return column.optional_begin<Element>(contains_nulls::YES{});
}

/**
 * @brief Constructs an optional iterator over a column's values and its validity.
 *
 * Dereferencing the returned iterator returns a `thrust::optional<Element>`.
 *
 * When the element of an iterator contextually converted to bool, the conversion returns true
 * if the object contains a value and false if it does not contain a value.
 *
 * make_optional_iterator with mode `NO` means that the column has no null values,
 * therefore the optional will always contain a value.
 *
 * Example:
 *
 * \code{.cpp}
 * template<typename T, bool has_nulls>
 * void some_function(cudf::column_view<T> const& col_view){
 *    auto d_col = cudf::column_device_view::create(col_view);
 *    if constexpr(has_nulls) {
 *      auto optional_iterator = cudf::detail::make_optional_iterator<T>(d_col,
 *                                                  cudf::contains_nulls::YES{});
 *      //use optional_iterator
 *    } else {
 *      auto optional_iterator = cudf::detail::make_optional_iterator<T>(d_col,
 *                                                  cudf::contains_nulls::NO{});
 *      //use optional_iterator
 *    }
 * }
 * \endcode
 *
 * @throws cudf::logic_error if column datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the column
 * @param column The column to iterate
 * @return Iterator that returns column elements and the validity of the
 * element in a thrust::optional
 */
template <typename Element>
auto make_optional_iterator(column_device_view const& column, contains_nulls::NO)
{
  return column.optional_begin<Element>(contains_nulls::NO{});
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
 * @brief Constructs a pair rep iterator over a column's representative values and its validity.
 *
 * Dereferencing the returned iterator returns a `thrust::pair<rep_type, bool>`,
 * where `rep_type` is `device_storage_type<T>`, the type used to store
 * the value on the device.
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
auto make_pair_rep_iterator(column_device_view const& column)
{
  return column.pair_rep_begin<Element, has_nulls>();
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
  return make_counting_transform_iterator(cudf::size_type{0}, validity_accessor{column});
}

/**
 * @brief Constructs a constant device iterator over a scalar's validity.
 *
 * Dereferencing the returned iterator returns a `bool`.
 *
 * For `p = *(iter + i)`, `p` is the validity of the scalar.
 *
 * @param scalar_value The scalar to iterate
 * @return auto Iterator that returns scalar validity
 */
auto inline make_validity_iterator(scalar const& scalar_value)
{
  return thrust::make_constant_iterator(scalar_value.is_valid());
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

template <typename Element, typename contains_nulls_mode>
struct scalar_optional_accessor;

/**
 * @brief optional accessor of a maybe-nullable scalar
 *
 * The scalar_optional_accessor always returns a thrust::optional of the scalar.
 * The validity of the optional is determined by the contains_nulls_mode template parameter
 * which has the following modes:
 *
 * `DYNAMIC`: Defer nullability checks to runtime
 *
 *  - When `with_nulls=true` the return value will be a `thrust::optional{scalar}`
 *    when scalar is valid, and `thrust::optional{}` when the scalar is invalid.
 *
 *  - When `with_nulls=false` the return value will always be `thrust::optional{scalar}`
 *
 * `NO`: No null values will occur for this scalar, no checks will occur
 *  and `thrust::optional{scalar}` will always be returned.
 *
 * `YES`: null values will occur for this scalar,
 *  and `thrust::optional{scalar}` will always be returned.
 *
 * @throws `cudf::logic_error` if scalar datatype and Element type mismatch.
 *
 * @tparam Element The type of return type of functor
 */
template <typename Element, typename contains_nulls_mode>
struct scalar_optional_accessor : public scalar_value_accessor<Element> {
  using super_t    = scalar_value_accessor<Element>;
  using value_type = thrust::optional<Element>;

  scalar_optional_accessor(scalar const& scalar_value)
    : scalar_value_accessor<Element>(scalar_value)
  {
  }

  /**
   * @brief returns a thrust::optional<Element>.
   *
   * @throw `cudf::logic_error` if this function is called in host.
   *
   * @return a thrust::optional<Element> for the scalar value.
   */
  CUDA_HOST_DEVICE_CALLABLE
  const value_type operator()(size_type) const
  {
    if constexpr (std::is_same_v<contains_nulls_mode, contains_nulls::YES>) {
      return (super_t::dscalar.is_valid()) ? Element{super_t::dscalar.value()}
                                           : value_type{thrust::nullopt};
    }
    return Element{super_t::dscalar.value()};
  }
};

template <typename Element>
struct scalar_optional_accessor<Element, cudf::contains_nulls::DYNAMIC>
  : public scalar_value_accessor<Element> {
  using super_t    = scalar_value_accessor<Element>;
  using value_type = thrust::optional<Element>;
  bool has_nulls;

  scalar_optional_accessor(scalar const& scalar_value, bool with_nulls)
    : scalar_value_accessor<Element>(scalar_value), has_nulls{with_nulls}
  {
  }

  CUDA_HOST_DEVICE_CALLABLE
  const value_type operator()(size_type) const
  {
    return (has_nulls and !super_t::dscalar.is_valid()) ? value_type{thrust::nullopt}
                                                        : Element{super_t::dscalar.value()};
  }
};

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
 * @brief Utility to discard template type arguments.
 *
 * Substitute for std::void_t.
 *
 * @tparam T Ignored template parameter
 */
template <typename... T>
using void_t = void;

/**
 * @brief Compile-time reflection to check if `Element` type has a `rep()` member.
 */
template <typename Element, typename = void>
struct has_rep_member : std::false_type {
};

template <typename Element>
struct has_rep_member<Element, void_t<decltype(std::declval<Element>().rep())>> : std::true_type {
};

/**
 * @brief Pair accessor for scalar's representation value and validity.
 *
 * @tparam Element The type of element in the scalar.
 */
template <typename Element>
struct scalar_representation_pair_accessor : public scalar_value_accessor<Element> {
  using base       = scalar_value_accessor<Element>;
  using rep_type   = device_storage_type_t<Element>;
  using value_type = thrust::pair<rep_type, bool>;

  scalar_representation_pair_accessor(scalar const& scalar_value) : base(scalar_value) {}

  /**
   * @brief returns a pair with representative value and validity of the scalar.
   *
   * @throw `cudf::logic_error` if this function is called in host.
   *
   * @return a pair with representative value and validity of the scalar.
   */
  CUDA_DEVICE_CALLABLE
  const value_type operator()(size_type) const
  {
    return {get_rep(base::dscalar), base::dscalar.is_valid()};
  }

 private:
  template <typename DeviceScalar,
            std::enable_if_t<!has_rep_member<DeviceScalar>::value, void>* = nullptr>
  CUDA_DEVICE_CALLABLE rep_type get_rep(DeviceScalar const& dscalar) const
  {
    return dscalar.value();
  }

  template <typename DeviceScalar,
            std::enable_if_t<has_rep_member<DeviceScalar>::value, void>* = nullptr>
  CUDA_DEVICE_CALLABLE rep_type get_rep(DeviceScalar const& dscalar) const
  {
    return dscalar.rep();
  }
};

/**
 * @brief Constructs an optional iterator over a scalar's values and its validity.
 *
 * Dereferencing the returned iterator returns a `thrust::optional<Element>`.
 *
 * When the element of an iterator contextually converted to bool, the conversion returns true
 * if the object contains a value and false if it does not contain a value.
 *
 * The iterator behavior is undefined if the scalar is destroyed before iterator dereferencing.
 *
 * make_optional_iterator with mode `DYNAMIC` defers the assumption of nullability to
 * runtime, with the user stating on construction of the iterator if scalar has nulls.
 *
 * Example:
 *
 * \code{.cpp}
 * template<typename T>
 * void some_function(cudf::column_view<T> const& col_view,
 *                    scalar const& scalar_value,
 *                    bool col_has_nulls){
 *    auto d_col = cudf::column_device_view::create(col_view);
 *    auto column_iterator = cudf::detail::make_optional_iterator<T>(d_col,
                                      cudf::contains_nulls::DYNAMIC{}, col_has_nulls);
 *    auto scalar_iterator = cudf::detail::make_optional_iterator<T>(scalar_value,
                                      cudf::contains_nulls::DYNAMIC{}, scalar_value.is_valid());
 *    //use iterators
 * }
 * \endcode
 *
 * @throws cudf::logic_error if the scalar is not nullable, and `DYNAMIC` mode used and
 *         the user has stated nulls exist
 * @throws cudf::logic_error if scalar datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the scalar
 * @tparam has_nulls If the scalar value will have a null at runtime
 * @param scalar_value The scalar to iterate
 * @return Iterator that returns scalar elements and validity of the
 * element in a thrust::optional
 */
template <typename Element>
auto inline make_optional_iterator(scalar const& scalar_value,
                                   contains_nulls::DYNAMIC,
                                   bool has_nulls)
{
  CUDF_EXPECTS(type_id_matches_device_storage_type<Element>(scalar_value.type().id()),
               "the data type mismatch");
  return thrust::make_transform_iterator(
    thrust::make_constant_iterator<size_type>(0),
    scalar_optional_accessor<Element, contains_nulls::DYNAMIC>{scalar_value, has_nulls});
}

/**
 * @brief Constructs an optional iterator over a scalar's values and its validity.
 *
 * Dereferencing the returned iterator returns a `thrust::optional<Element>`.
 *
 * When the element of an iterator contextually converted to bool, the conversion returns true
 * if the object contains a value and false if it does not contain a value.
 *
 * The iterator behavior is undefined if the scalar is destroyed before iterator dereferencing.
 *
 * make_optional_iterator ith mode `YES` means that the scalar supports nulls and
 * potentially has null values, therefore the optional might not contain a value
 * therefore the optional will always contain a value.
 *
 * Example:
 *
 * \code{.cpp}
 * template<typename T, bool any_nulls>
 * void some_function(cudf::column_view<T> const& col_view, scalar const& scalar_value){
 *    auto d_col = cudf::column_device_view::create(col_view);
 *    if constexpr(any_nulls) {
 *      auto column_iterator = cudf::detail::make_optional_iterator<T>(d_col,
 *                                                cudf::contains_nulls::YES{});
 *      auto scalar_iterator = cudf::detail::make_optional_iterator<T>(scalar_value,
 *                                                cudf::contains_nulls::YES{});
 *      //use iterators
 *    } else {
 *      auto column_iterator = cudf::detail::make_optional_iterator<T>(d_col,
 *                                                cudf::contains_nulls::NO{});
 *      auto scalar_iterator = cudf::detail::make_optional_iterator<T>(scalar_value,
 *                                                cudf::contains_nulls::NO{});
 *      //use iterators
 *    }
 * }
 * \endcode
 *
 * @throws cudf::logic_error if the scalar is not nullable, and `YES` mode used
 * @throws cudf::logic_error if scalar datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the scalar
 * @param scalar_value The scalar to iterate
 * @return Iterator that returns scalar elements and the validity of the
 * element in a thrust::optional
 */
template <typename Element>
auto inline make_optional_iterator(scalar const& scalar_value, contains_nulls::YES)
{
  CUDF_EXPECTS(type_id_matches_device_storage_type<Element>(scalar_value.type().id()),
               "the data type mismatch");
  return thrust::make_transform_iterator(
    thrust::make_constant_iterator<size_type>(0),
    scalar_optional_accessor<Element, contains_nulls::YES>{scalar_value});
}

/**
 * @brief Constructs an optional iterator over a scalar's values and its validity.
 *
 * Dereferencing the returned iterator returns a `thrust::optional<Element>`.
 *
 * When the element of an iterator contextually converted to bool, the conversion returns true
 * if the object contains a value and false if it does not contain a value.
 *
 * The iterator behavior is undefined if the scalar is destroyed before iterator dereferencing.
 *
 * make_optional_iterator with mode `NO` means that the scalar has no null values,
 * therefore the optional will always contain a value.
 *
 * Example:
 *
 * \code{.cpp}
 * template<typename T, bool any_nulls>
 * void some_function(cudf::column_view<T> const& col_view, scalar const& scalar_value){
 *    auto d_col = cudf::column_device_view::create(col_view);
 *    if constexpr(any_nulls) {
 *      auto column_iterator = cudf::detail::make_optional_iterator<T>(d_col,
 *                                                cudf::contains_nulls::YES{});
 *      auto scalar_iterator = cudf::detail::make_optional_iterator<T>(scalar_value,
 *                                                cudf::contains_nulls::YES{});
 *      //use iterators
 *    } else {
 *      auto column_iterator = cudf::detail::make_optional_iterator<T>(d_col,
 *                                                cudf::contains_nulls::NO{});
 *      auto scalar_iterator = cudf::detail::make_optional_iterator<T>(scalar_value,
 *                                                cudf::contains_nulls::NO{});
 *      //use iterators
 *    }
 * }
 * \endcode
 *
 * @throws cudf::logic_error if scalar datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the scalar
 * @param scalar_value The scalar to iterate
 * @return Iterator that returns scalar elements and the validity of the
 * element in a thrust::optional
 */
template <typename Element>
auto inline make_optional_iterator(scalar const& scalar_value, contains_nulls::NO)
{
  CUDF_EXPECTS(type_id_matches_device_storage_type<Element>(scalar_value.type().id()),
               "the data type mismatch");
  return thrust::make_transform_iterator(
    thrust::make_constant_iterator<size_type>(0),
    scalar_optional_accessor<Element, contains_nulls::NO>{scalar_value});
}

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

/**
 * @brief Constructs a constant device pair iterator over a scalar's representative value
 *        and its validity.
 *
 * Dereferencing the returned iterator returns a `thrust::pair<Element::rep, bool>`.
 * E.g. For a valid `decimal32` row, a `thrust::pair<int32_t, bool>` is returned,
 * with the value set to the `int32_t` representative value of the decimal,
 * and validity `true`, indicating that the row is valid.
 *
 * If scalar is valid, then for `p = *(iter + i)`, `p.first` contains
 * the representative value of the scalar and `p.second == true`.
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
 * @return auto Iterator that returns scalar's representative value,
 *         and validity of the scalar in a pair
 */
template <typename Element, bool = false>
auto make_pair_rep_iterator(scalar const& scalar_value)
{
  CUDF_EXPECTS(type_id_matches_device_storage_type<Element>(scalar_value.type().id()),
               "the data type mismatch");
  return make_counting_transform_iterator(
    0, scalar_representation_pair_accessor<Element>{scalar_value});
}

}  // namespace detail
}  // namespace cudf
