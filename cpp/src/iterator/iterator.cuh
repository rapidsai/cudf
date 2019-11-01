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
 * @brief provide column input iterator with/without nulls
 * @file iterator.cuh
 *
 * The column input iterator is designed to be used as an input
 * iterator for thrust and cub.
 *
 * The column input iterator is implemented using thrust::transform_iterator
 * and thrust::counting_iterator. The following example code creates
 * an input iterator for the iterator from a column with a validity (null)
 * bit mask. The second argument is the identity value.
 *
 *   template<typename T, bool has_nulls>
 *   auto it_dev = cudf::make_iterator<has_nulls, T>(column, T{0});
 *
 * (it_dev +id) returns static_cast<T*>( column.data() )[id] if 
 * column.is_null(id) is true, and T{0} if column.is_null(id) is false.
 * First index is after offset specified in column_view.
 *
 * The identity value depends on the data type and the aggregation type.
 * e.g.
 * T = int32_t and aggregation is a kind of `sum`:
 *     identity = int32_t{0}
 * T = cudf::date32 and aggregation is a kind of `max`:
 *     identity = std::numeric_limits<cudf::date32>::lowest()
 *
 * The column input iterator itself returns only a scalar value of
 * the data at id or identity value.
 * thrust::make_transform_iterator can trasform the iterator output
 * into various forms and use cases, like up casting, squared value,
 * struct of values, etc...
 *
 * Examples of use cases:
 * 1. template parameter for same precision input iterator
 *     auto it = make_iterator<has_nulls, T>(column, T{0});
 *
 * 2. template parameter for upcasting input iterator
 *     auto it = make_iterator<has_nulls, T, T_upcast>(column, T{0});
 *
 * 3. template parameter for upcasting + squared input iterator
 *     cudf::transformer_squared<T_upcast> transformer{};
 *     auto it = make_iterator<has_nulls, T, T_upcast>(column, T{0});
 *     auto it_squared = thrust::make_transform_iterator(it, transformer);
 *
 * 4. template parameter for using `meanvar`
 *     using ResultType = cudf::meanvar<T_upcast>;
 *     cudf::transformer_meanvar<T_upcast> transformer{};
 *     auto it_pair = make_pair_iterator<has_nulls, T, T_upcast>(column, T{0});
 *     auto it_meanvar = thrust::make_transform_iterator(it_pair, transformer);
 *
 * 5. template parameter for custom indexed iterator
 *     cudf::size_type *indices;
 *     auto it = make_iterator<has_nulls, T, T, T_index*>(column, T{0}, indices);
 * -------------------------------------------------------------------------**/

#pragma once

#include <cudf/cudf.h>
#include <iterator/transform_unary_functions.cuh>

#include <cudf/column/column_device_view.cuh>
#include <cudf/utilities/bit.cuh>
#include <utilities/cudf_utils.h>       // needed for CUDA_DEVICE_CALLABLE
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <type_traits>

namespace cudf {
namespace experimental {

/** -------------------------------------------------------------------------*
 * @brief value accessor with/without null bitmask
 * A unary function returns scalar value at `id`.
 * `operator() (cudf::size_type id)` computes
 * `data` value and valid flag at `id`
 *
 * If has_nulls = false, the data is always `static_cast<ResultType>(data[id])`
 * If has_nulls = true,  the data value = (is_valid(id))?
 *      static_cast<ResultType>(data[id]) : identity;
 *
 * @tparam  ElementType cudf data type of input element array
 * @tparam  ResultType  cudf data type of output and `identity` value
 *                    which is used when null bitmaps flag is false.
 * @tparam  has_nulls if false, this struct holds only data array.
 *                    else, this struct holds data array and
 *                    bitmask array and identity value
 * -------------------------------------------------------------------------**/
template <typename ElementType, typename ResultType, bool has_nulls>
struct value_accessor;

/** -------------------------------------------------------------------------*
 * @overload value_accessor<ElementType, ResultType, true>
 * @brief specialization for columns that contain null values
 * -------------------------------------------------------------------------**/
template <typename ElementType, typename ResultType>
struct value_accessor<ElementType, ResultType, true>
{
  column_device_view const col;         ///< column view of column in device
  ResultType const identity{};            ///< identity value used when the validity is false

/** -------------------------------------------------------------------------*
 * @brief constructor
 * @param[in] _col column device view of cudf column
 * @param[in] i identity value used when the validity is false at operator()
 * -------------------------------------------------------------------------**/
  value_accessor(column_device_view const _col, ResultType i)
    : col{_col}, identity{i}
  {
    // verify valid is non-null, otherwise, is_valid() will crash
    CUDF_EXPECTS(_col.null_mask() != nullptr, "non-null bit mask is required");
  }

  CUDA_DEVICE_CALLABLE
  ResultType operator()(cudf::size_type i) const {
    return col.is_valid_nocheck(i) ?
      static_cast<ResultType>(col.element<ElementType>(i)) : identity;
  }
};

/** -------------------------------------------------------------------------*
 * @overload value_accessor<ElementType, ResultType, false>
 * @brief specialization for columns that don't contain null values
 * -------------------------------------------------------------------------**/
template <typename ElementType, typename ResultType>
struct value_accessor<ElementType, ResultType, false>
{
  column_device_view const col;         ///< column view of column in device
  ResultType const identity{};            ///< identity value used when the validity is false

/** -------------------------------------------------------------------------*
 * @brief constructor
 * @param[in] _col column device view of cudf column
 * @param[in] i not used
 * -------------------------------------------------------------------------**/
  value_accessor(column_device_view const _col, ResultType i)
    : col{_col}, identity{i} {}

  CUDA_DEVICE_CALLABLE
  ResultType operator()(cudf::size_type i) const {
    return static_cast<ResultType>(col.element<ElementType>(i));
  }
};

/** -------------------------------------------------------------------------*
 * @brief pair accessor with/without null bitmask
 * A unary function returns `thrust::pair<ResultType, bool>`. 
 * If the element at index `i` is valid, returns `ResultType{data[i]}` and `true`
 * indicating the value was valid. If the element at `i` is null,
 * returns `ResultType{identity}` and `false` indicating the element was null.
 *
 * If has_nulls = false, the validity is always true.
 * If has_nulls = true, the valid flag corresponds to null bitmask flag at `id`,
 * and the data value = (is_valid(id))? static_cast<ResultType>(data[id]) : identity;
 *
 * @tparam  ElementType cudf data type of input element array
 * @tparam  ResultType  cudf data type of output and `identity` value
 *                      which is used when null bitmaps flag is false.
 * @tparam  has_nulls   if false, this struct holds only data array.
 *                      else, this struct holds data array and
 *                      bitmask array and identity value
 * -------------------------------------------------------------------------**/
template <typename ElementType, typename ResultType, bool has_nulls>
struct pair_accessor;

/** -------------------------------------------------------------------------*
 * @overload pair_accessor<ElementType, ResultType, true>
 * @brief specialization for columns that contain null values
 * -------------------------------------------------------------------------**/
template <typename ElementType, typename ResultType>
struct pair_accessor<ElementType, ResultType, true> : public value_accessor<ElementType, ResultType, true>
{
/** -------------------------------------------------------------------------*
 * @brief constructor
 * @param[in] _col column device view of cudf column
 * @param[in] i identity value used when the validity is false at operator()
 * -------------------------------------------------------------------------**/
  pair_accessor(column_device_view const _col, ResultType i)
    : value_accessor<ElementType, ResultType, true>(_col, i) {};

  CUDA_DEVICE_CALLABLE
  thrust::pair<ResultType, bool> operator()(cudf::size_type i) const {
    return this->col.is_valid_nocheck(i) ?
        thrust::make_pair(static_cast<ResultType>(this->col.template element<ElementType>(i)), true) :
        thrust::make_pair(this->identity, false) ;
  }
};

/** -------------------------------------------------------------------------*
 * @overload pair_accessor<ElementType, ResultType, false>
 * @brief specialization for columns that don't contain null values
 * -------------------------------------------------------------------------**/
template <typename ElementType, typename ResultType>
struct pair_accessor<ElementType, ResultType, false> : public value_accessor<ElementType, ResultType, false>
{
/** -------------------------------------------------------------------------*
 * @brief constructor
 * @param[in] _col column device view of cudf column
 * @param[in] i not used
 * -------------------------------------------------------------------------**/
  pair_accessor(column_device_view const _col, ResultType i)
    : value_accessor<ElementType, ResultType, false>(_col, i) {};

  CUDA_DEVICE_CALLABLE
  thrust::pair<ResultType, bool> operator()(cudf::size_type i) const {
    return thrust::make_pair(static_cast<ResultType>(this->col.template element<ElementType>(i)), true);
  }
};
} // namespace experimental

// ---------------------------------------------------------------------------
// helper functions to make iterator

/** -------------------------------------------------------------------------*
 * @brief Constructs an iterator over the elements of a column.
 *
 * If the column contains no null values (indicated by `has_nulls == false`) 
 * then dereferencing an iterator `it` returned by this function as `*(it + n)` 
 * will return `ResultType{ static_cast<ElementType*>(data)[n] }`.
 * 
 * If the column contains null values (indicated by `has_nulls == true`) 
 * then the result of de-referencing an iterator `it` returned by this function
 * as `*(it+n)` will depend if element is valid or null. 
 * If the element is valid, 
 * it will return `ResultType{ static_cast<ElementType*>(data)[n] }`.
 * If the element is null, it will return `ResultType{identity}`.
 *
 * @tparam has_nulls Indicates if the column contains null values 
 *                   (`null_count > 0`)
 * @tparam ElementType The cudf data type of input array
 * @tparam ResultType  cudf data type of output and `identity` value
 *                     which is used when null bitmaps flag is false.
 * @tparam Iterator_Index
 *                     The base iterator which gives the index of array.
 *                     The default is `thrust::counting_iterator`
 *
 * @param[in] column   The column device view of cudf column 
 * @param[in] identity The identity value used when the mask value is false
 * @param[in] it       The index iterator, `thrust::counting_iterator` by default
 * -------------------------------------------------------------------------**/
template <bool has_nulls, typename ElementType, typename ResultType = ElementType,
    typename Iterator_Index=thrust::counting_iterator<cudf::size_type> >
auto make_iterator(const column_device_view& column,
    ResultType identity=ResultType{0}, Iterator_Index const it = Iterator_Index(0))
{
    // check the data type
    CUDF_EXPECTS(data_type(experimental::type_to_id<ElementType>()) == column.type(), "the data type mismatch");

    CUDF_EXPECTS(column.head() != nullptr, "non-null data is required");
    CUDF_EXPECTS(not ( has_nulls == true && !column.nullable() ), "non-null bit mask is required");

    return thrust::make_transform_iterator(
      it, experimental::value_accessor<ElementType, ResultType, has_nulls>{column, identity});
}

/** -------------------------------------------------------------------------*
 * @brief Constructs an iterator over the elements of a column
 * Input iterator which can be used for cub and thrust.
 * 
 * The iterator returns thrust::pair<ResultType, bool>
 * This is useful for more complex logic that depends on the validity.
 * e.g. group_by.count, mean_var, sort algorism.
 *
 * @tparam has_nulls   True if the data has valid bit mask, False else
 * @tparam ElementType The cudf data type of input array
 * @tparam ResultType  cudf data type of output and `identity` value
 *                     which is used when null bitmaps flag is false.
 * @tparam Iterator_Index
 *                     The base iterator which gives the index of array.
 *                     The default is `thrust::counting_iterator`
 *
 * @param[in] column   The column device view of cudf column 
 * @param[in] identity The identity value used when the mask value is false
 * @param[in] it       The index iterator, `thrust::counting_iterator` by default
 * -------------------------------------------------------------------------**/
template <bool has_nulls, typename ElementType, typename ResultType = ElementType,
    typename Iterator_Index=thrust::counting_iterator<cudf::size_type> >
auto make_pair_iterator(const column_device_view& column,
    ResultType identity=ResultType{0}, Iterator_Index const it = Iterator_Index(0))
{
    // check the data type
    CUDF_EXPECTS(data_type(experimental::type_to_id<ElementType>()) == column.type(), "the data type mismatch");

    CUDF_EXPECTS(column.head() != nullptr, "non-null data is required");
    CUDF_EXPECTS(not ( has_nulls == true && !column.nullable() ), "non-null bit mask is required");
	
    return thrust::make_transform_iterator(
      it, experimental::pair_accessor<ElementType, ResultType, has_nulls>{column, identity});
}


} // namespace cudf

