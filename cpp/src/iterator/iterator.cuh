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
 * (it_dev +id) returns static_cast<T*>( column->data )[id] if column->valid
 * at id is true, and T{0} if column->valid at id = false.
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
 *     cudf::scalar_cast_transformer<T, T_upcast> transformer{};
 *     auto it = make_iterator<has_nulls, T>(column, T{0});
 *     auto it_cast = thrust::make_transform_iterator(it, transformer);
 *
 * 3. template parameter for upcasting + squared input iterator
 *     cudf::transformer_squared<T, T_upcast> transformer{};
 *     auto it = make_iterator<has_nulls, T>(column, T{0});
 *     auto it_squared = thrust::make_transform_iterator(it, transformer);
 *
 * 4. template parameter for using `meanvar`
 *     using T_output = cudf::meanvar<T_upcast>;
 *     cudf::transformer_meanvar<T, T_upcast> transformer{};
 *     auto it_pair = make_pair_iterator<has_nulls, T>(column, T{0});
 *     auto it_meanvar = thrust::make_transform_iterator(it_pair, transformer);
 *
 * 5. template parameter for custom indexed iterator
 *     gdf_index_type *indices;
 *     auto it = make_iterator<has_nulls, T, T_index*>(column, T{0}, indices);
 * -------------------------------------------------------------------------**/

#ifndef CUDF_ITERATOR_CUH
#define CUDF_ITERATOR_CUH

#include <cudf/cudf.h>
#include <iterator/transform_unary_functions.cuh>

#include <bitmask/bit_mask.cuh>         // need for bit_mask::bit_mask_t
#include <utilities/cudf_utils.h>       // need for CUDA_HOST_DEVICE_CALLABLE
#include <utilities/error_utils.hpp>
#include <utilities/type_dispatcher.hpp>

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <type_traits>

namespace cudf
{

/** -------------------------------------------------------------------------*
 * @brief value accessor with/without null bitmask
 * A unary function returns scalar value at `id`.
 * `operator() (gdf_index_type id)` computes
 * `data` value and valid flag at `id`
 *
 * If has_nulls = false, the data is always `data[id]`
 * If has_nulls = true,  the data value = (is_valid(id))? data[id] : identity;
 *
 * @tparam  T_element cudf data type of input element array and `identity` value
 *                    which is used when null bitmaps flag is false.
 * @tparam  has_nulls if true, this struct holds only data array.
 *                    else, this struct holds data array and
 *                    bitmask array and identity value
 * -------------------------------------------------------------------------**/
template <typename T_element, bool has_nulls>
struct value_accessor;

template <typename T_element>
struct value_accessor<T_element, true>
{
  T_element const* elements{};
  bit_mask::bit_mask_t const* bitmask{};
  T_element const identity{};

  value_accessor(T_element const* e, bit_mask::bit_mask_t const* b, T_element i)
    : elements{e}, bitmask{b}, identity{i}
  {
#if  !defined(__CUDA_ARCH__)
    // verify valid is non-null, otherwise, is_valid() will crash
    CUDF_EXPECTS(b != nullptr, "non-null bit mask is required");
#endif
  }

  CUDA_HOST_DEVICE_CALLABLE
  T_element operator()(gdf_index_type i) const {
    return bit_mask::is_valid(bitmask, i) ? elements[i] : identity;
  }
};

template <typename T_element>
struct value_accessor<T_element, false>
{
  T_element const* elements{};
  value_accessor(T_element const* e, bit_mask::bit_mask_t const*, T_element) : elements{e} {}

  CUDA_HOST_DEVICE_CALLABLE
  T_element operator()(gdf_index_type i) const { return elements[i]; }
};

/** -------------------------------------------------------------------------*
 * @brief pair accessor with/without null bitmask
 * A unary function returns pair of scalar value and validity at `id`.
 * `operator() (gdf_index_type id)` computes
 * `data` value and validity at `id`
 * and return `thrust::pair<T_element, bool>(data, validity)`.
 *
 * If has_nulls = false, the validity is always true.
 * If has_nulls = true, the valid flag corresponds to null bitmask flag at `id`,
 * and the data value = (is_valid(id))? data[id] : identity;
 *
 * @tparam  T_element cudf data type of input element array and `identity` value
 *                    which is used when null bitmaps flag is false.
 * @tparam  has_nulls if true, this struct holds only data array.
 *                    else, this struct holds data array and
 *                    bitmask array and identity value
 * -------------------------------------------------------------------------**/
template <typename T_element, bool has_nulls>
struct pair_accessor;

template <typename T_element>
struct pair_accessor<T_element, true> : public value_accessor<T_element, true>
{
  pair_accessor(T_element const* e, bit_mask::bit_mask_t const* b, T_element i)
    : value_accessor<T_element, true>(e, b, i) {};

  CUDA_HOST_DEVICE_CALLABLE
  thrust::pair<T_element, bool> operator()(gdf_index_type i) const {
    return bit_mask::is_valid(this->bitmask, i) ?
        thrust::make_pair(this->elements[i], true) :
        thrust::make_pair(this->identity, false) ;
  }
};

template <typename T_element>
struct pair_accessor<T_element, false> : public value_accessor<T_element, false>
{
  pair_accessor(T_element const* e, bit_mask::bit_mask_t const* b , T_element i)
    : value_accessor<T_element, false>(e, b, i) {};

  CUDA_HOST_DEVICE_CALLABLE
  thrust::pair<T_element, bool> operator()(gdf_index_type i) const {
    return thrust::make_pair(this->elements[i], true);
  }
};

// ---------------------------------------------------------------------------
// helper functions to make iterator

/** -------------------------------------------------------------------------*
 * @brief helper function to make a cudf column iterator
 * Input iterator which can be used for cub and thrust.
 * The iterator returns same cudf data type of input: `T_element`.
 *
 * @tparam has_nulls True if the data has valid bit mask, False else
 * @tparam T_element The cudf data type of input array
 * @tparam Iterator_Index
 *                   The base iterator which gives the index of array.
 *                   The default is `thrust::counting_iterator`
 *
 * @param[in] data     The pointer of column data array
 * @param[in] valid    The pointer of null bitmask of column
 * @param[in] identity The identity value used when the mask value is false
 * @param[in] it       The index iterator, `thrust::counting_iterator` by default
 * -------------------------------------------------------------------------**/
template <bool has_nulls, typename T_element,
    typename Iterator_Index=thrust::counting_iterator<gdf_index_type> >
auto make_iterator(const T_element *data, const bit_mask::bit_mask_t *valid,
    T_element identity, Iterator_Index const it = Iterator_Index(0))
{
    return thrust::make_transform_iterator(
      it, value_accessor<T_element, has_nulls>{data, valid, identity});
}

/** -------------------------------------------------------------------------*
 * @overload auto make_iterator(const T_element *data,
 *                     const gdf_valid_type *valid, T_element identity,
 *                     Iterator_Index const it = Iterator_Index(0))
 *
 * make iterator from the pointer of null bitmask of column as gdf_valid_type
 * -------------------------------------------------------------------------**/
template <bool has_nulls, typename T_element,
    typename Iterator_Index=thrust::counting_iterator<gdf_index_type> >
auto make_iterator(const T_element *data, const gdf_valid_type *valid,
    T_element identity, Iterator_Index const it = Iterator_Index(0))
{
    return make_iterator<has_nulls, T_element, Iterator_Index>
        (data, reinterpret_cast<const bit_mask::bit_mask_t*>(valid), identity, it);
}

/** -------------------------------------------------------------------------*
 * @overload auto make_iterator(const gdf_column& column, T_element identity,
 *                     Iterator_Index const it = Iterator_Index(0))
 *
 * make iterator from a column
 * -------------------------------------------------------------------------**/
template <bool has_nulls, typename T_element,
    typename Iterator_Index=thrust::counting_iterator<gdf_index_type> >
auto make_iterator(const gdf_column& column,
    T_element identity, const Iterator_Index it = Iterator_Index(0))
{
    // check the data type
    CUDF_EXPECTS(gdf_dtype_of<T_element>() == column.dtype, "the data type mismatch");

    return make_iterator<has_nulls, T_element, Iterator_Index>
        (static_cast<const T_element*>(column.data),
        reinterpret_cast<const bit_mask::bit_mask_t*>(column.valid), identity, it);
}

/** -------------------------------------------------------------------------*
 * @brief helper function to make iterator with nulls
 * Input iterator which can be used for cub and thrust.
 * The iterator returns thrust::pair<T_element, bool>
 * This is useful for more complex logic that depends on the validity.
 * e.g. group_by.count, mean_var, sort algorism.
 *
 * @tparam has_nulls True if the data has valid bit mask, False else
 * @tparam T_element The cudf data type of input array
 * @tparam Iterator_Index
 *                   The base iterator which gives the index of array.
 *                   The default is `thrust::counting_iterator`
 *
 * @param[in] data     The pointer of column data array
 * @param[in] valid    The pointer of null bitmask of column
 * @param[in] identity The identity value used when the mask value is false
 * @param[in] it       The index iterator, `thrust::counting_iterator` by default
 * -------------------------------------------------------------------------**/
template <bool has_nulls, typename T_element,
    typename Iterator_Index=thrust::counting_iterator<gdf_index_type> >
auto make_pair_iterator(const T_element *data, const bit_mask::bit_mask_t *valid,
    T_element identity, Iterator_Index const it = Iterator_Index(0))
{
    return thrust::make_transform_iterator(
      it, pair_accessor<T_element, has_nulls>{data, valid, identity});
}

/** -------------------------------------------------------------------------*
 * @overload auto make_pair_iterator(const T_element *data,
 *                     const gdf_valid_type *valid, T_element identity,
 *                     Iterator_Index const it = Iterator_Index(0))
 *
 * make iterator from the pointer of null bitmask of column as gdf_valid_type
 * -------------------------------------------------------------------------**/
template <bool has_nulls, typename T_element,
    typename Iterator_Index=thrust::counting_iterator<gdf_index_type> >
auto make_pair_iterator(const T_element *data, const gdf_valid_type *valid,
    T_element identity, Iterator_Index const it = Iterator_Index(0))
{
    return make_pair_iterator<has_nulls, T_element, Iterator_Index>
        (data, reinterpret_cast<const bit_mask::bit_mask_t*>(valid), identity, it);
}

/** -------------------------------------------------------------------------*
 * @overload auto make_pair_iterator(const gdf_column& column, T_element identity,
 *                     Iterator_Index const it = Iterator_Index(0))
 *
 * make iterator from a column
 * -------------------------------------------------------------------------**/
template <bool has_nulls, typename T_element,
    typename Iterator_Index=thrust::counting_iterator<gdf_index_type> >
auto make_pair_iterator(const gdf_column& column,
    T_element identity, const Iterator_Index it = Iterator_Index(0))
{
    // check the data type
    CUDF_EXPECTS(gdf_dtype_of<T_element>() == column.dtype, "the data type mismatch");

    return make_pair_iterator<has_nulls, T_element, Iterator_Index>
        (static_cast<const T_element*>(column.data),
        reinterpret_cast<const bit_mask::bit_mask_t*>(column.valid), identity, it);
}


} // namespace cudf

#endif