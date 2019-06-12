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
 * The column input iterator is implemented using thrust::iterator_adaptor
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
namespace detail
{
/** -------------------------------------------------------------------------*
 * @brief column input struct with/without null bitmask
 * A helper struct for column_input_iterator
 * `column_input.at(gdf_index_type id)` computes
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
template<typename T_element, bool has_nulls=true>
struct column_input;

// @overload column_input<T_element, false>
template<typename T_element>
struct column_input<T_element, false>{
    const T_element *data;

    CUDA_HOST_DEVICE_CALLABLE
    column_input(const T_element *_data, const bit_mask::bit_mask_t *_valid=nullptr, T_element _identity=T_element{0})
    : data(_data){};

    CUDA_HOST_DEVICE_CALLABLE
    auto at(gdf_index_type id) const {
        return data[id];
    };
};

// @overload column_input<T_element, true>
template<typename T_element>
struct column_input<T_element, true>{
    const T_element *data;
    const bit_mask::bit_mask_t *valid;
    T_element identity;

    CUDA_HOST_DEVICE_CALLABLE
    column_input(const T_element *_data, const bit_mask::bit_mask_t *_valid, T_element _identity)
    : data(_data), valid(_valid), identity(_identity){
#if  !defined(__CUDA_ARCH__)
        // verify valid is non-null, otherwise, is_valid() will crash
        CUDF_EXPECTS(valid != nullptr, "non-null bit mask is required");
#endif
    };

    CUDA_HOST_DEVICE_CALLABLE
    auto at(gdf_index_type id) const {
        return (is_valid(id) ? data[id] : identity );
    };

protected:
    CUDA_HOST_DEVICE_CALLABLE
    bool is_valid(gdf_index_type id) const
    {
        // `bit_mask::is_valid` never check if valid is nullptr,
        // while `gdf_is_valid` checks if valid is nullptr
        return bit_mask::is_valid(valid, id);
    }
};

/** -------------------------------------------------------------------------*
 * @brief column input struct with/without null bitmask
 * A helper struct for column_input_iterator
 * `column_input.at(gdf_index_type id)` computes
 * `data` value and valid flag at `id`
 * and return `thrust::pair<T_element, bool>(data, valid flag)`.
 *
 * If has_nulls = false, the valid flag is always true.
 * If has_nulls = true, the valid flag corresponds to null bitmask flag
 * at `id`, and the data value = (is_valid(id))? data[id] : identity;
 *
 * @tparam  T_element cudf data type of input element array and `identity` value
 *                    which is used when null bitmaps flag is false.
 * @tparam  has_nulls if true, this struct holds only data array.
 *                    else, this struct holds data array and
 *                    bitmask array and identity value
 * -------------------------------------------------------------------------**/
template<typename T_element, bool has_nulls>
struct column_input_pair;

// @overload column_input_pair<T_element, false>
template<typename T_element>
struct column_input_pair<T_element, false> : public column_input<T_element, false>
{
    CUDA_HOST_DEVICE_CALLABLE
    column_input_pair(const T_element *_data, const bit_mask::bit_mask_t *_valid, T_element _identity)
    : column_input<T_element, false>(_data, _valid, _identity){};

    CUDA_HOST_DEVICE_CALLABLE
    auto at(gdf_index_type id) const {
        return thrust::make_pair(this->data[id], true);
    };

};

// @overload column_input_pair<T_element, true>
template<typename T_element>
struct column_input_pair<T_element, true> : public column_input<T_element, true>
{
    CUDA_HOST_DEVICE_CALLABLE
    column_input_pair(const T_element *_data, const bit_mask::bit_mask_t *_valid, T_element _identity)
    : column_input<T_element, true>(_data, _valid, _identity){};

    CUDA_HOST_DEVICE_CALLABLE
    auto at(gdf_index_type id) const {
        return this->is_valid(id) ?
            thrust::make_pair(this->data[id], true):
            thrust::make_pair(this->identity, false);
    };
};

/** -------------------------------------------------------------------------*
 * @brief column input iterator to support null bitmask
 * The input iterator which can be used for cub and thrust.
 * This is derived from `thrust::iterator_adaptor`
 * T_column_input = column_input or column_input_pair
 * T_column_input will provide the value at index `id` with/without null bitmask.
 *
 * @tparam  T_iterator_output The output data value type of the iterator
 * @tparam  T_column_input  The input struct type of column_input or column_input_pair
 * @tparam  Iterator The base iterator which gives the index of array.
 *                   The default is `thrust::counting_iterator`
 * -------------------------------------------------------------------------**/
template<typename T_iterator_output, typename T_column_input,
    typename Iterator=thrust::counting_iterator<gdf_index_type> >
  class column_input_iterator
    : public thrust::iterator_adaptor<
        column_input_iterator<T_iterator_output, T_column_input, Iterator>, // the name of the iterator we're creating
        Iterator,                   // the name of the iterator we're adapting
        T_iterator_output,          // set `super_t::value` to `T_iterator_output`
        thrust::use_default, thrust::use_default,
        T_iterator_output,          // set `super_t::reference` to `T_iterator_output`
        thrust::use_default
      >
  {
  public:
    // shorthand for the name of the iterator_adaptor we're deriving from
    using super_t = thrust::iterator_adaptor<
      column_input_iterator<T_iterator_output, T_column_input, Iterator>, Iterator,
      T_iterator_output, thrust::use_default, thrust::use_default, T_iterator_output, thrust::use_default
    >;

    CUDA_HOST_DEVICE_CALLABLE
    column_input_iterator(const T_column_input col, const Iterator &it) : super_t(it), colData(col){}

    CUDA_HOST_DEVICE_CALLABLE
    column_input_iterator(const T_column_input col) : super_t(Iterator{0}), colData(col){}

    CUDA_HOST_DEVICE_CALLABLE
    column_input_iterator(const column_input_iterator &other) : super_t(other.base()), colData(other.colData){}

    // befriend thrust::iterator_core_access to allow it access to the private interface below
    friend class thrust::iterator_core_access;

  private:
    const T_column_input colData;

    // it is private because only thrust::iterator_core_access needs access to it
    CUDA_HOST_DEVICE_CALLABLE
    typename super_t::reference dereference() const
    {
      gdf_index_type id = *(this->base()); // base() returns base iterator: `Iterator`
      return colData.at(id);
    }
};

} // namespace detail

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
    using T_colunn_input = cudf::detail::column_input<T_element, has_nulls>;
    using T_iterator_output = T_element;
    using T_iterator = cudf::detail::column_input_iterator<
        T_iterator_output, T_colunn_input, Iterator_Index>;

    // column_input constructor checks if valid is not nullptr when has_nulls = true
    return T_iterator(T_colunn_input(data, valid, identity), it);
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
    using T_colunn_input = cudf::detail::column_input_pair<T_element, has_nulls>;
    using T_iterator_output = thrust::pair<T_element, bool>;
    using T_iterator = cudf::detail::column_input_iterator<
        T_iterator_output, T_colunn_input, Iterator_Index>;

    // column_input constructor checks if valid is not nullptr when has_nulls = true
    return T_iterator(T_colunn_input(data, valid, identity), it);
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