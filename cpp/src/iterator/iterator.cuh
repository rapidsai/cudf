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
 * This column input iterator is designed to be able to be used as input
 * iterator for thrust and cub.
 *
 * The input iterator is implemented using thrust::iterator_adaptor
 * and thrust::counting_iterator. When creating a null supported input iterator,
 * the iterator requires pointer of data and null bitmap and an identity value
 * like below.
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
 * Examples of template parameter:
 * 1. template parameter for upcasting input iterator
 *     make_iterator<has_nulls, T, T_upcast>(...);
 *
 * 2. template parameter for upcasting + squared input iterator
 *     using T_transformer = cudf::detail::transformer_squared<T, T_upcast>;
 *     make_iterator<has_nulls, T, T_upcast, T_transformer>(...);
 *
 * 3. template parameter for using `meanvar`
 *     using T_output = cudf::detail::meanvar<T_upcast>;
 *     using T_transformer = cudf::detail::transformer_meanvar<T, T_upcast>;
 *     make_iterator<has_nulls, T, T_output, T_transformer>(...)
 *
 * 4. template parameter for custom indexed iterator
 *     using T_transformer = cudf::detail::scalar_cast_transformer<T>;
 *     using T_index = gdf_index_type;
 *     make_iterator<has_nulls, T, T, T_transformer, T_index*>(...);
 * -------------------------------------------------------------------------**/

#ifndef CUDF_ITERATOR_CUH
#define CUDF_ITERATOR_CUH

#include <cudf/cudf.h>
#include <bitmask/bit_mask.cuh>         // need for bit_mask::bit_mask_t
#include <utilities/cudf_utils.h>       // need for CUDA_HOST_DEVICE_CALLABLE
#include <utilities/error_utils.hpp>
#include <utilities/type_dispatcher.hpp>

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>

namespace cudf
{
namespace detail
{

/** -------------------------------------------------------------------------*
 * @brief intermediate struct to calculate mean and variance
 * This is an example case to output a struct from column input.
 *
 * this will be used to calculate and hold `sum of values`, 'sum of squares',
 * 'sum of valid count'.
 * Those will be used to compute `mean` (= sum / count)
 * and `variance` (= sum of squares / count - mean^2).
 *
  @tparam  T  a element data type of value and value_squared.
 * -------------------------------------------------------------------------**/
template<typename T>
struct meanvar
{
    T value;                /// the value
    T value_squared;        /// the value of squared
    gdf_index_type count;   /// the count

    CUDA_HOST_DEVICE_CALLABLE
    meanvar(T _value=0, T _value_squared=0, gdf_index_type _count=0)
    : value(_value), value_squared(_value_squared), count(_count)
    {};

    using this_t = cudf::detail::meanvar<T>;

    CUDA_HOST_DEVICE_CALLABLE
    this_t operator+(this_t const &rhs) const
    {
        return this_t(
            (this->value + rhs.value),
            (this->value_squared + rhs.value_squared),
            (this->count + rhs.count)
        );
    };

    CUDA_HOST_DEVICE_CALLABLE
    bool operator==(this_t const &rhs) const
    {
        return (
            (this->value == rhs.value) &&
            (this->value_squared == rhs.value_squared) &&
            (this->count == rhs.count)
        );
    };
};

/** -------------------------------------------------------------------------*
 * @brief Construct an instance of `T_output` using a `thrust::pair<T_element, bool>
 *
 * Uses a scalar, boolean pair to construct a new object.
 * -------------------------------------------------------------------------**/

/** -------------------------------------------------------------------------*
 * @brief Transforms a scalar by casting it to another scalar type
 *
 * By default, performs an identity cast, i.e., casts to the scalar's original type.
 *
 * A transformer for `column_input_iterator`
 * It transforms `thrust::pair<T_element, bool>` into `T_output` form.
 *
 * This struct transforms the output value as `static_cast<T_output>(value)`.
 *
 * @tparam  T_element a scalar data type of input
 * @tparam  T_output  a scalar data type of output
 * -------------------------------------------------------------------------**/
template<typename T_element, typename T_output=T_element>
struct scalar_cast_transformer
{
    CUDA_HOST_DEVICE_CALLABLE
    T_output operator() (thrust::pair<T_element, bool> const & pair)
    {
        return static_cast<T_output>(pair.first);
    };
};

/** -------------------------------------------------------------------------*
 * @brief Transforms a scalar by first casting to another type, and then squaring the result.
 * A transformer for `column_input_iterator`
 * It transforms `thrust::pair<T_element, bool>` into `T_output` form.
 *
 * This struct transforms the output value as
 * `(static_cast<T_output>(_value))^2`.
 *
 * This will be used to compute "sum of squares".
 *
 * @tparam  T_element a scalar data type of input
 * @tparam  T_output  a scalar data type of output
 * -------------------------------------------------------------------------**/
template<typename T_element, typename T_output=T_element>
struct transformer_squared
{
    CUDA_HOST_DEVICE_CALLABLE
    T_output operator() (thrust::pair<T_element, bool> pair)
    {
        T_output v = static_cast<T_output>(pair.first);
        return (v*v);
    };
};

/** -------------------------------------------------------------------------*
 * @brief Uses a scalar value to construct a `meanvar` object.
 * A transformer for `column_input_iterator`
 * It transforms `thrust::pair<T_element, bool>` into
 * `T_output = meanvar<T_output_element>` form.
 *
 * This struct transforms the value and the squared value and the count at once.
 *
 * @tparam  T_element         a scalar data type of input
 * @tparam  T_output_element  a scalar data type of the element of output
 * -------------------------------------------------------------------------**/
template<typename T_element, typename T_output_element=T_element>
struct transformer_meanvar
{
    using T_output = meanvar<T_output_element>;

    CUDA_HOST_DEVICE_CALLABLE
    T_output operator() (thrust::pair<T_element, bool> const& pair)
    {
        T_output_element v = static_cast<T_output_element>(pair.first);
        return T_output(v, v*v, (pair.second)? 1 : 0 );
    };
};

// ---------------------------------------------------------------------------
/** -------------------------------------------------------------------------*
 * @brief column input struct with/without null bitmask
 * A helper struct struct for `column_input_iterator`
 * `column_input.at(gdf_index_type id)` computes
 * `data` value and valid flag at `id` and construct and return
 * `T_output(data, valid flag)`.
 * If has_nulls = true, the valid flag is always true.
 * If has_nulls = false, the valid flag corresponds to null bitmask flag
 * at `id`, and the data value = (is_valid(id))? data[id] : identity;
 *
 * @tparam  T_element a native element type (cudf data type) of input element array
 *           and `identity` value which is used when null bitmaps flag is false.
 * @tparam  has_nulls if true, this struct holds only data array.
 *           else, this struct holds data array and bitmask array and identity value
 * -------------------------------------------------------------------------**/
template<typename T_element, bool has_nulls=true>
struct column_input;

// @overload column_input<T_mutator, T_element, false>
template<typename T_element>
struct column_input<T_element, false>{
    const T_element *data;

    CUDA_HOST_DEVICE_CALLABLE
    column_input(const T_element *_data, const bit_mask::bit_mask_t *_valid=nullptr, T_element _identity=T_element{0})
    : data(_data){};

    CUDA_HOST_DEVICE_CALLABLE
    auto at(gdf_index_type id) const {
        return thrust::make_pair(data[id], true);
    };
};

// @overload column_input<T_output, T_element, true>
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
        return is_valid(id) ?
            thrust::make_pair(data[id], true):
            thrust::make_pair(identity, false);
    };

private:
    CUDA_HOST_DEVICE_CALLABLE
    bool is_valid(gdf_index_type id) const
    {
        // `bit_mask::is_valid` never check if valid is nullptr,
        // while `gdf_is_valid` checks if valid is nullptr
        return bit_mask::is_valid(valid, id);
    }
};

/** -------------------------------------------------------------------------*
 * @brief column input iterator to support null bitmask
 * The input iterator which can be used for cub and thrust.
 * This is derived from `thrust::iterator_adaptor`
 * `T_column_input = column_input` will provide the value at index `id`
 *  with/without null bitmask.
 *
 * @tparam  T_iterator_output The output data value type.
 * @tparam  T_column_input  The input struct type of `column_input`
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
 * @brief helper function to make iterator with nulls
 * Input iterator which can be used for cub and thrust.
 *
 * @tparam T_element The cudf data type of input array
 * @tparam T_output  The cudf data type of output data or array
 * @tparam Transformer
 *                   Transforms pair(value, bool) into T_output form.
 *                   The default is `scalar_cast_transformer<T_output, T_element>`
 * @tparam Iterator_Index
 *                   The base iterator which gives the index of array.
 *                   The default is `thrust::counting_iterator`
 *
 * @param[in] data     The pointer of column data array
 * @param[in] valid    The pointer of null bitmask of column
 * @param[in] identity The identity value used when the mask value is false
 * @param[in] it       The index iterator, `thrust::counting_iterator` by default
 * -------------------------------------------------------------------------**/
template <bool has_nulls, typename T_element, typename T_output = T_element,
    typename Transformer = cudf::detail::scalar_cast_transformer<T_element, T_output>,
    typename Iterator_Index=thrust::counting_iterator<gdf_index_type> >
auto make_iterator(const T_element *data, const bit_mask::bit_mask_t *valid,
    T_element identity, Iterator_Index const it = Iterator_Index(0))
{
    using T_colunn_input = cudf::detail::column_input<T_element, has_nulls>;
    using T_iterator_output = thrust::pair<T_element, bool>;
    using T_iterator = cudf::detail::column_input_iterator<T_iterator_output, T_colunn_input, Iterator_Index>;

    // column_input constructor checks if valid is not nullptr when has_nulls = true
    return thrust::make_transform_iterator(T_iterator(T_colunn_input(data, valid, identity), it), Transformer{});
}

/** -------------------------------------------------------------------------*
 *  @overload auto make_iterator(const T_element *data,
 *                     const gdf_valid_type *valid, T_element identity,
 *                     Iterator_Index const it = Iterator_Index(0))
 * -------------------------------------------------------------------------**/
template <bool has_nulls, typename T_element, typename T_output = T_element,
    typename Transformer = cudf::detail::scalar_cast_transformer<T_element, T_output>,
    typename Iterator_Index=thrust::counting_iterator<gdf_index_type> >
auto make_iterator(const T_element *data, const gdf_valid_type *valid,
    T_element identity, Iterator_Index const it = Iterator_Index(0))
{
    return make_iterator<has_nulls, T_element, T_output, Transformer, Iterator_Index>
        (data, reinterpret_cast<const bit_mask::bit_mask_t*>(valid), identity, it);
}

/** -------------------------------------------------------------------------*
 *  @overload auto make_iterator(const gdf_column& column, T_element identity,
 *                     Iterator_Index const it = Iterator_Index(0))
 * -------------------------------------------------------------------------**/
template <bool has_nulls, typename T_element, typename T_output = T_element,
    typename Transformer = cudf::detail::scalar_cast_transformer<T_element, T_output>,
    typename Iterator_Index=thrust::counting_iterator<gdf_index_type> >
auto make_iterator(const gdf_column& column,
    T_element identity, const Iterator_Index it = Iterator_Index(0))
{
    // check the data type
    CUDF_EXPECTS(gdf_dtype_of<T_element>() == column.dtype, "the data type mismatch");

    return make_iterator<has_nulls, T_element, T_output, Transformer, Iterator_Index>
        (static_cast<const T_element*>(column.data),
        reinterpret_cast<const bit_mask::bit_mask_t*>(column.valid), identity, it);
}

} // namespace cudf

#endif