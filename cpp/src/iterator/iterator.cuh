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
 * auto it_dev = cudf::make_iterator_with_nulls(static_cast<T*>( column->data ),
 *                                              column->valid, T{0});
 *
 * (it_dev +id) returns static_cast<T*>( column->data )[id] if column->valid
 * at id is true, and T{0} if column->valid at id = false.
 *
 * The identity value depends on the data type and the aggregation type.
 * e.g.
 * T = int32_t and aggregation is a kind of `sum`:
 *     identity = int32_t{0}
 * T = cudf::date32  and aggregation is a kind of `max`:
 *     identity = std::numeric_limits<cudf::date32>::lowest()
 *
 * Examples of template parameter:
 * 1. template parameter for upcasting input iterator
 *     make_iterator_xxx<T, T_upcast>(...)
 *
 * 2. template parameter for upcasting + squared input iterator
 *     make_iterator_xxx<T, T_upcast, cudf::mutator_squared<T_upcast>>(...)
 *
 * 3. template parameter for using `mutator_meanvar`
 *     using T_output = cudf::mutator_meanvar<T_upcast>
 *     make_iterator_xxx<T, T_output, T_output>(...)
 *
 * 4. template parameter for custom indexed iterator
 *     using out_helper = cudf::ColumnOutput<T>;
 *     using T_index = gdf_index_type;
 *     make_iterator_xxx<T, T, out_helper, T_index*>(...)
 * -------------------------------------------------------------------------**/

#ifndef CUDF_ITERATOR_CUH
#define CUDF_ITERATOR_CUH

#include <cudf.h>
#include <bitmask/bit_mask.cuh>         // need for bit_mask::bit_mask_t
#include <utilities/cudf_utils.h>       // need for CUDA_HOST_DEVICE_CALLABLE
#include <utilities/error_utils.hpp>
#include <utilities/type_dispatcher.hpp>

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf
{
namespace detail
{
// --------------------------------------------------------------------------------------------------------
// helper structs of output for column_input_iterator

/** -------------------------------------------------------------------------*
 * @brief helper struct to output a scalar value
 * A helper output struct for `column_input_iterator`
 * `column_input_iterator` creates this struct with parameter `_value`, `is_valid`
 *
 * This struct computes the output value as `static_cast<T>(_value)`.
 *
 * @tparam  T a scalar data type to be output
 * @tparam  T_input a scalar data type to be input
 * -------------------------------------------------------------------------**/
template<typename T>
struct mutator_single
{
    T value;    // the value to output

    template<typename T_input>
    CUDA_HOST_DEVICE_CALLABLE
    mutator_single(T_input _value, bool is_valid=false)
    : value( static_cast<T>(_value) )
    {};

    CUDA_HOST_DEVICE_CALLABLE
    mutator_single(){};

    CUDA_HOST_DEVICE_CALLABLE
    operator T() const { return value; }
};

/** -------------------------------------------------------------------------*
 * @brief helper struct to output a squared scalar value
 * A helper output struct for `column_input_iterator`
 * `column_input_iterator` creates this struct with parameter `_value`, `is_valid`
 *
 * This struct computes the output value as
 * `static_cast<T>(_value)*static_cast<T>(_value)`.
 *
 * This will be used to compute "sum of squares".
 *
 * @tparam  T a scalar data type to be output
 * @tparam  T_input a scalar data type to be input
 * -------------------------------------------------------------------------**/
template<typename T>
struct mutator_squared
{
    T value_squared;    /// the value of squared

    template<typename T_input>
    CUDA_HOST_DEVICE_CALLABLE
    mutator_squared(T_input _value, bool is_valid=false)
    {
        T v = static_cast<T>(_value);
        value_squared = v*v;
    };

    CUDA_HOST_DEVICE_CALLABLE
    mutator_squared(){};

    CUDA_HOST_DEVICE_CALLABLE
    operator T() const { return value_squared; }
};


/** -------------------------------------------------------------------------*
 * @brief helper struct to output a  squared scalar value
 * A helper output struct for `column_input_iterator`
 * `column_input_iterator` creates this struct with parameter `_value`, `is_valid`
 *
 * This struct computes the value and the squared value and the count at once
 * where the count is updated only if update_count = true.
 *
 * This is an example case to output a struct from column input.
 * The result of this will be used to compute mean (= sum / count)
 * and variance (= sum of squares / count - mean^2).
 * `update_count=false` will be used when the count is known before computation.
 *
 * @tparam  T a scalar data type to be output
 * @tparam  update_count if true, `count` is updated. else, count` is not updated
 * @tparam  T_input a scalar data type to be input
 * -------------------------------------------------------------------------**/
template<typename T, bool update_count=true>
struct mutator_meanvar;

// @overload mutator_meanvar<T, true>
template<typename T>
struct mutator_meanvar<T, true>
{
    T value;                /// the value
    T value_squared;        /// the value of squared
    gdf_index_type count;   /// the count

    template<typename T_input>
    CUDA_HOST_DEVICE_CALLABLE
    mutator_meanvar(T_input _value, bool is_valid)
    : value( static_cast<T>(_value) ), count(is_valid? 1 : 0)
    {
        value_squared = value*value;
    };

    CUDA_HOST_DEVICE_CALLABLE
    mutator_meanvar(T _value, T _value_squared=0, gdf_index_type _count=0)
    : value(_value), value_squared(_value_squared), count(_count)
    {};


    CUDA_HOST_DEVICE_CALLABLE
    mutator_meanvar()
    : value(0), value_squared(0), count(0)
    {};

    using this_t = cudf::detail::mutator_meanvar<T, true>;

    CUDA_HOST_DEVICE_CALLABLE
    this_t operator+(this_t const &rhs) const
    {
        return this_t(
            (this->value + rhs.value),
            (this->value_squared + rhs.value_squared),
            (this->count + rhs.count)
        );
    }

    CUDA_HOST_DEVICE_CALLABLE
    bool operator==(this_t const &rhs) const
    {
        return (
            (this->value == rhs.value) &&
            (this->value_squared == rhs.value_squared) &&
            (this->count == rhs.count)
        );
    }
};

// @overload mutator_meanvar<T, false>
template<typename T>
struct mutator_meanvar<T, false>
{
    T value;
    T value_squared;
    gdf_index_type count;

    template<typename T_input>
    CUDA_HOST_DEVICE_CALLABLE
    mutator_meanvar(T_input _value, bool is_valid)
    : value( static_cast<T>(_value) )
    {
        value_squared = value*value;
    };

    CUDA_HOST_DEVICE_CALLABLE
    mutator_meanvar(T _value, T _value_squared=0, gdf_index_type _count=0)
    : value(_value), value_squared(_value_squared), count(_count)
    {};

    CUDA_HOST_DEVICE_CALLABLE
    mutator_meanvar()
    : value(0), value_squared(0), count(0)
    {};

    using this_t = cudf::detail::mutator_meanvar<T, false>;

    CUDA_HOST_DEVICE_CALLABLE
    this_t operator+(this_t const &rhs) const
    {
        // count won't be updated.
        return this_t(
            (this->value + rhs.value),
            (this->value_squared + rhs.value_squared),
            (this->count)
        );
    }

    CUDA_HOST_DEVICE_CALLABLE
    bool operator==(this_t const &rhs) const
    {
        // count won't be compared.
        return (
            (this->value == rhs.value) &&
            (this->value_squared == rhs.value_squared)
        );
    }
};

// --------------------------------------------------------------------------------------------------------
/** -------------------------------------------------------------------------*
 * @brief input struct with/without null bitmask
 * A helper struct struct for `column_input_iterator`
 * `ColumnInput.at(gdf_index_type id)` computes
 * `data` value and valid flag at `id` and construct and return
 * `T_output(data, valid flag)`.
 * If nulls_present = true, the valid flag is always true.
 * If nulls_present = false, the valid flag corresponds to null bitmask flag
 * at `id`, and the data value = (is_valid(id))? data[id] : identity;
 *
 * @tparam  T_output a scalar data type to be output
 * @tparam  T_element a native element type (cudf dta type) of input element array
 *           and `identity` value which is used when null bitmaps flag is false.
 * @tparam  nulls_present if true, this struct holds only data array.
 *            else, this struct holds data array and bitmask array and identity value
 * -------------------------------------------------------------------------**/
template<typename T_output, typename T_element, bool nulls_present=true>
struct ColumnInput;

// @overload ColumnInput<T_output, T_element, false>
template<typename T_output, typename T_element>
struct ColumnInput<T_output, T_element, false>{
    const T_element *data;

    CUDA_HOST_DEVICE_CALLABLE
    ColumnInput(const T_element *_data, const bit_mask::bit_mask_t *_valid=nullptr, T_element _identity=T_element{0})
    : data(_data){};

    CUDA_HOST_DEVICE_CALLABLE
    ColumnInput(const T_element *_data, const gdf_valid_type*_valid, T_element _identity)
    : ColumnInput(_data, reinterpret_cast<const bit_mask::bit_mask_t*>(_valid), _identity) {};

    CUDA_HOST_DEVICE_CALLABLE
    T_output at(gdf_index_type id) const {
        return T_output(data[id], true);
    };
};

// @overload ColumnInput<T_output, T_element, true>
template<typename T_output, typename T_element>
struct ColumnInput<T_output, T_element, true>{
    const T_element *data;
    const bit_mask::bit_mask_t *valid;
    T_element identity;

    CUDA_HOST_DEVICE_CALLABLE
    ColumnInput(const T_element *_data, const bit_mask::bit_mask_t *_valid, T_element _identity)
    : data(_data), valid(_valid), identity(_identity){
#if  !defined(__CUDA_ARCH__)
        // verify valid is non-null, otherwise, is_valid() will crash
        CUDF_EXPECTS(valid != nullptr, "non-null bit mask is required");
#endif
    };

    CUDA_HOST_DEVICE_CALLABLE
    ColumnInput(const T_element *_data, const gdf_valid_type*_valid, T_element _identity)
    : ColumnInput(_data, reinterpret_cast<const bit_mask::bit_mask_t*>(_valid), _identity) {};

    CUDA_HOST_DEVICE_CALLABLE
    T_output at(gdf_index_type id) const {
        return T_output(get_value(id), is_valid(id));
    };

private:
    CUDA_HOST_DEVICE_CALLABLE
    T_element get_value(gdf_index_type id) const {
        return (is_valid(id))? data[id] : identity;
    }

    CUDA_HOST_DEVICE_CALLABLE
    bool is_valid(gdf_index_type id) const
    {
        // `bit_mask::is_valid` never check if valid is nullptr,
        // while `gdf_is_valid` checks if valid is nullptr
        return bit_mask::is_valid(valid, id);
    }
};

/** -------------------------------------------------------------------------*
 * @brief input iterator to support null bitmask
 * Input iterator which can be used for cub and thrust.
 * This is derived from `thrust::iterator_adaptor`
 * `T_input = ColumnInput` will provide the value at index `id`
 *  with/without null bitmask.
 *
 * @tparam  T_output The output helper struct type, usually `mutator_single`,
 *                   `mutator_squared`, or `mutator_meanvar`.
 * @tparam  T_input  The input struct type of `ColumnInput`
 * @tparam  Iterator The base iterator which gives the index of array.
 *                   The default is `thrust::counting_iterator`
 * -------------------------------------------------------------------------**/
template<typename T_output, typename T_input, typename Iterator=thrust::counting_iterator<gdf_index_type> >
  class column_input_iterator
    : public thrust::iterator_adaptor<
        column_input_iterator<T_output, T_input, Iterator>, // the name of the iterator we're creating
        Iterator,                   // the name of the iterator we're adapting
        thrust::use_default, thrust::use_default, thrust::use_default,
        T_output,                   // set `super_t::reference` to `T_output`
        thrust::use_default
      >
  {
  public:
    // shorthand for the name of the iterator_adaptor we're deriving from
    using super_t = thrust::iterator_adaptor<
      column_input_iterator<T_output, T_input, Iterator>, Iterator,
      thrust::use_default, thrust::use_default, thrust::use_default, T_output, thrust::use_default
    >;

    CUDA_HOST_DEVICE_CALLABLE
    column_input_iterator(const T_input col, const Iterator &it) : super_t(it), colData(col){}

    CUDA_HOST_DEVICE_CALLABLE
    column_input_iterator(const T_input col) : super_t(Iterator{0}), colData(col){}

    CUDA_HOST_DEVICE_CALLABLE
    column_input_iterator(const column_input_iterator &other) : super_t(other.base()), colData(other.colData){}

    // befriend thrust::iterator_core_access to allow it access to the private interface below
    friend class thrust::iterator_core_access;

  private:
    const T_input colData;

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
 * @tparam T_output_helper
 *                   The output helper struct type, usually `mutator_single`,
 *                   `mutator_squared`, or `mutator_meanvar`.
 * @tparam Iterator_Index
 *                  The base iterator which gives the index of array.
 *                  The default is `thrust::counting_iterator`
 *
 * @param[in] data     The pointer of column data array
 * @param[in] valid    The pointer of null bitmask of column
 * @param[in] identity The identity value used when the mask value is false
 * @param[in] it       The index iterator, `thrust::counting_iterator` by default
 * -------------------------------------------------------------------------**/
template <bool has_nulls, typename T_element, typename T_output = T_element,
    typename T_output_helper = cudf::detail::mutator_single<T_output>,
    typename Iterator_Index=thrust::counting_iterator<gdf_index_type> >
auto make_iterator(const T_element *data, const bit_mask::bit_mask_t *valid,
    T_element identity, Iterator_Index const it = Iterator_Index(0))
{
    using T_input = cudf::detail::ColumnInput<T_output_helper, T_element, has_nulls>;
    using T_iterator = cudf::detail::column_input_iterator<T_output, T_input, Iterator_Index>;

    // T_input constructor checks if valid is not nullptr
    return T_iterator(T_input(data, valid, identity), it);
}

/** -------------------------------------------------------------------------*
 *  @overload auto make_iterator(const T_element *data,
 *                     const gdf_valid_type *valid, T_element identity,
 *                     Iterator_Index const it = Iterator_Index(0))
 * -------------------------------------------------------------------------**/
template <bool has_nulls, typename T_element, typename T_output = T_element,
    typename T_output_helper = cudf::detail::mutator_single<T_output>,
    typename Iterator_Index=thrust::counting_iterator<gdf_index_type> >
auto make_iterator(const T_element *data, const gdf_valid_type *valid,
    T_element identity, Iterator_Index const it = Iterator_Index(0))
{
    return make_iterator<has_nulls, T_element, T_output, T_output_helper, Iterator_Index>
        (data, reinterpret_cast<const bit_mask::bit_mask_t*>(valid), identity, it);
}


/** -------------------------------------------------------------------------*
 *  @overload auto make_iterator(const gdf_column& column, T_element identity,
 *                     Iterator_Index const it = Iterator_Index(0))
 * -------------------------------------------------------------------------**/
template <bool has_nulls, typename T_element, typename T_output = T_element,
    typename T_output_helper = cudf::detail::mutator_single<T_output>,
    typename Iterator_Index=thrust::counting_iterator<gdf_index_type> >
auto make_iterator(const gdf_column& column,
    T_element identity, const Iterator_Index it = Iterator_Index(0))
{
    // check the data type
    CUDF_EXPECTS(gdf_dtype_of<T_element>() == column.dtype, "the data type mismatch");

    return make_iterator<has_nulls, T_element, T_output, T_output_helper, Iterator_Index>
        (static_cast<const T_element*>(column.data),
        reinterpret_cast<const bit_mask::bit_mask_t*>(column.valid), identity, it);
}

} // namespace cudf

#endif