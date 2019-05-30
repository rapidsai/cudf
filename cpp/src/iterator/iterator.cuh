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

#ifndef CUDF_ITERATOR_CUH
#define CUDF_ITERATOR_CUH

#include <cudf.h>
#include <bitmask/bit_mask.cuh>         // need for bit_mask::bit_mask_t
#include <utilities/cudf_utils.h>       // need for CUDA_HOST_DEVICE_CALLABLE

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf
{

// --------------------------------------------------------------------------------------------------------
// structs of output for column_input_iterator

template<typename T>
struct ColumnOutputSingle
{
    T value;

    template<typename T_input>
    CUDA_HOST_DEVICE_CALLABLE
    ColumnOutputSingle(T_input _value, bool is_valid=false)
    : value( static_cast<T>(_value) )
    {};

    CUDA_HOST_DEVICE_CALLABLE
    ColumnOutputSingle(){};

    CUDA_HOST_DEVICE_CALLABLE
    operator T() const { return value; }
};

template<typename T>
struct ColumnOutputSquared
{
    T value_squared;

    template<typename T_input>
    CUDA_HOST_DEVICE_CALLABLE
    ColumnOutputSquared(T_input _value, bool is_valid=false)
    {
        T v = static_cast<T>(_value);
        value_squared = v*v;
    };

    CUDA_HOST_DEVICE_CALLABLE
    ColumnOutputSquared(){};

    CUDA_HOST_DEVICE_CALLABLE
    operator T() const { return value_squared; }
};

template<typename T, bool update_count=true>
struct ColumnOutputMixed;


template<typename T>
struct ColumnOutputMixed<T, true>
{
    T value;
    T value_squared;
    gdf_index_type count;

    template<typename T_input>
    CUDA_HOST_DEVICE_CALLABLE
    ColumnOutputMixed(T_input _value, bool is_valid)
    : value( static_cast<T>(_value) ), count(is_valid? 1 : 0)
    {
        value_squared = value*value;
    };

    CUDA_HOST_DEVICE_CALLABLE
    ColumnOutputMixed(T _value, T _value_squared=0, gdf_index_type _count=0)
    : value(_value), value_squared(_value_squared), count(_count)
    {};


    CUDA_HOST_DEVICE_CALLABLE
    ColumnOutputMixed()
    : value(0), value_squared(0), count(0)
    {};

    using this_t = ColumnOutputMixed<T, true>;

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

template<typename T>
struct ColumnOutputMixed<T, false>
{
    T value;
    T value_squared;
    gdf_index_type count;

    template<typename T_input>
    CUDA_HOST_DEVICE_CALLABLE
    ColumnOutputMixed(T_input _value, bool is_valid)
    : value( static_cast<T>(_value) )
    {
        value_squared = value*value;
    };

    CUDA_HOST_DEVICE_CALLABLE
    ColumnOutputMixed(T _value, T _value_squared=0, gdf_index_type _count=0)
    : value(_value), value_squared(_value_squared), count(_count)
    {};

    CUDA_HOST_DEVICE_CALLABLE
    ColumnOutputMixed()
    : value(0), value_squared(0), count(0)
    {};

    using this_t = ColumnOutputMixed<T, false>;

    CUDA_HOST_DEVICE_CALLABLE
    this_t operator+(this_t const &rhs) const
    {
        return this_t(
            (this->value + rhs.value),
            (this->value_squared + rhs.value_squared),
            (this->count)
        );
    }

    CUDA_HOST_DEVICE_CALLABLE
    bool operator==(this_t const &rhs) const
    {
        return (
            (this->value == rhs.value) &&
            (this->value_squared == rhs.value_squared)
        );
    }
};

// --------------------------------------------------------------------------------------------------------
// structs for column_input_iterator
template<typename T_output, typename T_element, bool nulls_present=true>
struct ColumnInput;

template<typename T_output, typename T_element>
struct ColumnInput<T_output, T_element, false>{
    const T_element *data;

    CUDA_HOST_DEVICE_CALLABLE
    ColumnInput(const T_element *_data)
    : data(_data){};

    CUDA_HOST_DEVICE_CALLABLE
    T_output at(gdf_index_type id) const {
        return T_output(data[id], true);
    };
};

template<typename T_output, typename T_element>
struct ColumnInput<T_output, T_element, true>{
    const T_element *data;
    const bit_mask::bit_mask_t *valid;
    T_element identity;

    CUDA_HOST_DEVICE_CALLABLE
    ColumnInput(const T_element *_data, const bit_mask::bit_mask_t *_valid, T_element _identity)
    : data(_data), valid(_valid), identity(_identity){};

    CUDA_HOST_DEVICE_CALLABLE
    ColumnInput(const T_element *_data, const gdf_valid_type*_valid, T_element _identity)
    : ColumnInput(_data, reinterpret_cast<const bit_mask::bit_mask_t*>(_valid), _identity) {};

    CUDA_HOST_DEVICE_CALLABLE
    T_output at(gdf_index_type id) const {
        return T_output(get_value(id), is_valid(id));
    };

    CUDA_HOST_DEVICE_CALLABLE
    T_element get_value(gdf_index_type id) const {
        return (is_valid(id))? data[id] : identity;
    }

    CUDA_HOST_DEVICE_CALLABLE
    bool is_valid(gdf_index_type id) const
    {
        return bit_mask::is_valid(valid, id);
    }
};


// ---------------------------------------------------------------------------
// column_input_iterator
template<typename T_output, typename T_input, typename Iterator=thrust::counting_iterator<gdf_index_type> >
  class column_input_iterator
    : public thrust::iterator_adaptor<
        column_input_iterator<T_output, T_input, Iterator>, // the first template parameter is the name of the iterator we're creating
        Iterator,                   // the second template parameter is the name of the iterator we're adapting
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
    column_input_iterator(const column_input_iterator &it) : super_t(it.base()), colData(it.colData){}

    // befriend thrust::iterator_core_access to allow it access to the private interface below
    friend class thrust::iterator_core_access;

  private:
    const T_input colData;

    // it is private because only thrust::iterator_core_access needs access to it
    CUDA_HOST_DEVICE_CALLABLE
    typename super_t::reference dereference() const
    {
      int id = *(this->base());
      return colData.at(id);
    }
};

// ---------------------------------------------------------------------------
// helper functions to make iterator

template <typename T_element, typename T_output = T_element,
    typename T_output_helper = ColumnOutputSingle<T_output>,
    typename Iterator_Index=thrust::counting_iterator<gdf_index_type> >
auto make_iterator_with_nulls(const T_element *_data, const bit_mask::bit_mask_t *_valid,
    T_element _identity, Iterator_Index const _it = Iterator_Index(0))
{
    using T_input = ColumnInput<T_output_helper, T_element>;
    using T_iterator = column_input_iterator<T_output, T_input, Iterator_Index>;

    return T_iterator(T_input(_data, _valid, _identity), _it);
}

template <typename T_element, typename T_output = T_element,
    typename T_output_helper = ColumnOutputSingle<T_output>,
    typename Iterator_Index=thrust::counting_iterator<gdf_index_type> >
auto make_iterator_with_nulls(const T_element *_data, const gdf_valid_type *_valid,
    T_element _identity, Iterator_Index const _it = Iterator_Index(0))
{
    return make_iterator_with_nulls<T_element, T_output, T_output_helper, Iterator_Index>
        (_data, reinterpret_cast<const bit_mask::bit_mask_t*>(_valid), _identity, _it);
}

template <typename T_element, typename T_output = T_element,
    typename T_output_helper = ColumnOutputSingle<T_output>,
    typename Iterator_Index=thrust::counting_iterator<gdf_index_type> >
auto make_iterator_without_nulls(const T_element *_data, const Iterator_Index _it = Iterator_Index(0))
{
    using T_input = ColumnInput<T_output_helper, T_element, false>;
    using T_iterator = column_input_iterator<T_output, T_input, Iterator_Index>;

    return T_iterator(T_input(_data), _it);
}

} // namespace cudf

#endif