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

/* Proof the concept for iterator driven aggregations to reuse the logic

   The concepts:
   1. computes the aggregation by given iterators
   2. computes by using cub and thrust with same function parameters
   3. accepts nulls and group_by with same function parameters

    CUB:  https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html#aa4adabeb841b852a7a5ecf4f99a2daeb
    Thrust: https://thrust.github.io/doc/group__reductions.html#ga43eea9a000f912716189687306884fc7
*/

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <bitset>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <iterator>

#include <utilities/cudf_utils.h> // need for CUDA_HOST_DEVICE_CALLABLE
#include <utilities/device_atomics.cuh> // need for device operators.

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

#include <cub/device/device_reduce.cuh>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>


void gen_nullbitmap(std::vector<gdf_valid_type>& v, std::vector<bool>& host_bools)
{
    int length = host_bools.size();
    auto n_bytes = gdf_valid_allocation_size(length);

    v.resize(n_bytes);
    // TODO: generic
    for(int i=0; i<length; i++)
    {
        int pos = i/8;
        int bit_index = i%8;
        if( bit_index == 0)v[pos] = 0;
        if( host_bools[i] )v[pos] += (1 << bit_index);
    }
}


template <typename T>
class IteratorWithNulls : public std::iterator<std::random_access_iterator_tag, T>
{
public:
//    using difference_type = std::iterator<std::random_access_iterator_tag, T>::difference_type;
    using difference_type = gdf_size_type;

    // Discussion: std::iterator is deprecated in C++17
    CUDA_HOST_DEVICE_CALLABLE
    IteratorWithNulls(const T* _data, const gdf_valid_type* _valid, T _identity)
    : data(_data), valid(_valid), identity(_identity), index(0)
    {};

    CUDA_HOST_DEVICE_CALLABLE
    IteratorWithNulls(const IteratorWithNulls& ref)
    : data(ref.data), valid(ref.valid), identity(ref.identity), index(ref.index)
    {};

    CUDA_HOST_DEVICE_CALLABLE
    bool operator==(const IteratorWithNulls& others)
    {//printf("cmp (%d, %d)\n", index, others.index);
        return index == others.index; };

    CUDA_HOST_DEVICE_CALLABLE
    bool operator!=(const IteratorWithNulls& others)
    { return !(*this == others); };

    CUDA_HOST_DEVICE_CALLABLE
    difference_type operator-(const IteratorWithNulls& others)
    {
        return (index - others.index );
    };

    CUDA_HOST_DEVICE_CALLABLE
    T operator*() const
    {
//        return (gdf_is_valid(valid, index))? data[index] : identity;
        T val = (gdf_is_valid(valid, index))? data[index] : identity;
//        printf("val(%d, %d) = %d\n", data[index], gdf_is_valid(valid, index), val);
        return val;
    };

    CUDA_HOST_DEVICE_CALLABLE
    T operator[](const difference_type length) const
    {
        gdf_size_type id = index + length;
        return (gdf_is_valid(valid, id))? data[id] : identity;
    };

    CUDA_HOST_DEVICE_CALLABLE
    IteratorWithNulls& operator+=(const difference_type length)
    { //printf("operator+ (%d, %d)\n", index, length);
        index += length;
        return *this; };

    CUDA_HOST_DEVICE_CALLABLE
    IteratorWithNulls& operator-=(const difference_type length)
    {return (*this -= length ); };

    CUDA_HOST_DEVICE_CALLABLE
    IteratorWithNulls& operator++() { return (*this += 1);};

    CUDA_HOST_DEVICE_CALLABLE
    IteratorWithNulls operator++(int) {IteratorWithNulls retval = *this; ++(*this); return retval;}

    CUDA_HOST_DEVICE_CALLABLE
    IteratorWithNulls operator+(const difference_type length) {
        IteratorWithNulls tmp(*this);
        return (tmp += length);
    };

protected:
    const T *data;
    const gdf_valid_type *valid;
    const T identity;

    gdf_size_type index; // variables
};



template <typename T>
class IteratorGroupBy : public IteratorWithNulls<T>
{
public:
    using difference_type = gdf_size_type;

    // Discussion: std::iterator is deprecated in C++17
    CUDA_HOST_DEVICE_CALLABLE
    IteratorGroupBy(const T* _data, const int32_t* _indices)
    : IteratorWithNulls<T>(_data, nullptr, T{0}), indices(_indices)
    {};

    CUDA_HOST_DEVICE_CALLABLE
    IteratorGroupBy(const IteratorGroupBy& ref)
    : IteratorWithNulls<T>(ref), indices(ref.indices)
    {};

    CUDA_HOST_DEVICE_CALLABLE
    T operator*() const
    {
        T val = this->data[indices[this->index]];
        return val;
    };

    CUDA_HOST_DEVICE_CALLABLE
    IteratorGroupBy& operator+=(const difference_type length)
    { //printf("operator+ (%d, %d)\n", index, length);
        this->index += length;
        return *this; };

    CUDA_HOST_DEVICE_CALLABLE
    IteratorGroupBy& operator++() { return (*this += 1);};

    CUDA_HOST_DEVICE_CALLABLE
    IteratorGroupBy operator++(int) {IteratorGroupBy retval = *this; ++(*this); return retval;}

    CUDA_HOST_DEVICE_CALLABLE
    IteratorGroupBy operator+(const difference_type length) {
        IteratorGroupBy tmp(*this);
        return (tmp += length);
    };


protected:
    const int32_t *indices;

};


template <typename T>
struct IteratorTest : public GdfTest
{
    // iterator test case which uses cub
    template <typename InputIterator, typename T_output>
    void iterator_test_cub(T_output expected, InputIterator d_in, int num_items)
    {
        T init = T{0};
        thrust::device_vector<T> dev_result(1);

        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;

        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, dev_result.begin(), num_items,
            cudf::DeviceSum{}, init);
        // Allocate temporary storage
        RMM_TRY(RMM_ALLOC(&d_temp_storage, temp_storage_bytes, 0));

        // Run reduction
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, dev_result.begin(), num_items,
            cudf::DeviceSum{}, init);

        evaluate(expected, dev_result, "cub test");
    }

    // iterator test case which uses thrust
    template <typename InputIterator, typename T_output>
    void iterator_test_thrust(T_output expected, InputIterator d_in, int num_items, bool is_device=true)
    {
        T init = T{0};

        InputIterator d_in_last =  d_in + num_items;

        T result;
        if( is_device){
            result = thrust::reduce(thrust::device, d_in, d_in_last, init, cudf::DeviceSum{});
        }else{
            std::cout << "thrust test (host)" << std::endl;
            result = thrust::reduce(thrust::host, d_in, d_in_last, init, cudf::DeviceSum{});
            std::cout << "thrust test (host)" << std::endl;
        }

        EXPECT_EQ(expected, result) << "thrust test";
    }

    void evaluate(T expected, thrust::device_vector<T> &dev_result, const char* msg=nullptr)
    {
        thrust::host_vector<T>  hos_result(dev_result);

        EXPECT_EQ(expected, dev_result[0]) << msg ;
        std::cout << "expected <" << msg << "> = " << expected << std::endl;
    }
};

using TestingTypes = ::testing::Types<
    int32_t
>;

TYPED_TEST_CASE(IteratorTest, TestingTypes);


// tests for non-null iterator (pointer of device array)
TYPED_TEST(IteratorTest, non_null_iterator)
{
    using T = int32_t;
    std::vector<T> hos_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
    thrust::device_vector<T> dev_array(hos_array);

    T expected_value = std::accumulate(hos_array.begin(), hos_array.end(), T{0});

    auto it_dev = dev_array.begin();
    this->iterator_test_cub(expected_value, it_dev, dev_array.size());
    this->iterator_test_thrust(expected_value, it_dev, dev_array.size());
}


#if 0

template <typename T>
struct InputHelper
{
    struct ColumnData{
        const T *data;
        const gdf_valid_type *valid;
        T identity;

        ColumnData(T *_data, gdf_valid_type *_valid, T _identity)
        : data(_data), valid(_valid), identity(_identity){};
    };


    using IteratorTuple = thrust::tuple<
        thrust::constant_iterator<ColumnData>,
        thrust::counting_iterator<int>
    > ;
    using IteratorZipped = thrust::zip_iterator<IteratorTuple>;

    struct null_func  : public thrust::unary_function<IteratorZipped, T>
    {
        __host__ __device__
        T operator()(IteratorZipped zipped) const
        {
            ColumnData col = * thrust::get<0>(*zipped);
            int id = * thrust::get<1>(*zipped));

            return (gdf_is_valid(col.valid, id))? col.data[id] : col.identity;
        };
    };

    using IteratorTransformed = thrust::transform_iterator<InputHelper<T>::null_func, IteratorZipped>;

    IteratorTransformed make_transform_iterator(const T *_data, const gdf_valid_type *_valid, T _identity)
    {
        ColumnData col{_data, _valid, _identity};

        IteratorTransformed it = thrust::make_transform_iterator(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    thrust::make_constant_iterator(col),
                    thrust::make_counting_iterator(0)
                )
            ),
            InputHelper<T>::null_func()
        );

        return it;
    };


};

#endif

template<typename T>
struct ColumnData{
    const T *data;
    const gdf_valid_type *valid;
    T identity;

    ColumnData(T *_data, gdf_valid_type *_valid, T _identity)
    : data(_data), valid(_valid), identity(_identity){};
};


// derive repeat_iterator from iterator_adaptor
template<typename T, typename Iterator>
  class column_input_iterator
    : public thrust::iterator_adaptor<
        column_input_iterator<T, Iterator>, // the first template parameter is the name of the iterator we're creating
        Iterator,                   // the second template parameter is the name of the iterator we're adapting
        thrust::use_default, thrust::use_default, thrust::use_default, T, thrust::use_default
      >
      {
  public:
    // shorthand for the name of the iterator_adaptor we're deriving from
    using super_t = thrust::iterator_adaptor<
      column_input_iterator<T, Iterator>,
      Iterator,
      thrust::use_default, thrust::use_default, thrust::use_default, T, thrust::use_default
    >;

    __host__ __device__
    column_input_iterator(const Iterator &it, const ColumnData<T> col) : super_t(it), colData(col){}
    // befriend thrust::iterator_core_access to allow it access to the private interface below
    friend class thrust::iterator_core_access;
  private:
    const ColumnData<T> colData;

    // it is private because only thrust::iterator_core_access needs access to it
    __host__ __device__
    typename super_t::reference dereference() const
    {
      int id = *(this->base());

      return (gdf_is_valid(colData.valid, id))? colData.data[id] : colData.identity;
    }
};



/* tests for null iterator (column with null bitmap)
   actually, we can use cub for reduction with nulls.
    we may accelarate the reduction for a column using cub
*/
TYPED_TEST(IteratorTest, null_iterator)
{
    using T = int32_t;
    T init = T{0};

    std::vector<T> hos_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
    thrust::device_vector<T> dev_array(hos_array);

    std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1, 1});
    std::vector<gdf_valid_type> host_nulls;
    gen_nullbitmap(host_nulls, host_bools);
    thrust::device_vector<gdf_valid_type> dev_nulls(host_nulls);

    EXPECT_EQ(hos_array.size(), host_bools.size());

    std::vector<T> replaced_array(hos_array.size());
    std::transform(hos_array.begin(), hos_array.end(), host_bools.begin(),
        replaced_array.begin(), [&](T x, bool b) { return (b)? x : init; } );
    T expected_value = std::accumulate(replaced_array.begin(), replaced_array.end(), init);
    std::cout << "expected <null_iterator> = " << expected_value << std::endl;

    if(1)
    {  // check host side `IteratorWithNulls`.
        IteratorWithNulls<T> it_hos(hos_array.data(), host_nulls.data(), init);
        T expected_value_host = std::accumulate(it_hos, it_hos + hos_array.size(), T{0});
        EXPECT_EQ(expected_value, expected_value_host) << "CPU iterator test";

//        this->iterator_test_thrust(expected_value, it_hos2, dev_array.size(), false);

#if 0
        auto it_hos2 = InputHelper<T>::make_transform_iterator(
            hos_array.data(),
            host_nulls.data(), init);


        T expected_value_host2 = std::accumulate(it_hos2, it_hos2 + hos_array.size(), T{0});
        EXPECT_EQ(expected_value, expected_value_host2) << "CPU iterator test";
#else
        ColumnData<T> col(hos_array.data(), host_nulls.data(), init);
        thrust::counting_iterator<int32_t> counting_it = thrust::make_counting_iterator(int32_t{0});

        column_input_iterator<T, thrust::counting_iterator<int32_t>> it_hos2(counting_it, col);
        this->iterator_test_thrust(expected_value, it_hos2, dev_array.size(), false);

#endif

//        this->iterator_test_thrust(expected_value, it_hos2, dev_array.size(), false);


//        ZipIterator iter(thrust::make_tuple(int_v.begin(), float_v.begin(), char_v.begin()));



    }

    // create device side `IteratorWithNulls`.
    IteratorWithNulls<T> it_dev(
        static_cast<const T*>( dev_array.data().get() ),
        static_cast<const gdf_valid_type*>( dev_nulls.data().get() ),
        init);

    // reduction using cub
    this->iterator_test_cub(expected_value, it_dev, dev_array.size());

    ColumnData<T> col(hos_array.data(), host_nulls.data(), init);
    thrust::counting_iterator<int32_t> counting_it = thrust::make_counting_iterator(int32_t{0});

    column_input_iterator<T, thrust::counting_iterator<int32_t>> it_dev2(counting_it, col);

    // reduction using thrust
    this->iterator_test_thrust(expected_value, it_dev2, dev_array.size());
    // reduction using cub
    this->iterator_test_cub(expected_value, it_dev2, dev_array.size());


    std::cout << "test done." << std::endl;
    // reduction using thrust
//    this->iterator_test_thrust(expected_value, it_dev, dev_array.size());
}

/*
    tests for group_by iterator
    this was used by old implementation of group_by.

    This won't be used with the newer implementation
     (a.k.a. Single pass, distributive groupby https://github.com/rapidsai/cudf/pull/1478)
    distributive groupby uses atomic operation to accumulate.

*/

TYPED_TEST(IteratorTest, group_by_iterator)
{
#if 0
    using T = int32_t;
    using T_index = int32_t;

    std::vector<T> hos_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
    thrust::device_vector<T> dev_array(hos_array);

    std::vector<T_index> hos_indices({0, 1, 3, 5}); // sorted indices belongs to a group
    thrust::device_vector<T_index> dev_indices(hos_indices);

    T expected_value = std::accumulate(hos_indices.begin(), hos_indices.end(), T{0},
        [&](T acc, T_index id){ return (acc + hos_array[id]); } );
    std::cout << "expected <group_by_iterator> = " << expected_value << std::endl;

    if(0)
    {  // check host side `IteratorWithNulls`.
        IteratorGroupBy<T> it_hos(hos_array.data(), hos_indices.data());
        T expected_value_host = std::accumulate(it_hos, it_hos + hos_indices.size(), T{0});
        EXPECT_EQ(expected_value, expected_value_host) << "CPU iterator test";

        this->iterator_test_thrust(expected_value, it_hos, dev_array.size(), false);
    }
#endif

}

// tests for group_by iterator
TYPED_TEST(IteratorTest, group_by_iterator_null)
{
    // Discussion: how to do if all of values are nulls ?
    // maybe need to exclude null values first ? (it also gives `count` of a column value in the group)


    // TBD. it should be possible.
}