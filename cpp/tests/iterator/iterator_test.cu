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

/* Proof the concept of iterator driven aggregations to reuse the logic

   The concepts:
   1. computes the aggregation by given iterators
   2. computes by using cub and thrust with same function parameters
   3. accepts nulls and group_by with same function parameters

    CUB reduction:  https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html#aa4adabeb841b852a7a5ecf4f99a2daeb
    Thrust reduction: https://thrust.github.io/doc/group__reductions.html#ga43eea9a000f912716189687306884fc7

    Thrust iterators: https://thrust.github.io/doc/group__fancyiterator.html
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


// --------------------------------------------------------------------------------------------------------
// structs for column_input_iterator
template<typename T>
struct ColumnDataNonNull{
    const T *data;

    __device__ __host__
    ColumnDataNonNull(T *_data)
    : data(_data){};

    __device__ __host__
    T at(gdf_index_type id) const {
        return data[id];
    };
};

template<typename T>
struct ColumnData{
    const T *data;
    const gdf_valid_type *valid;
    T identity;

    __device__ __host__
    ColumnData(T *_data, gdf_valid_type *_valid, T _identity)
    : data(_data), valid(_valid), identity(_identity){};

    __device__ __host__
    T at(gdf_index_type id) const {
        return (gdf_is_valid(valid, id))? data[id] : identity;
    };
};

template<typename T>
struct ColumnDataSquare : public ColumnData<T>
{
    ColumnDataSquare(T *_data, gdf_valid_type *_valid, T _identity)
    : ColumnData<T>(_data, _valid, _identity){};

    __device__ __host__
    T at(gdf_index_type id) const {
        T val = ColumnData<T>::at(id);
        return (val * val);
    };
};

// TBD. ColumnDataSquareNonNull


// column_input_iterator
template<typename T_output, typename T_input, typename Iterator=thrust::counting_iterator<gdf_index_type> >
  class column_input_iterator
    : public thrust::iterator_adaptor<
        column_input_iterator<T_output, T_input, Iterator>, // the first template parameter is the name of the iterator we're creating
        Iterator,                   // the second template parameter is the name of the iterator we're adapting
        thrust::use_default, thrust::use_default, thrust::use_default, T_output, thrust::use_default
      >
      {
  public:
    // shorthand for the name of the iterator_adaptor we're deriving from
    using super_t = thrust::iterator_adaptor<
      column_input_iterator<T_output, T_input, Iterator>,
      Iterator,
      thrust::use_default, thrust::use_default, thrust::use_default, T_output, thrust::use_default
    >;

    __host__ __device__
    column_input_iterator(const T_input col, const Iterator &it) : super_t(it), colData(col){}

    __host__ __device__
    column_input_iterator(const T_input col) : super_t(Iterator{0}), colData(col){}

    __host__ __device__
    column_input_iterator(const column_input_iterator &it) : super_t(it.base()), colData(it.colData){}

    // befriend thrust::iterator_core_access to allow it access to the private interface below
    friend class thrust::iterator_core_access;

    __host__ __device__
    column_input_iterator &operator=(const column_input_iterator &other)
    {
        super_t::operator=(other);

        colData = other.colData;
        return *this;
    }

  private:
    const T_input colData;

    // it is private because only thrust::iterator_core_access needs access to it
    __host__ __device__
    typename super_t::reference dereference() const
    {
      int id = *(this->base());
      return colData.at(id);
    }
};

// --------------------------------------------------------------------------------------------------------


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

        EXPECT_EQ( thrust::distance(d_in, d_in_last), num_items);


        T result;
        if( is_device){
            result = thrust::reduce(thrust::device, d_in, d_in_last, init, cudf::DeviceSum{});
        }else{
            for(auto it = d_in; it != d_in_last; ++it){
                std::cout << "V: " << *it << std::endl;
            }

            std::cout << "thrust test (host start)" << std::endl;
            result = thrust::reduce(thrust::host, d_in, d_in_last, init, cudf::DeviceSum{});
            std::cout << "thrust test (host end)" << std::endl;
        }

        EXPECT_EQ(expected, result) << "thrust test";
    }

    void evaluate(T expected, thrust::device_vector<T> &dev_result, const char* msg=nullptr)
    {
        thrust::host_vector<T>  hos_result(dev_result);

        EXPECT_EQ(expected, dev_result[0]) << msg ;
        std::cout << "Done: expected <" << msg << "> = " << dev_result[0] << std::endl;
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

    // calculate the expected value by CPU.
    T expected_value = std::accumulate(hos_array.begin(), hos_array.end(), T{0});

    // driven by iterator as a pointer of device array.
    auto it_dev = dev_array.begin();
    this->iterator_test_cub(expected_value, it_dev, dev_array.size());
    this->iterator_test_thrust(expected_value, it_dev, dev_array.size());
}


/* tests for null input iterator (column with null bitmap)
   Actually, we can use cub for reduction with nulls without creating custom kernel or multiple steps.
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

    // calculate the expected value by CPU.
    std::vector<T> replaced_array(hos_array.size());
    std::transform(hos_array.begin(), hos_array.end(), host_bools.begin(),
        replaced_array.begin(), [&](T x, bool b) { return (b)? x : init; } );
    T expected_value = std::accumulate(replaced_array.begin(), replaced_array.end(), init);
    std::cout << "expected <null_iterator> = " << expected_value << std::endl;

    if(1)
    {  // check host side `IteratorWithNulls`.
        ColumnData<T> col(hos_array.data(), host_nulls.data(), init);

        column_input_iterator<T, ColumnData<T>> it_hos(col);
//        this->iterator_test_thrust(expected_value, it_hos, dev_array.size(), false);
    }

    ColumnData<T> col(dev_array.data().get(), dev_nulls.data().get(), init);
    column_input_iterator<T, ColumnData<T>> it_dev(col);

    // reduction using thrust
    this->iterator_test_thrust(expected_value, it_dev, dev_array.size());
    // reduction using cub
    this->iterator_test_cub(expected_value, it_dev, dev_array.size());

    std::cout << "test done." << std::endl;
}

/* tests for square input iterator
*/
TYPED_TEST(IteratorTest, null_iterator_square)
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

    // calculate the expected value by CPU.
    std::vector<T> replaced_array(hos_array.size());
    std::transform(hos_array.begin(), hos_array.end(), host_bools.begin(),
        replaced_array.begin(), [&](T x, bool b) { return (b)? x*x : init; } );
    T expected_value = std::accumulate(replaced_array.begin(), replaced_array.end(), init);
    std::cout << "expected <null_iterator> = " << expected_value << std::endl;

    if(1)
    {
        ColumnDataSquare<T> col(hos_array.data(), host_nulls.data(), init);
        column_input_iterator<T, ColumnDataSquare<T>> it_hos(col);
//        this->iterator_test_thrust(expected_value, it_hos, dev_array.size(), false);
    }

    ColumnDataSquare<T> col(dev_array.data().get(), dev_nulls.data().get(), init);
    column_input_iterator<T, ColumnDataSquare<T>> it_dev(col);

    // reduction using thrust
    this->iterator_test_thrust(expected_value, it_dev, dev_array.size());
    // reduction using cub
    this->iterator_test_cub(expected_value, it_dev, dev_array.size());

    std::cout << "test done." << std::endl;
}


/*
    tests for group_by iterator
    this was used by old implementation of group_by.

    This won't be used with the newer implementation
     (a.k.a. Single pass, distributive groupby https://github.com/rapidsai/cudf/pull/1478)
    distributive groupby uses atomic operation to accumulate.

    For group_by.cumsum() (scan base group_by) may not be single pass scan.
    There is a possiblity that this process may be used for group_by.cumsum().
*/
TYPED_TEST(IteratorTest, group_by_iterator)
{
    using T = int32_t;
    using T_index = gdf_index_type;

    std::vector<T> hos_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
    thrust::device_vector<T> dev_array(hos_array);

    std::vector<T_index> hos_indices({0, 1, 3, 5}); // sorted indices belongs to a group
    thrust::device_vector<T_index> dev_indices(hos_indices);

    // calculate the expected value by CPU.
    T expected_value = std::accumulate(hos_indices.begin(), hos_indices.end(), T{0},
        [&](T acc, T_index id){ return (acc + hos_array[id]); } );
    std::cout << "expected <group_by_iterator> = " << expected_value << std::endl;

    // pass `dev_indices` as base iterator of `column_input_iterator`.
    ColumnDataNonNull<T> col(dev_array.data().get());
    column_input_iterator<T, ColumnDataNonNull<T>, T_index*> it_dev(col, dev_indices.data().get());

    // reduction using thrust
    this->iterator_test_thrust(expected_value, it_dev, dev_indices.size());
    // reduction using cub
    this->iterator_test_cub(expected_value, it_dev, dev_indices.size());
}

// tests for group_by iterator
TYPED_TEST(IteratorTest, group_by_iterator_null)
{
    // Discussion: how to do if all of values are nulls ?
    // maybe need to exclude null values first ? (it also gives `count` of a column value in the group)


    // TBD. it should be possible.
}