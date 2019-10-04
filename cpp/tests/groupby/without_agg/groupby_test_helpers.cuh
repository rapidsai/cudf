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
#ifndef GROUPBY_TEST_HELPERS_CUH
#define GROUPBY_TEST_HELPERS_CUH

#include "test_parameters.cuh"
#include <string>
#include <limits>
#include <type_traits>
#include <bitmask/legacy/legacy_bitmask.hpp>
#include <tests/utilities/valid_vectors.h>

//Terminating call
//Extract the value of the Ith element of a tuple of vectors keys
//at index location and store it in the Ith element of the tuple key
template <std::size_t I = 0, typename... Keys>
inline typename std::enable_if<I == sizeof...(Keys), void>::type
extract(const std::tuple<std::vector<Keys>...>& keys, const size_t index, std::tuple<Keys...> &key) {
}

//Extract the value of the Ith element of a tuple of vectors keys
//at index location and store it in the Ith element of the tuple key
template <std::size_t I = 0, typename... Keys>
inline typename std::enable_if<I < sizeof...(Keys), void>::type
extract(const std::tuple<std::vector<Keys>...>& keys, const size_t index, std::tuple<Keys...> &key) {
    std::get<I>(key) = std::get<I>(keys)[index];
    extract<I + 1, Keys...>(keys, index, key);
}

//Extract a tuple of values from a tuple of vector for a given index
//keys Tuple of vectors of types Keys
//index Location of the value to be extracted in each vector
template <typename... Keys>
std::tuple<Keys...>
extractKey(std::tuple<std::vector<Keys>...>& keys, const size_t index) {
    std::tuple<Keys...> key;
    extract(keys, index, key);
    return key;
}


//Terminating call
//Extract the value of the Ith element of a tuple of vectors keys
//at index location and store it in the Ith element of the tuple key
template <std::size_t I = 0, typename... Keys>
inline typename std::enable_if<I == sizeof...(Keys), void>::type
extractWithNulls(const std::tuple<std::vector<Keys>...>& keys, std::basic_string<bool> &valids,  const size_t index, std::tuple<Keys...> &key) {

}

//Extract the value (replace nulls with MAX_VALUE of ValueType) of the Ith element of a tuple of vectors keys
//at index location and store it in the Ith element of the tuple key
template <std::size_t I = 0, typename... Keys>
inline typename std::enable_if<I < sizeof...(Keys), void>::type
extractWithNulls(const std::tuple<std::vector<Keys>...>& keys, std::basic_string<bool> &valids, const size_t index, std::tuple<Keys...> &key) {
    using key_type =  typename std::decay<decltype(std::get<I>(keys))>::type;
    std::get<I>(key) = valids[I] ? std::get<I>(keys)[index] : std::numeric_limits<typename key_type::value_type>::max();
    extractWithNulls<I + 1, Keys...>(keys, valids, index, key);
}


//Extract a tuple of values (replace nulls with MAX_VALUE of ValueType) from a tuple of vector for a given index
//keys Tuple of vectors of types Keys
//index Location of the value to be extracted in each vector
template <typename... Keys>
std::tuple<Keys...>
extractKeyWithNulls(std::tuple<std::vector<Keys>...>& keys, std::basic_string<bool> &valids, const size_t index) {
    std::tuple<Keys...> key;
    extractWithNulls(keys, valids, index, key);
    return key;
}


//Terminating call
//Extract the value of the Ith element of a tuple of vectors keys
//at index location and store it in the Ith element of the tuple key
template <std::size_t I = 0, typename... Keys>
inline typename std::enable_if<I == sizeof...(Keys), void>::type
print_tuple_value(std::stringstream &ss, const std::tuple<Keys...> &item) {

}

//Extract the value (replace nulls with MAX_VALUE of ValueType) of the Ith element of a tuple of vectors keys
//at index location and store it in the Ith element of the tuple key
template <std::size_t I = 0, typename... Keys>
inline typename std::enable_if<I < sizeof...(Keys), void>::type
print_tuple_value(std::stringstream &ss, const std::tuple<Keys...> &item) {
    using val_type = typename std::decay<decltype(std::get<I>(item))>::type;
    auto val = std::get<I>(item) == std::numeric_limits<val_type>::max()? "@" : std::to_string(std::get<I>(item));
    ss << val << "|" ;
    print_tuple_value<I + 1, Keys...>(ss, item);
}


//Struct to generate random values of type T
template <typename K>
struct RandomValues {
    //Depending upon the type T, select
    //real or integer distribution
    using Distribution = typename
    std::conditional<
    std::is_integral<K>::value,
    typename std::uniform_int_distribution<K>,
    typename std::uniform_real_distribution<K>>::type;

    //Minimum value of the distribution
    K min;

    //Maximum value of the distribution
    K max;

    //Random device
    mutable std::random_device rd;

    //Mersenne Twister generator
    mutable std::mt19937 gen;

    //Object of selected distribution type
    mutable Distribution dis;

    //Constructor to set minimum and maximum values of the distribution
    RandomValues(const K _min, const K _max) :
        min(_min), max(_max), gen(rd()), dis(min, max) {}

    //Constructor to set minimum and maximum values of the distribution
    RandomValues(const K _min, const K _max, size_t seed) :
        min(_min), max(_max), gen(seed), dis(min, max) {}

    //Operator to generate random value (const variant)
    K operator()(void) const { return dis(gen); }

    //Operator to generate random value
    K operator()(void) { return dis(gen); }
};

//Integral specialization of createUniqueKeys
//Creates keys in an std::vector using scan operation
//key_count number of a unique key is generated
//value_per_key number of times the key is repeated
//column_range maximum value of the keys
//v std::vector to which generated keys are pushed into
template <typename T>
void createUniqueKeys(
        const size_t key_count,
        const size_t value_per_key,
        const size_t column_range,
        typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type& v,
        const size_t shuffle_seed = 0) {
    T ratio = static_cast<T>(column_range)/static_cast<T>(key_count);
    RandomValues<T> r(1, ratio);
    RandomValues<size_t> key_l(1, value_per_key, shuffle_seed);
    T rand_key{0};
    for (size_t i = 0; i < key_count; ++i) {
        rand_key += r();
        for (size_t j = 0; j < key_l(); ++j) { v.push_back(rand_key); }
    }
}

//Non integral specialization of createUniqueKeys
//Creates keys in an std::vector using scan operation
//key_count number of a unique key is generated
//value_per_key number of times the key is repeated
//column_range maximum value of the keys
//v std::vector to which generated keys are pushed into
template <typename T>
void createUniqueKeys(
        const size_t key_count,
        const size_t value_per_key,
        const size_t column_range,
        typename std::enable_if<!std::is_integral<T>::value, std::vector<T>>::type& v,
        const size_t shuffle_seed = 0) {
    T ratio = static_cast<T>(column_range)/static_cast<T>(key_count);
    RandomValues<T> r(std::numeric_limits<T>::epsilon()*4, ratio);
    RandomValues<size_t> key_l(1, value_per_key, shuffle_seed);
    T rand_key{0};
    for (size_t i = 0; i < key_count; ++i) {
        rand_key += r();
        for (size_t j = 0; j < key_l(); ++j) { v.push_back(rand_key); }
    }
}

//Initialize a key vector with random data.
//k The vector of keys to be populated
//key_count The number of keys
//value_per_key The number of times a random aggregation value is generated for a key
//column_range The maximum value of the key columns
//shuffle_seed The seed provided to shuffle the generated vector randomly
//unique Ensures that the keys generated are only repeated value_per_key times.
template <typename K>
void initialize_key_vector(std::vector<K>& k,
        const size_t key_count, const size_t value_per_key,
        const size_t column_range, const size_t shuffle_seed, bool unique = false) {
    if (key_count*value_per_key == 0) { return; }
    k.reserve(key_count*value_per_key);
    if (unique) {
        assert((column_range >= key_count));
        createUniqueKeys<K>(key_count, value_per_key, column_range, k, shuffle_seed);
    } else {
        RandomValues<K> r(0, static_cast<K>(column_range));
        for (size_t i = 0; i < key_count; ++i) {
            K rand_key = r();
            for (size_t j = 0; j < value_per_key; ++j) { k.push_back(rand_key); }
        }
    }
    //Shuffle current vector
    std::mt19937 g(shuffle_seed);
    std::shuffle(k.begin(), k.end(), g);
}

//Initialize a value vector with random data
template <typename V>
void initialize_values(std::vector<V>& v,
        const size_t key_count, const size_t value_per_key,
        const size_t column_range, const size_t shuffle_seed) {
    if (key_count*value_per_key == 0) { return; }
    v.reserve(key_count*value_per_key);
    RandomValues<V> r(0, static_cast<V>(column_range));
    RandomValues<size_t> key_l(1, value_per_key, shuffle_seed);
    for (size_t i = 0; i < key_count; ++i) {
        auto val = r();
        for (size_t j = 0; j < key_l(); ++j) { v.push_back(val); }
    }
    //Shuffle current vector
    std::mt19937 g(shuffle_seed);
    std::shuffle(v.begin(), v.end(), g);
}

//compile time recursion to initialize a tuple of vectors
template<std::size_t I = 0, typename... K>
inline typename std::enable_if<I == sizeof...(K), void>::type
initialize_keys(std::tuple<std::vector<K>...>& k,
        const size_t key_count, const size_t value_per_key,
        const size_t column_range, const size_t shuffle_seed, bool unique = true)
{
 //bottom of compile-time recursion
 //purposely empty...
}

//compile time recursion to initialize a tuple of vectors
template<std::size_t I = 0, typename... K>
inline typename std::enable_if<I < sizeof...(K), void>::type
initialize_keys(std::tuple<std::vector<K>...>& k,
        const size_t key_count, const size_t value_per_key,
        const size_t column_range, const size_t shuffle_seed, bool unique = true)
{
  //Initialize the current vector
 initialize_key_vector(std::get<I>(k),
         key_count, value_per_key,
         column_range, shuffle_seed, unique);

 //recurse to next vector in tuple
 initialize_keys<I + 1, K...>(k,
         key_count, value_per_key,
         column_range, shuffle_seed, unique);
}

//Copy device side gdf_column data to an std::vector
template <typename T>
void copy_gdf_column(gdf_column* column, std::vector<T>& vec) {
    //TODO : Add map of sizes of gdf_dtype and assert against sizeof(T)
    vec.resize(column->size);
    cudaMemcpy(vec.data(), column->data, column->size * sizeof(T), cudaMemcpyDeviceToHost);
}

//Empty terminal call
template<std::size_t I = 0, typename... K>
inline typename std::enable_if<I == sizeof...(K), void>::type
copy_gdf_tuple(
    gdf_column **group_by_output_key,
    std::tuple<std::vector<K>...>& output_key) {}

//Non terminating call to copy the Ith element of group_by_output_key
//to the Ith element of output_key
template<std::size_t I = 0, typename... K>
inline typename std::enable_if<I < sizeof...(K), void>::type
copy_gdf_tuple(
    gdf_column **group_by_output_key,
    std::tuple<std::vector<K>...>& output_key) {
    copy_gdf_column(group_by_output_key[I], std::get<I>(output_key));
    copy_gdf_tuple<I + 1, K...>(group_by_output_key, output_key);
}

//Copy the contents of gdf_columns to std::vectors
//group_by_output_key is copied to a tuple of vectors output_key
//group_by_output_value is copied to a vector output_value
template <typename gdf_column, typename multi_column_t, typename output_t>
void copy_output(
    gdf_column **group_by_output_key,
    multi_column_t& output_key,
    gdf_column *group_by_output_value,
    std::vector<output_t>& output_value,
    gdf_column *gdf_out_indices,
    std::vector<output_t>& cpu_out_indices) {
    copy_gdf_tuple(group_by_output_key, output_key);
    copy_gdf_column(group_by_output_value, output_value);
    copy_gdf_column(gdf_out_indices, cpu_out_indices);
}

template <typename gdf_column, typename multi_column_t, typename output_t>
void copy_output(
    gdf_column **group_by_output_key,
    multi_column_t& output_key,
    gdf_column *group_by_output_value,
    std::vector<output_t>& output_value) {
    copy_gdf_tuple(group_by_output_key, output_key);
    copy_gdf_column(group_by_output_value, output_value);
}

template <typename gdf_column, typename multi_column_t>
void copy_output_with_array(
    gdf_column **group_by_output_key,
    multi_column_t& output_key,
    gdf_size_type *group_by_output_value_array,
    gdf_size_type group_by_output_value_size,
    std::vector<gdf_size_type>& output_value) {
    copy_gdf_tuple(group_by_output_key, output_key);
    output_value.resize(group_by_output_value_size);
    cudaMemcpy(output_value.data(), group_by_output_value_array, group_by_output_value_size * sizeof(gdf_size_type), cudaMemcpyDeviceToHost);
}

//Copy device side gdf_column data to an std::vector
template <typename T>
void copy_gdf_column_with_nulls(gdf_column* column, std::vector<T>& vec, host_valid_pointer& output_valids) {
    //TODO : Add map of sizes of gdf_dtype and assert against sizeof(T)
    vec.resize(column->size);
    cudaMemcpy(vec.data(), column->data, column->size * sizeof(T), cudaMemcpyDeviceToHost);

    cudaMemcpy(output_valids.get(),  column->valid, gdf_num_bitmask_elements(column->size), cudaMemcpyDeviceToHost);
}

//Empty terminal call
template<std::size_t I = 0, typename... K>
inline typename std::enable_if<I == sizeof...(K), void>::type
copy_gdf_tuple_with_nulls(
    gdf_column **group_by_output_key,
    std::tuple<std::vector<K>...>& output_key, std::vector<host_valid_pointer>& output_valids) {}

//Non terminating call to copy the Ith element of group_by_output_key
//to the Ith element of output_key
template<std::size_t I = 0, typename... K>
inline typename std::enable_if<I < sizeof...(K), void>::type
copy_gdf_tuple_with_nulls(
    gdf_column **group_by_output_key,
    std::tuple<std::vector<K>...>& output_key, std::vector<host_valid_pointer>& output_valids) {
    copy_gdf_column_with_nulls(group_by_output_key[I], std::get<I>(output_key), output_valids[I]);
    copy_gdf_tuple_with_nulls<I + 1, K...>(group_by_output_key, output_key, output_valids);
}

//Copy the contents of gdf_columns to std::vectors
//group_by_output_key is copied to a tuple of vectors output_key
//group_by_output_value is copied to a vector output_value
template <typename gdf_column, typename multi_column_t, typename output_t>
void copy_output_with_nulls(
    gdf_column **group_by_output_key,
    multi_column_t& output_key,
    std::vector<host_valid_pointer>& output_valids,
    gdf_column *group_by_output_value,
    std::vector<output_t>& output_value) {
    copy_gdf_tuple_with_nulls(group_by_output_key, output_key, output_valids);
    copy_gdf_column(group_by_output_value, output_value);
}

template <typename gdf_column, typename multi_column_t>
void copy_output_with_array_with_nulls(
    gdf_column **group_by_output_key,
    multi_column_t& output_key,
    std::vector<host_valid_pointer>& output_valids,
    gdf_size_type *group_by_output_value_array,
    gdf_size_type group_by_output_value_size,
    std::vector<gdf_size_type>& output_value) {
    copy_gdf_tuple_with_nulls(group_by_output_key, output_key, output_valids);
    output_value.resize(group_by_output_value_size);
    cudaMemcpy(output_value.data(), group_by_output_value_array, group_by_output_value_size * sizeof(gdf_size_type), cudaMemcpyDeviceToHost);
}

//
// custom functions
//

//print a tuple recursively. Terminating empty call.
template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
print_basic_tuple(std::tuple<Tp...> t) { std::cout<<"\n"; }

//print a tuple recursively. Recursive call.
template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type
print_basic_tuple(std::tuple<Tp...> t) {
    std::cout<<std::get<I>(t)<<"\t";
    print_basic_tuple<I+1, Tp...>(t);
}

#endif