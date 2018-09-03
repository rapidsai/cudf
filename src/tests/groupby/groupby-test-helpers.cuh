#pragma once
#include "test_parameters.cuh"
#include <string>

template <std::size_t I = 0, typename... Keys>
inline typename std::enable_if<I == sizeof...(Keys), void>::type
extract(const std::tuple<std::vector<Keys>...>& keys, const size_t index, std::tuple<Keys...> &key) {
}

template <std::size_t I = 0, typename... Keys>
inline typename std::enable_if<I < sizeof...(Keys), void>::type
extract(const std::tuple<std::vector<Keys>...>& keys, const size_t index, std::tuple<Keys...> &key) {
    std::get<I>(key) = std::get<I>(keys)[index];
    extract<I + 1, Keys...>(keys, index, key);
}

template <typename... Keys>
std::tuple<Keys...>
extractKey(std::tuple<std::vector<Keys>...>& keys, const size_t index) {
    std::tuple<Keys...> key;
    extract(keys, index, key);
    return key;
}

template <typename K>
struct RandomValues {
    using Distribution = typename
    std::conditional<
    std::is_integral<K>::value,
    typename std::uniform_int_distribution<K>,
    typename std::uniform_real_distribution<K>>::type;

    K min;
    K max;
    mutable std::random_device rd;
    mutable std::mt19937 gen;
    mutable Distribution dis;
    RandomValues(const K _min, const K _max) :
        min(_min), max(_max), gen(rd()), dis(min, max) {}
    K operator()(void) const { return dis(gen); }
    K operator()(void) { return dis(gen); }
};

template <typename T>
void createUniqueKeys(
        const size_t key_count,
        const size_t value_per_key,
        const size_t column_range,
        typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type& v) {
    T ratio = static_cast<T>(column_range)/static_cast<T>(key_count);
    RandomValues<T> r(1, ratio);
    T rand_key{0};
    for (size_t i = 0; i < key_count; ++i) {
        rand_key += r();
        for (size_t j = 0; j < value_per_key; ++j) { v.push_back(rand_key); }
    }
}

template <typename T>
void createUniqueKeys(
        const size_t key_count,
        const size_t value_per_key,
        const size_t column_range,
        typename std::enable_if<!std::is_integral<T>::value, std::vector<T>>::type& v) {
    T ratio = static_cast<T>(column_range)/static_cast<T>(key_count);
    RandomValues<T> r(std::numeric_limits<T>::epsilon()*4, ratio);
    T rand_key{0};
    for (size_t i = 0; i < key_count; ++i) {
        rand_key += r();
        for (size_t j = 0; j < value_per_key; ++j) { v.push_back(rand_key); }
    }
}

// Initialize a key vector with random data
template <typename K>
void initialize_key_vector(std::vector<K>& k,
        const size_t key_count, const size_t value_per_key,
        const size_t column_range, const size_t shuffle_seed, bool unique = false) {
    k.reserve(key_count*value_per_key);
    if (unique) {
        assert((column_range > key_count));
        createUniqueKeys<K>(key_count, value_per_key, column_range, k);
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

// Initialize a value vector with random data
template <typename V>
void initialize_values(std::vector<V>& v,
        const size_t key_count, const size_t value_per_key,
        const size_t column_range, const size_t shuffle_seed) {
    v.reserve(key_count*value_per_key);
    RandomValues<V> r(0, static_cast<V>(column_range));
    for (size_t i = 0; i < key_count * value_per_key; ++i) {
        v.push_back(r());
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

template<std::size_t I = 0, typename... K>
inline typename std::enable_if<I < sizeof...(K), void>::type
initialize_keys(std::tuple<std::vector<K>...>& k,
        const size_t key_count, const size_t value_per_key,
        const size_t column_range, const size_t shuffle_seed, bool unique = true)
{
  // Initialize the current vector
 initialize_key_vector(std::get<I>(k),
         key_count, value_per_key,
         column_range, shuffle_seed, unique);

 //recurse to next vector in tuple
 initialize_keys<I + 1, K...>(k,
         key_count, value_per_key,
         column_range, shuffle_seed, unique);
}

template <typename T>
void copy_gdf_column(gdf_column* column, std::vector<T>& vec) {
    //TODO : Add map of sizes of gdf_dtype and assert against sizeof(T)
    vec.resize(column->size);
    cudaMemcpy(vec.data(), column->data, column->size * sizeof(T), cudaMemcpyDeviceToHost);
}

template<std::size_t I = 0, typename... K>
inline typename std::enable_if<I == sizeof...(K), void>::type
copy_gdf_tuple(
    gdf_column **group_by_output_key,
    std::tuple<std::vector<K>...>& output_key) {}

template<std::size_t I = 0, typename... K>
inline typename std::enable_if<I < sizeof...(K), void>::type
copy_gdf_tuple(
    gdf_column **group_by_output_key,
    std::tuple<std::vector<K>...>& output_key) {
    copy_gdf_column(group_by_output_key[I], std::get<I>(output_key));
    copy_gdf_tuple<I + 1, K...>(group_by_output_key, output_key);
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

// Prints a vector
template<typename T>
void print_vector(std::vector<T>& v)
{
 std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, ", "));
}

template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
print_tuple_vector(std::tuple<std::vector<Tp>...>& t)
{
 //bottom of compile-time recursion
 //purposely empty...
}

//compile time recursion to print a tuple of vectors
template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type
print_tuple_vector(std::tuple<std::vector<Tp>...>& t)
{
 // print the current vector:
 print_vector(std::get<I>(t));
 std::cout << std::endl;

 //recurse to next vector in tuple
 print_tuple_vector<I + 1, Tp...>(t);
}

template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
print_tuple(std::tuple<Tp...>& t) {
}

template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type
print_tuple(std::tuple<Tp...>& t) {
    if (I == 0) {std::cout<<"\n";}
    std::cout<<std::get<I>(t)<<"\t";
    print_tuple<I+1, Tp...>(t);
}
