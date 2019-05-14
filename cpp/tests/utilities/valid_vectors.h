/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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

#ifndef VALID_VECTORS_H
#define VALID_VECTORS_H

#include <functional>
#include <memory>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <string>

#include <utilities/cudf_utils.h>
#include <utilities/bit_util.cuh>
#include <bitmask/legacy_bitmask.hpp>

// host_valid_pointer is a wrapper for gdf_valid_type* with custom deleter
using host_valid_pointer = typename std::unique_ptr<gdf_valid_type, std::function<void(gdf_valid_type*)>>;


// Create a valid pointer and init it with null_count invalids
host_valid_pointer create_and_init_valid(size_t length, size_t null_count);

void initialize_valids(std::vector<host_valid_pointer>& valids, size_t size, size_t length, size_t null_count);

// Prints a vector and valids
template <typename T>
void print_vector_and_valid(std::vector<T>& v, const gdf_valid_type* valid)
{
  auto functor = [&valid, &v](int index) -> std::string {
    std::string ret = (sizeof(v[index]) == 1)
                      ? std::to_string((int)v[index])
                      : std::to_string(v[index]);

    return gdf_is_valid(valid, index) ? ret : (ret + "|@");
  };
  std::vector<int> indexes(v.size());
  std::iota(std::begin(indexes), std::end(indexes), 0);
  std::transform(indexes.begin(), indexes.end(), std::ostream_iterator<std::string>(std::cout, ", "), functor);
  std::cout << std::endl;
}

template <std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
print_tuples_and_valids(std::tuple<std::vector<Tp>...>& t, std::vector<host_valid_pointer>& valid)
{
  //bottom of compile-time recursion
  //purposely empty...
}

//compile time recursion to print a tuple of vectors
template <std::size_t I = 0, typename... Tp>
  inline typename std::enable_if < I<sizeof...(Tp), void>::type
                                   print_tuples_and_valids(std::tuple<std::vector<Tp>...>& t, std::vector<host_valid_pointer>& valid)
{
  // print the current vector:
  print_vector_and_valid(std::get<I>(t), valid[I].get());

  //recurse to next vector in tuple
  print_tuples_and_valids<I + 1, Tp...>(t, valid);
}

// compile time recursion to compute the element wise equality of two rows
// in two tuples of vectors
template <std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == (sizeof...(Tp)), bool>::type
rows_equal_using_valids(const std::tuple<std::vector<Tp>...>& left, const std::tuple<std::vector<Tp>...>& right, std::vector<host_valid_pointer>& left_valid, std::vector<host_valid_pointer>& right_valid, const size_t left_index, const size_t right_index)
{
  // bottom of recursion
  // If we reach this point, we know the ve
  return true;
}

template <std::size_t I = 0, typename... Tp>
  inline typename std::enable_if < I<sizeof...(Tp), bool>::type
                                   rows_equal_using_valids(const std::tuple<std::vector<Tp>...>& left, const std::tuple<std::vector<Tp>...>& right, std::vector<host_valid_pointer>& left_valid, std::vector<host_valid_pointer>& right_valid, const size_t left_index, const size_t right_index)
{
  auto l_valid = left_valid[I].get();
  auto r_valid = right_valid[I].get();
  if (gdf_is_valid(l_valid, left_index) && gdf_is_valid(r_valid, right_index) && std::get<I>(left)[left_index] == std::get<I>(right)[right_index]) {
    return rows_equal_using_valids<I + 1, Tp...>(left, right, left_valid, right_valid, left_index, right_index);
  } else {
    return false;
  }
}

#endif // VALID_VECTORS_H
