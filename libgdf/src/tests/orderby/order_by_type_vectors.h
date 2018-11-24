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

#ifndef ORDER_BY_TYPE_VECTORS_H
#define ORDER_BY_TYPE_VECTORS_H

#include <memory>
#include <vector>
#include <string>
#include <gdf/utils.h>

#include "valid_vectors.h"

// Initialize valids
void initialize_order_by_types(std::vector<char>& order_by_types, size_t length)
{
  order_by_types.clear();
  order_by_types.reserve(length);

  for (size_t i = 0; i < length; ++i) {
    order_by_types.push_back((char)(std::rand() % 2));
  }
}

// Prints a vector, its valids and the its sort order
template <typename T>
void print_vector_valids_and_sort_order(std::vector<T>& v, gdf_valid_type* valid, order_by_type sort_order)
{
  std::cout << (sort_order == GDF_ORDER_ASC ? "ASC ORDER: " : "DESC ORDER: " );

  auto functor = [&valid, &v](int index) -> std::string {
    if (gdf_is_valid(valid, index))
      return std::to_string((int)v[index]);
    return std::string("@");
  };
  std::vector<int> indexes(v.size());
  std::iota(std::begin(indexes), std::end(indexes), 0);
  std::transform(indexes.begin(), indexes.end(), std::ostream_iterator<std::string>(std::cout, ", "), functor);
  std::cout << std::endl;
}

template <std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
print_tuples_valids_and_order_by_types(std::tuple<std::vector<Tp>...>& t, std::vector<host_valid_pointer>& valid, std::vector<char>& order_by_types )
{
  //bottom of compile-time recursion
  //purposely empty...
}

//compile time recursion to print a tuple of vectors
template <std::size_t I = 0, typename... Tp>
inline typename std::enable_if < I<sizeof...(Tp), void>::type
print_tuples_valids_and_order_by_types(std::tuple<std::vector<Tp>...>& t, std::vector<host_valid_pointer>& valid, std::vector<char>& order_by_types)
{
  // print the current vector:
  print_vector_valids_and_sort_order(std::get<I>(t), valid[I].get(), (order_by_type)order_by_types[I]);

  //recurse to next vector in tuple
  print_tuples_valids_and_order_by_types<I + 1, Tp...>(t, valid, order_by_types);
}
#endif // ORDER_BY_TYPE_VECTORS_H
