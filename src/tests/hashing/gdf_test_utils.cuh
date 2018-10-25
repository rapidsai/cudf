/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#ifndef GDF_TEST_UTILS_H
#define GDF_TEST_UTILS_H

// See this header for all of the recursive handling of tuples of vectors
#include "../join/tuple_vectors.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>

// Type for a unique_ptr to a gdf_column with a custom deleter
// Custom deleter is defined at construction
using gdf_col_pointer = typename std::unique_ptr<gdf_column, 
                                                 std::function<void(gdf_column*)>>;

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Creates a unique_ptr that wraps a gdf_column structure intialized with a host vector
 *
 * @Param host_vector The host vector whose data is used to initialize the gdf_column
 *
 * @Returns A unique_ptr wrapping the new gdf_column
 */
/* ----------------------------------------------------------------------------*/
  template <typename col_type>
gdf_col_pointer create_gdf_column(std::vector<col_type> const & host_vector)
{
  // Deduce the type and set the gdf_dtype accordingly
  gdf_dtype gdf_col_type;
  if(std::is_same<col_type,int8_t>::value) gdf_col_type = GDF_INT8;
  else if(std::is_same<col_type,uint8_t>::value) gdf_col_type = GDF_INT8;
  else if(std::is_same<col_type,int16_t>::value) gdf_col_type = GDF_INT16;
  else if(std::is_same<col_type,uint16_t>::value) gdf_col_type = GDF_INT16;
  else if(std::is_same<col_type,int32_t>::value) gdf_col_type = GDF_INT32;
  else if(std::is_same<col_type,uint32_t>::value) gdf_col_type = GDF_INT32;
  else if(std::is_same<col_type,int64_t>::value) gdf_col_type = GDF_INT64;
  else if(std::is_same<col_type,uint64_t>::value) gdf_col_type = GDF_INT64;
  else if(std::is_same<col_type,float>::value) gdf_col_type = GDF_FLOAT32;
  else if(std::is_same<col_type,double>::value) gdf_col_type = GDF_FLOAT64;

  // Create a new instance of a gdf_column with a custom deleter that will free
  // the associated device memory when it eventually goes out of scope
  auto deleter = [](gdf_column* col){col->size = 0; rmmFree(col->data, 0);};
  gdf_col_pointer the_column{new gdf_column, deleter};

  // Allocate device storage for gdf_column and copy contents from host_vector
  rmmAlloc(&(the_column->data), host_vector.size() * sizeof(col_type), 0);
  cudaMemcpy(the_column->data, host_vector.data(), host_vector.size() * sizeof(col_type), cudaMemcpyHostToDevice);

  // Fill the gdf_column members
  the_column->valid = nullptr;
  the_column->size = host_vector.size();
  the_column->dtype = gdf_col_type;
  gdf_dtype_extra_info extra_info;
  extra_info.time_unit = TIME_UNIT_NONE;
  the_column->dtype_info = extra_info;

  return the_column;
}

// Compile time recursion to convert each vector in a tuple of vectors into
// a gdf_column and append it to a vector of gdf_columns
template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
convert_tuple_to_gdf_columns(std::vector<gdf_col_pointer> &gdf_columns,std::tuple<std::vector<Tp>...>& t)
{
  //bottom of compile-time recursion
  //purposely empty...
}
template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type
convert_tuple_to_gdf_columns(std::vector<gdf_col_pointer> &gdf_columns,std::tuple<std::vector<Tp>...>& t)
{
  // Creates a gdf_column for the current vector and pushes it onto
  // the vector of gdf_columns
  gdf_columns.push_back(create_gdf_column(std::get<I>(t)));

  //recurse to next vector in tuple
  convert_tuple_to_gdf_columns<I + 1, Tp...>(gdf_columns, t);
}

// Converts a tuple of host vectors into a vector of gdf_columns
template<typename... Tp>
std::vector<gdf_col_pointer> initialize_gdf_columns(std::tuple<std::vector<Tp>...> & host_columns)
{
  std::vector<gdf_col_pointer> gdf_columns;
  convert_tuple_to_gdf_columns(gdf_columns, host_columns);
  return gdf_columns;
}

#endif
