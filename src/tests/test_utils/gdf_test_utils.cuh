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
#include "tuple_vectors.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <gdf/utils.h>
#include <memory>
#include "../../util/bit_util.cuh"

// Type for a unique_ptr to a gdf_column with a custom deleter
// Custom deleter is defined at construction
using gdf_col_pointer = typename std::unique_ptr<gdf_column, 
                                                 std::function<void(gdf_column*)>>;

template <typename col_type>
void print_typed_column(col_type * col_data, 
                        gdf_valid_type * validity_mask, 
                        const size_t num_rows)
{

  std::vector<col_type> h_data(num_rows);
  cudaMemcpy(h_data.data(), col_data, num_rows * sizeof(col_type), cudaMemcpyDeviceToHost);


  const size_t num_masks = gdf_get_num_chars_bitmask(num_rows);
  std::vector<gdf_valid_type> h_mask(num_masks);
  if(nullptr != validity_mask)
  {
    cudaMemcpy(h_mask.data(), validity_mask, num_masks * sizeof(gdf_valid_type), cudaMemcpyDeviceToHost);
  }


  for(size_t i = 0; i < num_rows; ++i)
  {
    // If the element is valid, print it's value
    if(true == gdf_is_valid(h_mask.data(), i))
    {
      std::cout << h_data[i] << " ";
    }
    // Otherwise, print an @ to represent a null value
    else
    {
      std::cout << "@" << " ";
    }
  }
  std::cout << std::endl;
}

void print_gdf_column(gdf_column const * the_column)
{
  const size_t num_rows = the_column->size;

  const gdf_dtype gdf_col_type = the_column->dtype;
  switch(gdf_col_type)
  {
    case GDF_INT8:
      {
        using col_type = int8_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_INT16:
      {
        using col_type = int16_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_INT32:
      {
        using col_type = int32_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_INT64:
      {
        using col_type = int64_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_FLOAT32:
      {
        using col_type = float;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_FLOAT64:
      {
        using col_type = double;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    default:
      {
        std::cout << "Attempted to print unsupported type.\n";
      }
  }
}

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
gdf_col_pointer create_gdf_column(std::vector<col_type> const & host_vector,
                                  std::vector<gdf_valid_type> const & valid_vector = std::vector<gdf_valid_type>())
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
  auto deleter = [](gdf_column* col){
                                      col->size = 0; 
                                      if(nullptr != col->data){cudaFree(col->data);} 
                                      if(nullptr != col->valid){cudaFree(col->valid);}
                                    };
  gdf_col_pointer the_column{new gdf_column, deleter};

  // Allocate device storage for gdf_column and copy contents from host_vector
  cudaMalloc(&(the_column->data), host_vector.size() * sizeof(col_type));
  cudaMemcpy(the_column->data, host_vector.data(), host_vector.size() * sizeof(col_type), cudaMemcpyHostToDevice);


  // If a validity bitmask vector was passed in, allocate device storage 
  // and copy its contents from the host vector
  if(valid_vector.size() > 0)
  {
    cudaMalloc(&(the_column->valid), valid_vector.size() * sizeof(gdf_valid_type));
    cudaMemcpy(the_column->valid, valid_vector.data(), valid_vector.size() * sizeof(gdf_valid_type), cudaMemcpyHostToDevice);
  }
  else
  {
    the_column->valid = nullptr;
  }

  // Fill the gdf_column members
  the_column->size = host_vector.size();
  the_column->dtype = gdf_col_type;
  gdf_dtype_extra_info extra_info;
  extra_info.time_unit = TIME_UNIT_NONE;
  the_column->dtype_info = extra_info;

  return the_column;
}

// Compile time recursion to convert each vector in a tuple of vectors into
// a gdf_column and append it to a vector of gdf_columns
template<typename valid_initializer_t, std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
convert_tuple_to_gdf_columns(std::vector<gdf_col_pointer> &gdf_columns,std::tuple<std::vector<Tp>...>& t, 
                             valid_initializer_t bit_initializer)
{
  //bottom of compile-time recursion
  //purposely empty...
}
template<typename valid_initializer_t, std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type
convert_tuple_to_gdf_columns(std::vector<gdf_col_pointer> &gdf_columns,std::tuple<std::vector<Tp>...>& t,
                             valid_initializer_t bit_initializer)
{

  const size_t num_rows = std::get<I>(t).size();
  const size_t num_masks = gdf_get_num_chars_bitmask(num_rows);

  // Initialize the valid mask for this column using the initializer
  std::vector<gdf_valid_type> valid_masks(num_masks,0);
  const size_t column = I;
  for(size_t row = 0; row < num_rows; ++row){
    if(true == bit_initializer(row, column))
    {
      gdf::util::turn_bit_on(valid_masks.data(), row);
    }
  }

  // Creates a gdf_column for the current vector and pushes it onto
  // the vector of gdf_columns
  gdf_columns.push_back(create_gdf_column(std::get<I>(t), valid_masks));

  //recurse to next vector in tuple
  convert_tuple_to_gdf_columns<valid_initializer_t,I + 1, Tp...>(gdf_columns, t, bit_initializer);
}

// Converts a tuple of host vectors into a vector of gdf_columns

template<typename valid_initializer_t, typename... Tp>
std::vector<gdf_col_pointer> initialize_gdf_columns(std::tuple<std::vector<Tp>...> & host_columns, 
                                                    valid_initializer_t bit_initializer)
{
  std::vector<gdf_col_pointer> gdf_columns;
  convert_tuple_to_gdf_columns(gdf_columns, host_columns, bit_initializer);
  return gdf_columns;
}


// Overload for default initialization of validity bitmasks which 
// sets every element to valid
template<typename... Tp>
std::vector<gdf_col_pointer> initialize_gdf_columns(std::tuple<std::vector<Tp>...> & host_columns )
{
  return initialize_gdf_columns(host_columns, 
                                [](const size_t row, const size_t col){return true;});
}


#endif
