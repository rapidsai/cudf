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

#include <cudf.h>
#include <rmm/rmm.h>
#include <cudf/functions.h>
#include <utilities/cudf_utils.h>
#include <utilities/bit_util.cuh>

#include <bitset>
#include <numeric> // for std::accumulate
#include <memory>

#include <thrust/equal.h>

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

/** ---------------------------------------------------------------------------*
 * @brief prints validity data from either a host or device pointer
 * 
 * @param validity_mask The validity bitmask to print
 * @param num_rows The length of the column (not the bitmask) in rows
 * ---------------------------------------------------------------------------**/
void print_valid_data(const gdf_valid_type *validity_mask, 
                      const size_t num_rows)
{
  cudaError_t error;
  cudaPointerAttributes attrib;
  cudaPointerGetAttributes(&attrib, validity_mask);
  error = cudaGetLastError();

  const size_t num_masks = gdf_get_num_chars_bitmask(num_rows);
  std::vector<gdf_valid_type> h_mask(num_masks);
  if (error != cudaErrorInvalidValue && attrib.memoryType == cudaMemoryTypeDevice)
    cudaMemcpy(h_mask.data(), validity_mask, num_masks * sizeof(gdf_valid_type), cudaMemcpyDeviceToHost);
  else
    memcpy(h_mask.data(), validity_mask, num_masks * sizeof(gdf_valid_type));

  std::transform(h_mask.begin(), h_mask.end(), std::ostream_iterator<std::string>(std::cout, " "), 
                 [](gdf_valid_type x){ 
                   auto bits = std::bitset<GDF_VALID_BITSIZE>(x).to_string('@'); 
                   return std::string(bits.rbegin(), bits.rend());  
                 });
  std::cout << std::endl;
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
                                      if(nullptr != col->data){RMM_FREE(col->data, 0);} 
                                      if(nullptr != col->valid){RMM_FREE(col->valid, 0);}
                                    };
  gdf_col_pointer the_column{new gdf_column, deleter};

  // Allocate device storage for gdf_column and copy contents from host_vector
  RMM_ALLOC(&(the_column->data), host_vector.size() * sizeof(col_type), 0);
  cudaMemcpy(the_column->data, host_vector.data(), host_vector.size() * sizeof(col_type), cudaMemcpyHostToDevice);

  // Fill the gdf_column members
  the_column->size = host_vector.size();
  the_column->dtype = gdf_col_type;
  gdf_dtype_extra_info extra_info;
  extra_info.time_unit = TIME_UNIT_NONE;
  the_column->dtype_info = extra_info;

  // If a validity bitmask vector was passed in, allocate device storage 
  // and copy its contents from the host vector
  if(valid_vector.size() > 0)
  {
    RMM_ALLOC((void**)&(the_column->valid), valid_vector.size() * sizeof(gdf_valid_type), 0);
    cudaMemcpy(the_column->valid, valid_vector.data(), valid_vector.size() * sizeof(gdf_valid_type), cudaMemcpyHostToDevice);

    // Count the number of null bits
    // count in all but last element in case it is not full
    the_column->null_count = std::accumulate(valid_vector.begin(), valid_vector.end() - 1, 0,
      [](gdf_size_type s, gdf_valid_type x) { 
        return s + std::bitset<GDF_VALID_BITSIZE>(x).flip().count(); 
      });
    // Now count the bits in the last mask
    size_t unused_bits = GDF_VALID_BITSIZE - the_column->size % GDF_VALID_BITSIZE;
    if (GDF_VALID_BITSIZE == unused_bits) unused_bits = 0;
    auto last_mask = std::bitset<GDF_VALID_BITSIZE>(*(valid_vector.end()-1)).flip();
    last_mask = (last_mask << unused_bits) >> unused_bits;
    the_column->null_count += last_mask.count();
  }
  else
  {
    the_column->valid = nullptr;
    the_column->null_count = 0;
  }

  return the_column;
}

// This helper generates the validity mask and creates the GDF column
// Used by the various initializers below.
template <typename T, typename valid_initializer_t>
gdf_col_pointer init_gdf_column(std::vector<T> data, size_t col_index, valid_initializer_t bit_initializer)
{
  const size_t num_rows = data.size();
  const size_t num_masks = gdf_get_num_chars_bitmask(num_rows);

  // Initialize the valid mask for this column using the initializer
  std::vector<gdf_valid_type> valid_masks(num_masks,0);
  for(size_t row = 0; row < num_rows; ++row){
    if(true == bit_initializer(row, col_index))
    {
      gdf::util::turn_bit_on(valid_masks.data(), row);
    }
  }

  return create_gdf_column(data, valid_masks);
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
  const size_t column = I;

  // Creates a gdf_column for the current vector and pushes it onto
  // the vector of gdf_columns
  gdf_columns.push_back(init_gdf_column(std::get<I>(t), column, bit_initializer));

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

// This version of initialize_gdf_columns assumes takes a vector of same-typed column data as input
// and a validity mask initializer function
template<typename T, typename valid_initializer_t>
std::vector<gdf_col_pointer> initialize_gdf_columns(std::vector< std::vector<T> > columns, valid_initializer_t bit_initializer)
{
  std::vector<gdf_col_pointer> gdf_columns;

  size_t col = 0;
  
  for (auto column : columns)
  {
    // Creates a gdf_column for the current vector and pushes it onto
    // the vector of gdf_columns
    gdf_columns.push_back(init_gdf_column(column, col++, bit_initializer));
  }

  return gdf_columns;
}

/** ---------------------------------------------------------------------------*
 * @brief Compare two gdf_columns on all fields, including pairwise comparison of 
 * data and valid arrays
 * 
 * @tparam T The type of columns to compare
 * @param left The left column
 * @param right The right column
 * @return bool Whether or not the columns are equal
 * ---------------------------------------------------------------------------**/
template <typename T>
bool gdf_equal_columns(gdf_column* left, gdf_column* right)
{
  if (left->size != right->size) return false;
  if (left->dtype != right->dtype) return false;
  if (left->null_count != right->null_count) return false;
  if (left->dtype_info.time_unit != right->dtype_info.time_unit) return false;
  
  if (!(left->data && right->data)) return false; // if one is null but not both
  
  if (!thrust::equal(thrust::cuda::par, 
                     reinterpret_cast<T*>(left->data), 
                     reinterpret_cast<T*>(left->data) + left->size, 
                     reinterpret_cast<T*>(right->data)) ) 
    return false;
  
  if (!(left->valid && right->valid)) return false; // if one is null but not both
  
  if (!thrust::equal(thrust::cuda::par, 
                     left->valid, 
                     left->valid + gdf_get_num_chars_bitmask(left->size), 
                     right->valid))
    return false;
  
  return true;
}


#endif
