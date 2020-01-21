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
#ifndef CUDF_TEST_UTILS_CUH_
#define CUDF_TEST_UTILS_CUH_

// See this header for all of the recursive handling of tuples of vectors
#include "tuple_vectors.h"

#include <utilities/legacy/cudf_utils.h>
#include <utilities/legacy/bit_util.cuh>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <bitmask/legacy/legacy_bitmask.hpp>

#include <cudf/cudf.h>

#include <rmm/rmm.h>

#include <thrust/logical.h>

#include <bitset>
#include <numeric> // for std::accumulate
#include <memory>

#include <tests/utilities/cudf_gmock.hpp>
#define ASSERT_RMM_SUCCEEDED(expr)  ASSERT_EQ(RMM_SUCCESS, expr)
#define EXPECT_RMM_SUCCEEDED(expr)  EXPECT_EQ(RMM_SUCCESS, expr)

// We use this single character to represent a gdf_column element being null
constexpr const char null_representative = '@';

// Type for a unique_ptr to a gdf_column with a custom deleter
// Custom deleter is defined at construction
using gdf_col_pointer = typename std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>;

/**
 * @brief  Counts the number of valid bits for the specified number of rows
 * in the host vector of cudf::valid_type masks
 *
 * @param masks The host vector of masks whose bits will be counted
 * @param num_rows The number of bits to count
 *
 * @returns The number of valid bits in [0, num_rows) in the host vector of
 * masks
 **/
cudf::size_type count_valid_bits_host(
  std::vector<cudf::valid_type> const& masks, cudf::size_type const num_rows);


/**
 * @brief Prints a "broken-down" column's data to the standard output stream,
 * while accounting for null indication.
 *
 * @note See the @ref gdf_column variant
 *
 */
template <typename Element>
void print_typed_column(
    const Element *    __restrict__  col_data,
    cudf::valid_type *   __restrict__  validity_mask,
    const size_t                     size_in_elements,
    unsigned                         min_element_print_width = 1)
{
  static_assert(not std::is_same<Element, void>::value, "Can't print void* columns - a concrete type is needed");
  if (col_data == nullptr) {
      std::cout << "(nullptr column data pointer - nothing to print)";
      return;
  }
  if (size_in_elements == 0) {
      std::cout << "(empty column)";
      return;
  }
  std::vector<Element> h_data(size_in_elements);

  cudaMemcpy(h_data.data(), col_data, size_in_elements * sizeof(Element), cudaMemcpyDefault);

  const size_t num_valid_type_elements = gdf_valid_allocation_size(size_in_elements);
  std::vector<cudf::valid_type> h_mask(num_valid_type_elements );
  if(nullptr != validity_mask)
  {
    cudaMemcpy(h_mask.data(), validity_mask, num_valid_type_elements, cudaMemcpyDefault);
  }

  for(size_t i = 0; i < size_in_elements; ++i)
  {
      std::cout << std::setw(min_element_print_width);
      if ((validity_mask == nullptr) or gdf_is_valid(h_mask.data(), i))
      {
          if (sizeof(Element) < sizeof(int)) {
              std::cout << (int) h_data[i];
          }
          else {
              std::cout << h_data[i];
          }
      }
      else {
          std::cout << null_representative;
      }
      if (i + 1 < size_in_elements) { std::cout << ' '; }
  }
  std::cout << std::endl;
}

/**
 * @brief No-frills, single-line printing of a gdf_column's (typed) data to the
 * standard output stream, while accounting for nulls
 *
 * @todo More bells and whistles here would be nice to have.
 *
 * @param column[in] a @ref gdf_column to print.
 * @param min_element_print_width[in] Every element will take up this any
 * characters when printed.
 */
template <typename Element>
inline void print_typed_column(const gdf_column& column, unsigned min_element_print_width = 1)
{
    print_typed_column<Element>(
        static_cast<const Element*>(column.data),
        column.valid,
        column.size,
        min_element_print_width);
}

/**
 * @brief No-frills, single-line printing of a gdf_column's (typed) data to the
 * standard output stream, while accounting for nulls
 *
 * @note See the @ref gdf_column variant
 */
void print_gdf_column(gdf_column const *column, unsigned min_element_print_width = 1,
                      std::ostream& stream = std::cout);


/** ---------------------------------------------------------------------------*
 * @brief prints validity data from either a host or device pointer
 *
 * @param validity_mask The validity bitmask to print
 * @param length Length of the mask in bits
 * @param ostream Output stream, defaults to the standard output stream
 *
 * @note the mask may have more space allocated for it than is necessary 
 * for the specified length; in particular, it will have "slack" bits
 * in the last byte, if length % 8 != 0. Such slack bits are ignored and
 * not printed. Usually, length is the number of elements of a gdf_column.
 * ---------------------------------------------------------------------------**/
void print_valid_data(const cudf::valid_type* validity_mask, const size_t length,
                      std::ostream& stream = std::cout);

/* --------------------------------------------------------------------------*/
/**
 * @brief  Creates a unique_ptr that wraps a gdf_column structure intialized with a host vector
 *
 * @param host_vector The host vector whose data is used to initialize the gdf_column
 *
 * @returns A unique_ptr wrapping the new gdf_column
 */
/* ----------------------------------------------------------------------------*/
template <typename ColumnType>
gdf_col_pointer create_gdf_column(std::vector<ColumnType> const & host_vector,
                                  std::vector<cudf::valid_type> const & valid_vector = std::vector<cudf::valid_type>())
{
  // Get the corresponding gdf_dtype for the ColumnType
  gdf_dtype gdf_col_type{cudf::gdf_dtype_of<ColumnType>()};

   CUDF_EXPECTS(gdf_col_type != GDF_invalid, "Cannot create columns with the GDF_invalid element type");

  // Create a new instance of a gdf_column with a custom deleter that will free
  // the associated device memory when it eventually goes out of scope
  auto deleter = [](gdf_column* col) {
    col->size = 0;
    RMM_FREE(col->data, 0);
    RMM_FREE(col->valid, 0);
  };
  gdf_col_pointer the_column{new gdf_column{}, deleter};

  // Allocate device storage for gdf_column and copy contents from host_vector
  RMM_ALLOC(&(the_column->data), host_vector.size() * sizeof(ColumnType), 0);
  cudaMemcpy(the_column->data, host_vector.data(), host_vector.size() * sizeof(ColumnType), cudaMemcpyHostToDevice);

  // Fill the gdf_column members
  the_column->size = host_vector.size();
  the_column->dtype = gdf_col_type;
  gdf_dtype_extra_info extra_info{TIME_UNIT_NONE};
  the_column->dtype_info = extra_info;
  the_column->col_name = nullptr;

  // If a validity bitmask vector was passed in, allocate device storage
  // and copy its contents from the host vector
  if(valid_vector.size() > 0)
  {
    RMM_ALLOC((void**)&(the_column->valid), valid_vector.size() * sizeof(cudf::valid_type), 0);
    cudaMemcpy(the_column->valid, valid_vector.data(), valid_vector.size() * sizeof(cudf::valid_type), cudaMemcpyHostToDevice);
    the_column->null_count = (host_vector.size() - count_valid_bits_host(valid_vector, host_vector.size()));
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
  const size_t num_masks = gdf_valid_allocation_size(num_rows);

  // Initialize the valid mask for this column using the initializer
  std::vector<cudf::valid_type> valid_masks(num_masks,0);
  for(size_t row = 0; row < num_rows; ++row){
    if(true == bit_initializer(row, col_index))
    {
      cudf::util::turn_bit_on(valid_masks.data(), row);
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
template <typename T, typename valid_initializer_t>
std::vector<gdf_col_pointer> initialize_gdf_columns(
    std::vector<std::vector<T>> columns, valid_initializer_t bit_initializer) {
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

template <typename T>
std::vector<gdf_col_pointer> initialize_gdf_columns(
    std::vector<std::vector<T>> columns) {

  return initialize_gdf_columns(columns,
                                [](size_t row, size_t col) { return true; });
}

/**
 * ---------------------------------------------------------------------------*
 * @brief Compare two gdf_columns on all fields, including pairwise comparison
 * of data and valid arrays. 
 * 
 * Uses type_dispatcher to dispatch the data comparison
 *
 * @param left The left column
 * @param right The right column
 * @return bool Whether or not the columns are equal
 * ---------------------------------------------------------------------------**/
bool gdf_equal_columns(gdf_column const& left, gdf_column const &right);

#endif // CUDF_TEST_UTILS_CUH_
