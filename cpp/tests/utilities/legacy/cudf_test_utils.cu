/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Eyal Rozenberg <eyalroz@blazingdb.com>
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

#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <nvstrings/NVCategory.h>
#include <nvstrings/NVStrings.h>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <tests/utilities/legacy/nvcategory_utils.cuh>
#include <cudf/legacy/functions.h>

namespace {

static constexpr char null_signifier = '@';

namespace detail {

// When streaming char-like types, the standard library streams tend to treat
// them as characters rather than numbers, e.g. you would get an 'a' instead of 97.
// The following function(s) ensure we "promote" such values to integers before
// they're streamed

template <typename T>
const T& promote_for_streaming(const T& x) { return x; }


//int promote_for_streaming(const char& x)          { return x; }
//int promote_for_streaming(const unsigned char& x) { return x; }
int promote_for_streaming(const signed char& x)   { return x; }

} // namespace detail


struct column_printer {
  template<typename Element>
  void operator()(gdf_column const* the_column, unsigned min_printing_width,
                  std::ostream& stream)
  {
    cudf::size_type num_rows { the_column->size };

    Element const* column_data { static_cast<Element const*>(the_column->data) };

    std::vector<Element> host_side_data(num_rows);
    cudaMemcpy(host_side_data.data(), column_data, num_rows * sizeof(Element),
               cudaMemcpyDeviceToHost);

    cudf::size_type const num_masks { gdf_valid_allocation_size(num_rows) };
    std::vector<cudf::valid_type> h_mask(num_masks, ~cudf::valid_type { 0 });
    if (nullptr != the_column->valid) {
      cudaMemcpy(h_mask.data(), the_column->valid, num_masks * sizeof(cudf::valid_type),
                 cudaMemcpyDeviceToHost);
    }

    for (cudf::size_type i = 0; i < num_rows; ++i) {
      stream << std::setw(min_printing_width);
      if (gdf_is_valid(h_mask.data(), i)) {
        stream << detail::promote_for_streaming(host_side_data[i]);
      }
      else {
        stream << null_representative;
      }
      stream << ' ';
    }
    stream << std::endl;

    if(the_column->dtype == GDF_STRING_CATEGORY){
      stream<<"Category Data (index | key):\n";

      if(the_column->dtype_info.category != nullptr){
        NVCategory *category =
          static_cast<NVCategory *>(the_column->dtype_info.category);
        
        size_t keys_size = category->keys_size();
        NVStrings *keys = category->get_keys();
        
        if (keys_size>0) {
          char ** data = new char *[keys_size];
          int * byte_sizes = new int[keys_size];
          keys->byte_count(byte_sizes, false);
          for(size_t i=0; i<keys_size; i++){
            data[i]=new char[std::max(2, byte_sizes[i])];
          }

          keys->to_host(data, 0, keys_size);

          for(size_t i=0; i<keys_size; i++){ // null terminate strings
            // TODO: nvstrings overwrites data[i] ifit is a null string
            // Update this based on resolution of https://github.com/rapidsai/custrings/issues/330
            if (byte_sizes[i]!=-1)  
              data[i][byte_sizes[i]]=0;
          }
          
          for(size_t i=0; i<keys_size; i++){ // print category strings
            stream << "(" << i << "|";
            if (data[i] == nullptr)
               stream << null_signifier; // account for null
            else
              stream << data[i];
            stream << ")\t";
          }
          stream<<std::endl;

          for(size_t i=0; i<keys_size; i++){
              delete data[i];
          }
          delete [] data;
          delete [] byte_sizes;
        }
      }
    }
  }
};

/**---------------------------------------------------------------------------*
 * @brief Functor for comparing whether two elements from two gdf_columns are
 * equal.
 *
 *---------------------------------------------------------------------------**/
template <typename T>
struct elements_equal {
  gdf_column lhs_col;
  gdf_column rhs_col;
  bool nulls_are_equivalent;

  using bit_mask_t = bit_mask::bit_mask_t;

  /**---------------------------------------------------------------------------*
   * @brief Constructs functor for comparing elements between two gdf_column's
   *
   * @param lhs The left column for comparison
   * @param rhs The right column for comparison
   * @param nulls_are_equal Desired behavior for whether or not nulls are
   * treated as equal to other nulls. Defaults to true.
   *---------------------------------------------------------------------------**/
  __host__ __device__ elements_equal(gdf_column lhs, gdf_column rhs,
                                     bool nulls_are_equal = true)
      : lhs_col{lhs}, rhs_col{rhs}, nulls_are_equivalent{nulls_are_equal} {}

  __device__ bool operator()(cudf::size_type row) {
    bool const lhs_is_valid{gdf_is_valid(lhs_col.valid, row)};
    bool const rhs_is_valid{gdf_is_valid(rhs_col.valid, row)};

    if (lhs_is_valid and rhs_is_valid) {
      return static_cast<T const*>(lhs_col.data)[row] ==
             static_cast<T const*>(rhs_col.data)[row];
    }

    // If one value is valid but the other is not
    if (lhs_is_valid != rhs_is_valid) {
      return false;
    }

    return nulls_are_equivalent;
  }
};

} // namespace anonymous

/**
 * ---------------------------------------------------------------------------*
 * @brief Compare two gdf_columns on all fields, including pairwise comparison
 * of data and valid arrays
 *
 * @tparam T The type of columns to compare
 * @param left The left column
 * @param right The right column
 * @return bool Whether or not the columns are equal
 * ---------------------------------------------------------------------------**/
template <typename T>
bool gdf_equal_columns(gdf_column const& left, gdf_column const& right)
{
  if (left.size != right.size) return false;
  if (left.dtype != right.dtype) return false;
  if (left.null_count != right.null_count) return false;
  if (left.dtype_info.time_unit != right.dtype_info.time_unit) return false;

  if ((left.col_name == nullptr) != (right.col_name == nullptr))
    return false; // if one is null but not both

  if (left.col_name != nullptr && std::strcmp(left.col_name, right.col_name) != 0)
    return false;

  if ((left.data == nullptr) != (right.data == nullptr))
    return false;  // if one is null but not both
  
  if ((left.valid == nullptr) != (right.valid == nullptr))
    return false;  // if one is null but not both

  if (left.data == nullptr)
    return true;  // logically, both are null

  if (left.dtype == GDF_STRING_CATEGORY) {
    // Transfer input column to host
    std::vector<std::string> left_data, right_data;
    std::vector<cudf::valid_type> left_bitmask, right_bitmask;
    std::tie(left_data, left_bitmask) =
      cudf::test::nvcategory_column_to_host(const_cast<gdf_column*>(&left));
    std::tie(right_data, right_bitmask) =
      cudf::test::nvcategory_column_to_host(const_cast<gdf_column*>(&right));

    CHECK_CUDA(0);

    if (left_data.size() != right_data.size())
      return false;
    
    for (size_t i = 0; i < left_data.size(); i++) {
      bool const left_is_valid{gdf_is_valid(left_bitmask.data(), i)};
      bool const right_is_valid{gdf_is_valid(right_bitmask.data(), i)};

      if (left_is_valid != right_is_valid)
        return false;
      else if (left_is_valid && (left_data[i] != right_data[i]))
        return false;
    }

    return true;
  }
  else {
    if ((left.dtype_info.category != nullptr) || (right.dtype_info.category != nullptr))
      return false;  // category must be nullptr

    bool equal_data = thrust::all_of(rmm::exec_policy()->on(0),
                                     thrust::make_counting_iterator(0),
                                     thrust::make_counting_iterator(left.size),
                                     elements_equal<T>{left, right});
    
    CHECK_CUDA(0);
  
    return equal_data;
  }
}

namespace {

struct columns_equal
{
  template <typename T>
  bool operator()(gdf_column const& left, gdf_column const& right) {
    return gdf_equal_columns<T>(left, right);
  }
};

}; // namespace anonymous

// Type-erased version of gdf_equal_columns
bool gdf_equal_columns(gdf_column const& left, gdf_column const& right)
{
  return cudf::type_dispatcher(left.dtype, columns_equal{}, left, right);
}

void print_gdf_column(gdf_column const * the_column, unsigned min_printing_width, std::ostream& stream)
{
  cudf::type_dispatcher(the_column->dtype, column_printer{}, 
                        the_column, min_printing_width, stream);
}

void print_valid_data(const cudf::valid_type *validity_mask,
                      const size_t num_rows, std::ostream& stream)
{
  cudaError_t error;
  cudaPointerAttributes attrib;
  cudaPointerGetAttributes(&attrib, validity_mask);
  error = cudaGetLastError();

  std::vector<cudf::valid_type> h_mask(gdf_valid_allocation_size(num_rows));
  if (error != cudaErrorInvalidValue && isDeviceType(attrib))
    cudaMemcpy(h_mask.data(), validity_mask, gdf_valid_allocation_size(num_rows),
               cudaMemcpyDeviceToHost);
  else
    memcpy(h_mask.data(), validity_mask, gdf_valid_allocation_size(num_rows));

  std::transform(
      h_mask.begin(), h_mask.begin() + gdf_num_bitmask_elements(num_rows),
      std::ostream_iterator<std::string>(stream, " "), [](cudf::valid_type x) {
        auto bits = std::bitset<GDF_VALID_BITSIZE>(x).to_string(null_signifier);
        return std::string(bits.rbegin(), bits.rend());
      });
  stream << std::endl;
}

cudf::size_type count_valid_bits_host(
    std::vector<cudf::valid_type> const& masks, cudf::size_type const num_rows)
{
  if ((0 == num_rows) || (0 == masks.size())) {
    return 0;
  }

  cudf::size_type count{0};

  // Count the valid bits for all masks except the last one
  for (cudf::size_type i = 0; i < (gdf_num_bitmask_elements(num_rows) - 1); ++i) {
    cudf::valid_type current_mask = masks[i];

    while (current_mask > 0) {
      current_mask &= (current_mask - 1);
      count++;
    }
  }

  // Only count the bits in the last mask that correspond to rows
  int num_rows_last_mask = num_rows % GDF_VALID_BITSIZE;
  if (num_rows_last_mask == 0) {
    num_rows_last_mask = GDF_VALID_BITSIZE;
  }

  // Mask off only the bits that correspond to rows
  cudf::valid_type const rows_mask = ( cudf::valid_type{1} << num_rows_last_mask ) - 1;
  cudf::valid_type last_mask = masks[gdf_num_bitmask_elements(num_rows) - 1] & rows_mask;

  while (last_mask > 0) {
    last_mask &= (last_mask - 1);
    count++;
  }

  return count;
}
