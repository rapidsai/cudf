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

#ifndef GDF_TABLE_H
#define GDF_TABLE_H

#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/error_utils.hpp"
#include "hash/hash_functions.cuh"
#include "hash/managed.cuh"
#include "copying/gather.hpp"
#include "sqls/sqls_rtti_comp.h"
#include "bitmask/legacy_bitmask.hpp"

#include <thrust/tabulate.h>
#include <cassert>
#include "utilities/type_dispatcher.hpp"



/* --------------------------------------------------------------------------*/
/** 
 * @brief  Computes the validity mask for the rows in the gdf_table.

   If a single value in a row of the table is NULL, then the entire row is 
   considered to be NULL. Therefore, we can AND all of the bitmasks of each
   column together to get a bitmask for the validity of each row.
 */
/* ----------------------------------------------------------------------------*/
template <typename size_type>
struct row_masker
{
  row_masker(gdf_valid_type ** column_masks, const size_type num_cols)
    : column_valid_masks{column_masks}, num_columns(num_cols)
    { }
   
  /* --------------------------------------------------------------------------*/
  /** 
   * @brief Computes the bit-wise AND across all columns for the specified mask
   * 
   * @param mask_number The index of the mask to compute the bit-wise AND across all columns
   * 
   * @returns The bit-wise AND across all columns for the specified mask number
   */
  /* ----------------------------------------------------------------------------*/
  __device__ __forceinline__
  gdf_valid_type operator()(const size_type mask_number)
  {
    // Intialize row validity mask with all bits set to 1
    gdf_valid_type row_valid_mask{0};
    row_valid_mask = ~(row_valid_mask);

    for(size_type i = 0; i < num_columns; ++i) 
    {
      const gdf_valid_type * current_column_mask = column_valid_masks[i];

      // The column validity mask is optional and can be nullptr
      if(nullptr != current_column_mask){
        row_valid_mask &= current_column_mask[mask_number];
      }
    }
    return row_valid_mask;
  }

  const size_type num_columns;
  gdf_valid_type ** column_valid_masks;
};

/* --------------------------------------------------------------------------*/
/** 

    The gdf_table class is meant to wrap a set of gdf_columns and provide functions
    for operating across all of the columns. It can be thought of as a `matrix`
    whose columns can be of different data types. Thinking of it as a matrix,
    many row-wise operations are defined, such as checking if two rows in a table
    are equivalent.
 */
/* ----------------------------------------------------------------------------*/
template <typename T, typename byte_t = unsigned char>
class gdf_table : public managed
{
public:

  using size_type = T;
  using byte_type = byte_t;

  gdf_table(size_type num_cols, gdf_column ** gdf_columns) 
    : num_columns(num_cols), host_columns(gdf_columns)
  {
    assert(num_cols > 0);
    assert(nullptr != host_columns[0]);
    column_length = host_columns[0]->size;

    if(column_length > 0)
    {
      assert(nullptr != host_columns[0]->data);
    }


    // Copy pointers to each column's data, types, and validity bitmasks 
    // to contiguous host vectors (AoS to SoA conversion)
    std::vector<void*> columns_data(num_cols);
    std::vector<gdf_valid_type*> columns_valids(num_cols);
    std::vector<gdf_dtype> columns_types(num_cols);
    std::vector<byte_type> columns_byte_widths(num_cols);

    for(size_type i = 0; i < num_cols; ++i)
    {
      gdf_column * const current_column = host_columns[i];
      assert(nullptr != current_column);
      assert(column_length == current_column->size);
      if(column_length > 0)
      {
        assert(nullptr != current_column->data);
      }
	
      // Compute the size of a row in the table in bytes
      int column_width_bytes{0};
      if(GDF_SUCCESS == get_column_byte_width(current_column, &column_width_bytes))
      {
        row_size_bytes += column_width_bytes;
        // Store the byte width of each column in a device array
        columns_byte_widths[i] = (static_cast<byte_type>(row_size_bytes));
      }
      else
      {
        std::cerr << "Attempted to get column byte width of unsupported GDF datatype.\n";
        columns_byte_widths[i] = 0;
      }

      columns_data[i] = (host_columns[i]->data);
      columns_valids[i] = (host_columns[i]->valid);
      columns_types[i] = (host_columns[i]->dtype);
    }

    // Copy host vectors to device vectors
    device_columns_data = columns_data;
    device_columns_valids = columns_valids;
    device_columns_types = columns_types;
    device_column_byte_widths = columns_byte_widths;

    d_columns_data_ptr = device_columns_data.data().get();
    d_columns_valids_ptr = device_columns_valids.data().get();
    d_columns_types_ptr = device_columns_types.data().get();
    d_columns_byte_widths_ptr = device_column_byte_widths.data().get();

    // Allocate storage sufficient to hold a validity bit for every row
    // in the table
    const size_type mask_size = gdf_valid_allocation_size(column_length);
    device_row_valid.resize(mask_size);

    // If a row contains a single NULL value, then the entire row is considered
    // to be NULL, therefore initialize the row-validity mask with the
    // bit-wise AND of the validity mask of all the columns
    thrust::tabulate(rmm::exec_policy()->on(0),
                     device_row_valid.begin(),
                     device_row_valid.end(),
                     row_masker<size_type>(d_columns_valids_ptr, num_cols));

    d_row_valid = device_row_valid.data().get();
  }

  ~gdf_table(){}


  /* --------------------------------------------------------------------------*/
  /** 
   * @brief  Updates the length of the gdf_columns in the table
   * 
   * @param new_length The new length
   */
  /* ----------------------------------------------------------------------------*/
  void set_column_length(const size_type new_length)
  {
    column_length = new_length;

    for(size_type i = 0; i < num_columns; ++i)
    {
      host_columns[i]->size = this->column_length;
    }
  }


  size_type get_num_columns() const
  {
    return num_columns;
  }


  __host__ 
  gdf_column * get_column(size_type column_index) const
  {
    return host_columns[column_index];
  }

   __host__ 
  gdf_column ** get_columns() const
  {
    return host_columns;
  }

  __host__ __device__
  size_type get_column_length() const
  {
    return column_length;
  }

  __device__ bool is_row_valid(size_type row_index) const
  {
    const bool row_valid = gdf_is_valid(d_row_valid, row_index);

    return row_valid;
  }


  /* --------------------------------------------------------------------------*/
  /** 
   * @brief  Gets the size in bytes of a row in the gdf_table, i.e., the sum of 
   * the byte widths of all columns in the table
   * 
   * @returns The size in bytes of the row in the table
   */
  /* ----------------------------------------------------------------------------*/
  byte_type get_row_size_bytes() const
  {
    return row_size_bytes;
  }


  /* --------------------------------------------------------------------------*/
  /** 
   * @brief  Packs the elements of a specified row into a contiguous byte-buffer
   *
   * This function is called by a single thread, and the thread will copy each element
   * of the row into a single contiguous buffer. TODO: This could be done by multiple threads
   * by passing in a cooperative group. 
   * 
   * @param index The row of the table to return
   * @param row_byte_buffer A pointer to a preallocated buffer large enough to hold a 
      row of the table 
   * 
   */
  /* ----------------------------------------------------------------------------*/
  // TODO Is there a less hacky way to do this? 
  __device__
  gdf_error get_packed_row_values(size_type row_index, void * row_byte_buffer) const
  {
    if(nullptr == row_byte_buffer) {
      return GDF_DATASET_EMPTY;
    }

    // Pack the element from each column in the row into the buffer
    for(size_type i = 0; i < num_columns; ++i)
    {
      const gdf_dtype source_col_type = d_columns_types_ptr[i];

      cudf::type_dispatcher(source_col_type,
                            copy_element{},
                            row_byte_buffer, i,
                            d_columns_data_ptr[i], row_index);

    }
    return GDF_SUCCESS;
  }

  struct copy_element{
    template <typename ColumnType>
    __device__ __forceinline__
    void operator()(void * target_column, size_type target_row_index,
                    void const * source_column, size_type source_row_index)
    {
      ColumnType& target_value { static_cast<ColumnType*>(target_column)[target_row_index] };
      ColumnType const& source_value{static_cast<ColumnType const*>(source_column)[source_row_index]};
      target_value = source_value;
    }

  };

  /* --------------------------------------------------------------------------*/
  /**
   * @Synopsis  Packs the validity mask of a specified row into a contiguous byte-buffer 
   * 
   * This function is called by a single thread, and the thread will copy each element
   * of the row into a single contiguous buffer.
   * 
   * @param row_index The row of the table to return validity mask for
   * @param row_valid_byte_buffer A pointer to a preallocated buffer large enough to hold
      the validity bitmask of a row of the table
   */
  /* ----------------------------------------------------------------------------*/
  __device__
  gdf_error get_row_valids(size_type row_index, gdf_valid_type * row_valid_byte_buffer) const
  {
    if(nullptr == row_valid_byte_buffer) {
      return GDF_DATASET_EMPTY;
    }
    
    for(size_type i = 0; i < num_columns; i++)
    {
      // get validity of item in column in self
      if (gdf_is_valid(d_columns_valids_ptr[i], row_index))
        // set validity in output buffer
        row_valid_byte_buffer[i / GDF_VALID_BITSIZE] |= (gdf_valid_type{1} << (i % GDF_VALID_BITSIZE));
    }
    return GDF_SUCCESS;
  }

  __device__
  gdf_valid_type* get_columns_device_valids_ptr(size_type column_index)
  {
    return d_columns_valids_ptr[column_index];
  }
    /* --------------------------------------------------------------------------*/
    /** 
     * @brief  Copies a row from a source table to a target row in this table
     *  
     * This device function should be called by a single thread and the thread will copy all of 
     * the elements in the row from one table to the other. TODO: In the future, this could be done
     * by multiple threads by passing in a cooperative group.
     * 
     * @param other The other table from which the row is copied
     * @param my_row_index The index of the row in this table that will be written to
     * @param other_row_index The index of the row from the other table that will be copied from
     */
    /* ----------------------------------------------------------------------------*/
  __device__ 
  gdf_error copy_row(gdf_table const & source,
                     const size_type target_row_index,
                     const size_type source_row_index)
  {
    for(size_type i = 0; i < num_columns; ++i)
    {
      const gdf_dtype target_col_type = d_columns_types_ptr[i];
      const gdf_dtype source_col_type = source.d_columns_types_ptr[i];
    
      if(target_col_type != source_col_type)
      {
        return GDF_DTYPE_MISMATCH;
      }

      cudf::type_dispatcher(target_col_type,
                            copy_element{},
                            d_columns_data_ptr[i],
                            target_row_index,
                            source.d_columns_data_ptr[i],
                            source_row_index);

    }
    return GDF_SUCCESS;
  }


  struct elements_are_equal{
    template <typename ColumnType>
    __device__ __forceinline__
    bool operator()(void const * lhs_column, size_type lhs_row_index,
                    void const * rhs_column, size_type rhs_row_index)
    {
      ColumnType const lhs_elem{static_cast<ColumnType const*>(lhs_column)[lhs_row_index]};
      ColumnType const rhs_elem{static_cast<ColumnType const*>(rhs_column)[rhs_row_index]};
      return lhs_elem == rhs_elem;
    }
  };

  /* --------------------------------------------------------------------------*/
  /** 
   * @brief  Checks for equality between a target row in this table and a source 
   * row in another table.
   * 
   * @param rhs The other table whose row is compared to this tables
   * @param this_row_index The row index of this table to compare
   * @param rhs_row_index The row index of the rhs table to compare
   * 
   * @returns True if the elements in both rows are equivalent, otherwise False
   */
  /* ----------------------------------------------------------------------------*/
  __device__
  bool rows_equal(gdf_table const & rhs, 
                  const size_type this_row_index, 
                  const size_type rhs_row_index) const
  {

    // If either row contains a NULL, then by definition, because NULL != x for all x,
    // the two rows are not equal
    bool const valid = this->is_row_valid(this_row_index) && rhs.is_row_valid(rhs_row_index);
    if (false == valid) 
    {
      return false;
    }

    for(size_type i = 0; i < num_columns; ++i)
    {
      gdf_dtype const this_col_type = d_columns_types_ptr[i];
      gdf_dtype const rhs_col_type = rhs.d_columns_types_ptr[i];
    
      if(this_col_type != rhs_col_type)
      {
        return false;
      }

      bool is_equal = cudf::type_dispatcher(this_col_type, 
                                            elements_are_equal{}, 
                                            d_columns_data_ptr[i], 
                                            this_row_index, 
                                            rhs.d_columns_data_ptr[i], 
                                            rhs_row_index);

      // If the elements in column `i` do not match, return false
      // Otherwise, continue to column i+1
      if(false == is_equal){
        return false;
      }
    }

    // If we get through all the columns without returning false,
    // then the rows are equivalent
    return true;
  }

  template < template <typename> typename hash_function >
  struct hash_element
  {
    template <typename col_type>
    __device__ __forceinline__
    void operator()(hash_value_type& hash_value, 
                    void const * col_data,
                    size_type row_index,
                    size_type col_index,
                    bool use_initial_value = false,
                    const hash_value_type& initial_value = 0)
    {
      hash_function<col_type> hasher;
      col_type const * const current_column{static_cast<col_type const*>(col_data)};
      hash_value_type key_hash{hasher(current_column[row_index])};

      if (use_initial_value)
        key_hash = hasher.hash_combine(initial_value, key_hash);

      // Only combine hash-values after the first column
      if(0 == col_index)
        hash_value = key_hash;
      else
        hash_value = hasher.hash_combine(hash_value, key_hash);
    }
  };

  /** --------------------------------------------------------------------------*
   * @brief Device function to compute a hash value for a given row in the table
   * 
   * @param[in] row_index The row of the table to compute the hash value for
   * @param[in] num_columns_to_hash The number of columns in the row to hash. If 0,
   * hashes all columns
   * @param[in] initial_hash_values Optional initial hash values to combine with each column's hashed values
   * @tparam hash_function The hash function that is used for each element in the row,
   * as well as combine hash values
   * 
   * @return The hash value of the row
   * ----------------------------------------------------------------------------**/
  template <template <typename> class hash_function = default_hash>
  __device__ 
  hash_value_type hash_row(size_type row_index,
                           hash_value_type* initial_hash_values = nullptr,
                           size_type num_columns_to_hash = 0) const
  {
    hash_value_type hash_value{0};

    // If num_columns_to_hash is zero, hash all columns
    if(0 == num_columns_to_hash) 
    {
      num_columns_to_hash = this->num_columns;
    }

    bool const use_initial_value{ initial_hash_values != nullptr };
    // Iterate all the columns and hash each element, combining the hash values together
    for(size_type i = 0; i < num_columns_to_hash; ++i)
    {
      gdf_dtype const current_column_type = d_columns_types_ptr[i];

      hash_value_type const initial_hash_value = (use_initial_value) ? initial_hash_values[i] : 0;
      cudf::type_dispatcher(current_column_type, 
                          hash_element<hash_function>{}, 
                          hash_value, d_columns_data_ptr[i], row_index, i,
                          use_initial_value, initial_hash_value);
    }

    return hash_value;
  }

private:

  const size_type num_columns; /** The number of columns in the table */
  size_type column_length{0};     /** The number of rows in the table */

  gdf_column ** host_columns{nullptr};  /** The set of gdf_columns that this table wraps */

  rmm::device_vector<void*> device_columns_data; /** Device array of pointers to each columns data */
  void ** d_columns_data_ptr{nullptr};                  /** Raw pointer to the device array's data */

  rmm::device_vector<gdf_valid_type*> device_columns_valids;  /** Device array of pointers to each columns validity bitmask*/
  gdf_valid_type** d_columns_valids_ptr{nullptr};                   /** Raw pointer to the device array's data */

  rmm::device_vector<gdf_valid_type> device_row_valid;  /** Device array of bitmask for the validity of each row. */
  gdf_valid_type * d_row_valid{nullptr};                   /** Raw pointer to device array's data */

  rmm::device_vector<gdf_dtype> device_columns_types; /** Device array of each columns data type */
  gdf_dtype * d_columns_types_ptr{nullptr};                 /** Raw pointer to the device array's data */

  size_type row_size_bytes{0};
  rmm::device_vector<byte_type> device_column_byte_widths;
  byte_type * d_columns_byte_widths_ptr{nullptr};

};

#endif
