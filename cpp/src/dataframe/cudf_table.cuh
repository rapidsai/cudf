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
#include "utilities/error_utils.h"
#include "hash/hash_functions.cuh"
#include "hash/managed.cuh"
#include "sqls/sqls_rtti_comp.h"

#include <thrust/tabulate.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <cassert>
#include "utilities/type_dispatcher.hpp"



template <typename size_type>
struct ValidRange {
    size_type start, stop;
    __host__ __device__
    ValidRange(
            const size_type begin,
            const size_type end) :
        start(begin), stop(end) {}

    __host__ __device__ __forceinline__
    bool operator()(const size_type index)
    {
        return ((index >= start) && (index < stop));
    }
};



/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Computes the validity mask for the rows in the gdf_table.

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
   * @Synopsis Computes the bit-wise AND across all columns for the specified mask
   * 
   * @Param mask_number The index of the mask to compute the bit-wise AND across all columns
   * 
   * @Returns The bit-wise AND across all columns for the specified mask number
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
 * @Synopsis  Scatters a validity bitmask.
 * 
 * This kernel is used in order to scatter the validity bit mask for a gdf_column.
 * 
 * @Param input_mask The mask that will be scattered.
 * @Param output_mask The output after scattering the input
 * @Param scatter_map The map that indicates where elements from the input
   will be scattered to in the output. output_bit[ scatter_map [i] ] = input_bit[i]
 * @Param num_rows The number of bits in the masks
 */
/* ----------------------------------------------------------------------------*/
template <typename size_type>
__global__ 
void scatter_valid_mask( gdf_valid_type const * const input_mask,
                         gdf_valid_type * const output_mask,
                         size_type const * const __restrict__ scatter_map,
                         size_type const num_rows)
{
  using mask_type = uint32_t;
  constexpr uint32_t BITS_PER_MASK = 8 * sizeof(mask_type);

  // Cast the validity type to a type where atomicOr is natively supported
  const mask_type * __restrict__ input_mask32 = reinterpret_cast<mask_type const *>(input_mask);
  mask_type * const __restrict__ output_mask32 = reinterpret_cast<mask_type * >(output_mask);

  size_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  while(row_number < num_rows)
  {
    // Get the bit corresponding to the row
    const mask_type input_bit = input_mask32[row_number/BITS_PER_MASK] & (static_cast<mask_type>(1) << (row_number % BITS_PER_MASK));

    // Only scatter the input bit if it is valid
    if(input_bit > 0)
    {
      const size_type output_row = scatter_map[row_number];
      // Set the according output bit
      const mask_type output_bit = static_cast<mask_type>(1) << (output_row % BITS_PER_MASK);

      // Find the mask in the output that will hold the bit for the scattered row
      const size_type output_location = output_row / BITS_PER_MASK;

      // Bitwise OR to set the scattered row's bit
      atomicOr(&output_mask32[output_location], output_bit);
    }

    row_number += blockDim.x * gridDim.x;
  }
}

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Gathers a validity bitmask.
 * 
 * This kernel is used in order to gather the validity bit mask for a gdf_column.
 * 
 * @Param input_mask The mask that will be gathered.
 * @Param output_mask The output after gathering the input
 * @Param gather_map The map that indicates where elements from the input
   will be gathered to in the output. output_bit[ gather_map [i] ] = input_bit[i]
 * @Param num_rows The number of bits expected in the output masks
 * @Param input_mask_length The number of bits in the input mask
 */
/* ----------------------------------------------------------------------------*/
template <typename index_type>
__global__ 
void gather_valid_mask( gdf_valid_type const * const input_mask,
                        gdf_valid_type * const output_mask,
                        index_type const * const __restrict__ gather_map,
                        index_type const num_rows,
                        index_type const input_mask_length)
{
  using mask_type = uint32_t;
  constexpr uint32_t BITS_PER_MASK = 8 * sizeof(mask_type);

  // Cast the validity type to a type where atomicOr is natively supported
  const mask_type * __restrict__ input_mask32 = reinterpret_cast<mask_type const *>(input_mask);
  mask_type * const __restrict__ output_mask32 = reinterpret_cast<mask_type * >(output_mask);

  index_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  ValidRange<index_type> valid(0, input_mask_length);
  while(row_number < num_rows)
  {
    const index_type gather_location = gather_map[row_number];
    if (!valid(gather_location)) {
        row_number += blockDim.x * gridDim.x; continue;
    }

    // Get the bit corresponding from the gathered row
    // FIXME Replace with a standard `get_bit` function
    mask_type input_bit = (static_cast<mask_type>(1) << (gather_location % BITS_PER_MASK));
    if (nullptr != input_mask) {
        input_bit = input_bit & input_mask32[gather_location/BITS_PER_MASK];
    }

    // Only set the output bit if the input is valid
    if(input_bit > 0)
    {
      // FIXME Replace with a standard `set_bit` function
      // Construct the mask that sets the bit for the output row
      const mask_type output_bit = static_cast<mask_type>(1) << (row_number % BITS_PER_MASK);

      // Find the mask in the output that will hold the bit for output row
      const index_type output_location = row_number / BITS_PER_MASK;

      // Bitwise OR to set the gathered row's bit
      atomicOr(&output_mask32[output_location], output_bit);
    }

    row_number += blockDim.x * gridDim.x;
  }
}

//Wrapper around gather_valid_mask
template <typename index_type>
void gather_valid( gdf_valid_type const * const input_mask,
                   gdf_valid_type * const output_mask,
                   index_type const * const __restrict__ gather_map,
                   index_type const num_rows,
                   index_type const input_mask_length,
                   cudaStream_t stream = 0) {
    const index_type BLOCK_SIZE = 256;
    const index_type gather_grid_size = (num_rows + BLOCK_SIZE - 1)/BLOCK_SIZE;
    gather_valid_mask<<<gather_grid_size, BLOCK_SIZE, 0, stream>>>(
            input_mask, output_mask, gather_map, num_rows, input_mask_length);
}

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis A class provides useful functionality for operating on a set of gdf_columns. 

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
    const size_type mask_size = gdf_get_num_chars_bitmask(column_length);
    device_row_valid.resize(mask_size);

    // If a row contains a single NULL value, then the entire row is considered
    // to be NULL, therefore initialize the row-validity mask with the
    // bit-wise AND of the validity mask of all the columns
    thrust::tabulate(rmm::exec_policy(cudaStream_t{0}),
                     device_row_valid.begin(), device_row_valid.end(),
                     row_masker<size_type>(d_columns_valids_ptr, num_cols));

    d_row_valid = device_row_valid.data().get();
  }

  ~gdf_table(){}


  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis  Updates the length of the gdf_columns in the table
   * 
   * @Param new_length The new length
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
   * @Synopsis  Gets the size in bytes of a row in the gdf_table, i.e., the sum of 
   * the byte widths of all columns in the table
   * 
   * @Returns The size in bytes of the row in the table
   */
  /* ----------------------------------------------------------------------------*/
  byte_type get_row_size_bytes() const
  {
    return row_size_bytes;
  }


  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis  Packs the elements of a specified row into a contiguous byte-buffer
   *
   * This function is called by a single thread, and the thread will copy each element
   * of the row into a single contiguous buffer. TODO: This could be done by multiple threads
   * by passing in a cooperative group. 
   * 
   * @Param index The row of the table to return
   * @Param row_byte_buffer A pointer to a preallocated buffer large enough to hold a 
      row of the table 
   * 
   */
  /* ----------------------------------------------------------------------------*/
  // TODO Is there a less hacky way to do this? 
  __device__
  gdf_error get_packed_row_values(size_type row_index, byte_type * row_byte_buffer)
  {
    if(nullptr == row_byte_buffer) {
      return GDF_DATASET_EMPTY;
    }

    byte_type * write_pointer{row_byte_buffer};

    // Pack the element from each column in the row into the buffer
    for(size_type i = 0; i < num_columns; ++i)
    {
      const byte_type current_column_byte_width = d_columns_byte_widths_ptr[i];
      switch(current_column_byte_width)
      {
        case 1:
          {
            using col_type = int8_t;
            const col_type * const current_row_element = static_cast<col_type *>(d_columns_data_ptr[i])[row_index];
            col_type * write_location = static_cast<col_type*>(write_pointer);
            *write_location = *current_row_element;
            write_pointer += sizeof(col_type);
            break;
          }
        case 2:
          {
            using col_type = int16_t;
            const col_type * const current_row_element = static_cast<col_type *>(d_columns_data_ptr[i])[row_index];
            col_type * write_location = static_cast<col_type*>(write_pointer);
            *write_location = *current_row_element;
            write_pointer += sizeof(col_type);
            break;
          }
        case 4:
          {
            using col_type = int32_t;
            const col_type * const current_row_element = static_cast<col_type *>(d_columns_data_ptr[i])[row_index];
            col_type * write_location = static_cast<col_type*>(write_pointer);
            *write_location = *current_row_element;
            write_pointer += sizeof(col_type);
            break;
          }
        case 8:
          {
            using col_type = int64_t;
            const col_type * const current_row_element = static_cast<col_type *>(d_columns_data_ptr[i])[row_index];
            col_type * write_location = static_cast<col_type*>(write_pointer);
            *write_location = *current_row_element;
            write_pointer += sizeof(col_type);
            break;
          }
        default:
          {
            return GDF_UNSUPPORTED_DTYPE;
          }
      }
    }
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
     * @Synopsis  Copies a row from a source table to a target row in this table
     *  
     * This device function should be called by a single thread and the thread will copy all of 
     * the elements in the row from one table to the other. TODO: In the future, this could be done
     * by multiple threads by passing in a cooperative group.
     * 
     * @Param other The other table from which the row is copied
     * @Param my_row_index The index of the row in this table that will be written to
     * @Param other_row_index The index of the row from the other table that will be copied from
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
   * @Synopsis  Checks for equality between a target row in this table and a source 
   * row in another table.
   * 
   * @Param rhs The other table whose row is compared to this tables
   * @Param this_row_index The row index of this table to compare
   * @Param rhs_row_index The row index of the rhs table to compare
   * 
   * @Returns True if the elements in both rows are equivalent, otherwise False
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
                    size_type col_index)
    {
      hash_function<col_type> hasher;
      col_type const * const current_column{static_cast<col_type const*>(col_data)};
      hash_value_type const key_hash{hasher(current_column[row_index])};

      // Only combine hash-values after the first column
      if(0 == col_index)
        hash_value = key_hash;
      else
        hash_value = hasher.hash_combine(hash_value,key_hash);
    }
  };

  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis  Device function to compute a hash value for a given row in the table
   * 
   * @Param row_index The row of the table to compute the hash value for
   * @Param num_columns_to_hash The number of columns in the row to hash. If 0, 
   * hashes all columns
   * @tparam hash_function The hash function that is used for each element in the row,
   * as well as combine hash values
   * 
   * @Returns The hash value of the row
   */
  /* ----------------------------------------------------------------------------*/
  template <template <typename> class hash_function = default_hash>
  __device__ 
  hash_value_type hash_row(size_type row_index, size_type num_columns_to_hash = 0) const
  {
    hash_value_type hash_value{0};

    // If num_columns_to_hash is zero, hash all columns
    if(0 == num_columns_to_hash) 
    {
      num_columns_to_hash = this->num_columns;
    }

    // Iterate all the columns and hash each element, combining the hash values together
    for(size_type i = 0; i < num_columns_to_hash; ++i)
    {
      gdf_dtype const current_column_type = d_columns_types_ptr[i];

      cudf::type_dispatcher(current_column_type, 
                          hash_element<hash_function>{}, 
                          hash_value, d_columns_data_ptr[i], row_index, i);
    }

    return hash_value;
  }


  /* --------------------------------------------------------------------------*/
  /** 
   * @brief  Creates a rearrangement of the table from another table by gathering
     the rows of the input table to rows of this table based on a gather map that
     maps every row of the input table to a corresponding row in this table.
   * 
   * @Param[in] row_gather_map The mapping from input row locations to output row
     locations, i.e., Row 'row_gather_map[i]' of this table will be gathered to 
     gather_output_table[i]
   * @Param[in] gather_output_table The output table to which the rows of this table
     will be mapped
   * @Param[in] range_check Flag to check if row_gather_map has valid values
   * 
   * @Returns   
   */
  /* ----------------------------------------------------------------------------*/
  template <typename index_type>
  gdf_error gather(index_type const * const row_gather_map,
                   gdf_table<size_type> & gather_output_table,
                   bool range_check = false)
  {
    gdf_error gdf_status{GDF_SUCCESS};
  
    // Each column can be gathered in parallel, therefore create a 
    // separate stream for every column
    std::vector<cudaStream_t> column_streams(num_columns);
    for(auto & s : column_streams)
    {
      cudaStreamCreate(&s);
    }
  
    // Scatter columns one by one
    for(size_type i = 0; i < num_columns; ++i)
    {
      gdf_column * const current_input_column = this->get_column(i);
      gdf_column * const current_output_column = gather_output_table.get_column(i);
      int column_width_bytes{0};
      gdf_status = get_column_byte_width(current_input_column, &column_width_bytes);
  
      if(GDF_SUCCESS != gdf_status)
        return gdf_status;
  
      // Scatter each column based on it's byte width
      switch(column_width_bytes)
      {
        case 1:
          {
            using column_type = int8_t;
            gdf_status = gather_column<column_type, index_type>(
                                                     current_input_column, 
                                                     row_gather_map, 
                                                     current_output_column,
                                                     range_check,
                                                     column_streams[i]);
            break;
          }
        case 2:
          {
            using column_type = int16_t;
            gdf_status = gather_column<column_type, index_type>(
                                                     current_input_column, 
                                                     row_gather_map, 
                                                     current_output_column,
                                                     range_check,
                                                     column_streams[i]);
            break;
          }
        case 4:
          {
            using column_type = int32_t;
            gdf_status = gather_column<column_type, index_type>(
                                                     current_input_column, 
                                                     row_gather_map, 
                                                     current_output_column,
                                                     range_check,
                                                     column_streams[i]);
            break;
          }
        case 8:
          {
            using column_type = int64_t;
            gdf_status = gather_column<column_type, index_type>(
                                                     current_input_column, 
                                                     row_gather_map, 
                                                     current_output_column,
                                                     range_check,
                                                     column_streams[i]);
            break;
          }
        default:
          gdf_status = GDF_UNSUPPORTED_DTYPE;
      }
  
      if(GDF_SUCCESS != gdf_status)
        return gdf_status;
    }
  
    // Synchronize all the streams
    CUDA_TRY( cudaDeviceSynchronize() );
  
    // Destroy all streams
    for(auto & s : column_streams)
    {
      cudaStreamDestroy(s);
    }
  
    return gdf_status;
  }

  template <typename index_type>
  gdf_error gather(rmm::device_vector<index_type> const & row_gather_map,
          gdf_table<size_type> & gather_output_table, bool range_check = false)
  {
      return gather(row_gather_map.data().get(), gather_output_table, range_check);
  }

  template <typename index_type>
  gdf_error gather(gdf_column * row_gather_map,
          gdf_table<size_type> & gather_output_table, bool range_check = false)
  {
      auto ptr = static_cast<index_type*>(row_gather_map->data);
      return gather(ptr, gather_output_table, range_check);
  }

  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis  An in-place gather operation that permutes the rows of the table
   * according to a map. permuted_table[i] = original_table[ row_gather_map[i] ]
   * 
   * @Param row_gather_map The map the determines the reordering of rows in the 
   table 
   * 
   * @Param range_check Flag to check if row_gather_map has valid values
   table 
   * 
   * @Returns   
   */
  /* ----------------------------------------------------------------------------*/
  template <typename size_type>
  gdf_error gather(size_type const * const row_gather_map,
          bool range_check = false) {
      return gather(row_gather_map, *this, range_check);
  }

  template <typename size_type>
  gdf_error gather(rmm::device_vector<size_type> const & row_gather_map,
          bool range_check = false) {
      return gather(row_gather_map, *this, range_check);
  }

  template <typename size_type>
  gdf_error gather(gdf_column * row_gather_map,
          bool range_check = false) {
      return gather(row_gather_map, *this, range_check);
  }

  
  
/* --------------------------------------------------------------------------*/
/** 
 * @brief  Creates a rearrangement of the table into another table by scattering
   the rows of this table to rows of the output table based on a scatter map that
   maps every row of this table to a corresponding row in the output table.
 * 
 * @Param[out] scattered_output_table The rearrangement of the input table based 
   on the mappings from the row_scatter_map array
 * @Param[in] row_scatter_map The mapping from input row locations to output row
   locations, i.e., Row 'i' of this table will be scattered to 
   scattered_output_table[row_scatter_map[i]]
 * 
 * @Returns   
 */
/* ----------------------------------------------------------------------------*/
template <typename size_type>
gdf_error scatter( gdf_table<size_type> & scattered_output_table,
                   size_type const * const row_scatter_map) const
{
  gdf_error gdf_status{GDF_SUCCESS};

  // Each column can be scattered in parallel, therefore create a 
  // separate stream for every column
  std::vector<cudaStream_t> column_streams(num_columns);
  for(auto & s : column_streams){
    cudaStreamCreate(&s);
  }

  // Each columns validity bit mask can be scattered in parallel,
  // therefore create a separate stream for each column
  std::vector<cudaStream_t> valid_streams(num_columns);
  for(auto & s : valid_streams){
    cudaStreamCreate(&s);
  }

  // Scatter columns one by one
  for(size_type i = 0; i < num_columns; ++i)
  {
    gdf_column * const current_input_column = this->get_column(i);
    gdf_column * const current_output_column = scattered_output_table.get_column(i);
    int column_width_bytes{0};
    gdf_status = get_column_byte_width(current_input_column, &column_width_bytes);

    if(GDF_SUCCESS != gdf_status)
      return gdf_status;

    // If this column has a validity mask, scatter the mask
    if((nullptr != current_input_column->valid)
        && (nullptr != current_output_column->valid))
    {
      // Ensure the output bitmask is initialized to zero
      const size_type num_masks = gdf_get_num_chars_bitmask(column_length);
      cudaMemsetAsync(current_output_column->valid, 0,  num_masks * sizeof(gdf_valid_type), valid_streams[i]);

      // Scatter the validity bits from the input column to output column
      constexpr int BLOCK_SIZE = 256;
      const int grid_size = (column_length + BLOCK_SIZE - 1)/BLOCK_SIZE;
      scatter_valid_mask<<<grid_size, BLOCK_SIZE, 0, valid_streams[i]>>>(current_input_column->valid, 
                                                                         current_output_column->valid,
                                                                         row_scatter_map,
                                                                         column_length);
    }

    // Scatter each column based on it's byte width
    switch(column_width_bytes)
    {
      case 1:
        {
          using column_type = int8_t;
          column_type * input = static_cast<column_type*>(current_input_column->data);
          column_type * output = static_cast<column_type*>(current_output_column->data);
          gdf_status = scatter_column<column_type>(input, 
                                                   column_length,
                                                   row_scatter_map, 
                                                   output,
                                                   column_streams[i]);
          break;
        }
      case 2:
        {
          using column_type = int16_t;
          column_type * input = static_cast<column_type*>(current_input_column->data);
          column_type * output = static_cast<column_type*>(current_output_column->data);
          gdf_status = scatter_column<column_type>(input, 
                                                   column_length,
                                                   row_scatter_map, 
                                                   output,
                                                   column_streams[i]);
          break;
        }
      case 4:
        {
          using column_type = int32_t;
          column_type * input = static_cast<column_type*>(current_input_column->data);
          column_type * output = static_cast<column_type*>(current_output_column->data);
          gdf_status = scatter_column<column_type>(input, 
                                                   column_length,
                                                   row_scatter_map, 
                                                   output,
                                                   column_streams[i]);
          break;
        }
      case 8:
        {
          using column_type = int64_t;
          column_type * input = static_cast<column_type*>(current_input_column->data);
          column_type * output = static_cast<column_type*>(current_output_column->data);
          gdf_status = scatter_column<column_type>(input, 
                                                   column_length,
                                                   row_scatter_map, 
                                                   output,
                                                   column_streams[i]);
          break;
        }
      default:
        gdf_status = GDF_UNSUPPORTED_DTYPE;
    }

    if(GDF_SUCCESS != gdf_status)
      return gdf_status;
  }

  // Synchronize all the streams
  CUDA_TRY( cudaDeviceSynchronize() );

  // Destroy all streams
  for(auto & s : column_streams)
  {
    cudaStreamDestroy(s);
  }
  // Destroy all streams
  for(auto & s : valid_streams)
  {
    cudaStreamDestroy(s);
  }

  return gdf_status;
}


private:
/* --------------------------------------------------------------------------*/
  /** 
   * @brief Gathers the values of a column into a new column based on a map that
     maps rows in the input column to rows in the output column.
     input_column[row_gather_map[i]] will be assigned to output_column[i]
   * 
   * @Param[in] input_column The input column whose rows will be gathered
   * @Param[in] num_rows The number of rows in the input and output columns
   * @Param[in] row_gather_map An array that maps rows in the input column
     to rows in the output column
   * @Param[out] output_column The rearrangement of the input column 
     based on the mapping determined by the row_gather_map array
   * @Param[in] range_check Flag to check validity of the values in row_gather_map
   * 
   * @Returns GDF_SUCCESS upon successful computation
   */
  /* ----------------------------------------------------------------------------*/
  template <typename column_type,
            typename index_type>
  gdf_error gather_column(gdf_column * input_column,
                          index_type const * const row_gather_map,
                          gdf_column * output_column,
                          const bool range_check,
                          cudaStream_t stream = 0) const
  {
    column_type * const i_data = static_cast<column_type*>(input_column->data);
    column_type * const o_data = static_cast<column_type*>(output_column->data);
    index_type num_rows = output_column->size;
  
    gdf_error gdf_status{GDF_SUCCESS};


    // Gathering from one table to another
    if (i_data != o_data) {
        if (range_check) {
            thrust::gather_if(rmm::exec_policy(stream),
                    row_gather_map,
                    row_gather_map + num_rows,
                    row_gather_map,
                    i_data,
                    o_data,
                    ValidRange<index_type>(0, input_column->size));

        } else {
            thrust::gather(rmm::exec_policy(stream),
                    row_gather_map,
                    row_gather_map + num_rows,
                    i_data,
                    o_data);
        }
    } 
    // Gather is in-place
    else {
        rmm::device_vector<column_type> remapped_copy(num_rows);
        if (range_check) {
            thrust::gather_if(rmm::exec_policy(stream),
                           row_gather_map,
                           row_gather_map + num_rows,
                           row_gather_map,
                           i_data,
                           remapped_copy.begin(),
                           ValidRange<index_type>(0, input_column->size));
        } else {
            thrust::gather(rmm::exec_policy(stream),
                           row_gather_map,
                           row_gather_map + num_rows,
                           i_data,
                           remapped_copy.begin());
        }
        thrust::copy(rmm::exec_policy(stream),
                remapped_copy.begin(),
                remapped_copy.end(),
                o_data);
    }
    //If gather is in-place
    if ((input_column->valid == output_column->valid) &&
            (input_column->valid != nullptr)) {
        rmm::device_vector<gdf_valid_type> remapped_valid_copy(gdf_get_num_chars_bitmask(num_rows));
        gather_valid<index_type>(
                input_column->valid,
                remapped_valid_copy.data().get(),
                row_gather_map, 
                num_rows, input_column->size, stream);
        thrust::copy(rmm::exec_policy(stream),
                remapped_valid_copy.begin(),
                remapped_valid_copy.end(), output_column->valid);
    }
    //If both input and output columns have a non null valid pointer
    else if (nullptr != output_column->valid) {
        gather_valid<index_type>(
                input_column->valid,
                output_column->valid,
                row_gather_map, 
                num_rows, input_column->size, stream);
    }
  
    CUDA_CHECK_LAST();
  
    return gdf_status;
  }


/* --------------------------------------------------------------------------*/
/** 
 * @brief Scatters the values of a column into a new column based on a map that
   maps rows in the input column to rows in the output column. input_column[i]
   will be scattered to output_column[ row_scatter_map[i] ]
 * 
 * @Param[in] input_column The input column whose rows will be scattered
 * @Param[in] num_rows The number of rows in the input and output columns
 * @Param[in] row_scatter_map An array that maps rows in the input column
   to rows in the output column
 * @Param[out] output_column The rearrangement of the input column 
   based on the mapping determined by the row_scatter_map array
 * 
 * @Returns GDF_SUCCESS upon successful computation
 */
/* ----------------------------------------------------------------------------*/
template <typename column_type,
          typename size_type>
gdf_error scatter_column(column_type const * const __restrict__ input_column,
                         size_type const num_rows,
                         size_type const * const __restrict__ row_scatter_map,
                         column_type * const __restrict__ output_column,
                         cudaStream_t stream = 0) const
{

  gdf_error gdf_status{GDF_SUCCESS};


  thrust::scatter(rmm::exec_policy(stream),
                  input_column,
                  input_column + num_rows,
                  row_scatter_map,
                  output_column);

  CUDA_CHECK_LAST();

  return gdf_status;
}


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
