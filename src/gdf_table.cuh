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

#include <gdf/gdf.h>
#include <thrust/device_vector.h>
#include <cassert>
#include "hashmap/hash_functions.cuh"
#include "hashmap/managed.cuh"
#include "sqls_rtti_comp.hpp"

// TODO Inherit from managed class to allocate with managed memory?
template <typename T, typename byte_t = unsigned char>
class gdf_table : public managed
{
public:

  using size_type = T;
  using byte_type = byte_t;

  gdf_table(size_type num_cols, gdf_column ** gdf_columns) 
    : num_columns(num_cols), host_columns(gdf_columns)
  {

    column_length = host_columns[0]->size;

    // Copy the pointers to the column's data and types to the device 
    // as contiguous arrays
    device_columns.reserve(num_cols);
    device_types.reserve(num_cols);
    column_byte_widths.reserve(num_cols);
    for(size_type i = 0; i < num_cols; ++i)
    {
      gdf_column * const current_column = host_columns[i];
      assert(column_length == current_column->size);

      device_columns.push_back(current_column->data);
      device_types.push_back(current_column->dtype);

      // Compute the size of a row in the table in bytes
      int column_width_bytes{0};
      if(GDF_SUCCESS == get_column_byte_width(current_column, &column_width_bytes))
      {
        row_size_bytes += column_width_bytes;

        // Store the byte width of each column in a device array
        column_byte_widths.push_back(static_cast<byte_type>(row_size_bytes));
      }
      else
      {
        std::cerr << "Attempted to get column byte width of unsupported GDF datatype.\n";
        column_byte_widths.push_back(0);
      }
    }

    d_columns_data = device_columns.data().get();
    d_columns_types = device_types.data().get();
    d_column_byte_widths = column_byte_widths.data().get();
  }

  ~gdf_table(){}

  __host__ 
  gdf_column * get_column(size_type column_index) const
  {
    return host_columns[column_index];
  }

  __host__ __device__
  size_type get_column_length() const
  {
    return column_length;
  }

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


  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis  Gets the GDF data type of the column to be used for building the hash table
   * 
   * @Returns The GDF data type of the build column
   */
  /* ----------------------------------------------------------------------------*/
  gdf_dtype get_build_column_type() const
  {
    return host_columns[build_column_index]->dtype;
  }

  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis  Gets a pointer to the data of the column to be used for building the hash table
   * 
   * @Returns Pointer to data of the build column
   */
  /* ----------------------------------------------------------------------------*/
  void * get_build_column_data() const
  {
    return host_columns[build_column_index]->data;
  }


  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis  Gets a pointer to the data of the column to be used for probing the hash table
   * 
   * @Returns  Pointer to data of the probe column
   */
  /* ----------------------------------------------------------------------------*/
  void * get_probe_column_data() const
  {
    return host_columns[probe_column_index]->data;
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
   * @Synopsis  Packs the elements of a specified row into a contiguous, dense buffer
   * 
   * @Param index The row of the table to return
   * @Param row_byte_buffer A pointer to a preallocated buffer large enough to hold a 
      row of the table 
   * 
   */
  /* ----------------------------------------------------------------------------*/
  // TODO Is there a less hacky way to do this? 
  __device__
  gdf_error get_dense_row(size_type row_index, byte_type * row_byte_buffer)
  {
    if(nullptr == row_byte_buffer) {
      return GDF_DATASET_EMPTY;
    }

    byte_type * write_pointer{row_byte_buffer};

    // Pack the element from each column in the row into the buffer
    for(size_type i = 0; i < num_columns; ++i)
    {
      const byte_type current_column_byte_width = d_column_byte_widths[i];
      switch(current_column_byte_width)
      {
        case 1:
          {
            using col_type = int8_t;
            const col_type * const current_row_element = static_cast<col_type *>(d_columns_data[i])[row_index];
            col_type * write_location = static_cast<col_type*>(write_pointer);
            *write_location = *current_row_element;
            write_pointer += sizeof(col_type);
            break;
          }
        case 2:
          {
            using col_type = int16_t;
            const col_type * const current_row_element = static_cast<col_type *>(d_columns_data[i])[row_index];
            col_type * write_location = static_cast<col_type*>(write_pointer);
            *write_location = *current_row_element;
            write_pointer += sizeof(col_type);
            break;
          }
        case 4:
          {
            using col_type = int32_t;
            const col_type * const current_row_element = static_cast<col_type *>(d_columns_data[i])[row_index];
            col_type * write_location = static_cast<col_type*>(write_pointer);
            *write_location = *current_row_element;
            write_pointer += sizeof(col_type);
            break;
          }
        case 8:
          {
            using col_type = int64_t;
            const col_type * const current_row_element = static_cast<col_type *>(d_columns_data[i])[row_index];
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


    /* --------------------------------------------------------------------------*/
    /** 
     * @Synopsis  Copies a row from another table to a row in this table
     * 
     * @Param other The other table from which the row is copied
     * @Param my_row_index The index of the row in this table that will be written to
     * @Param other_row_index The index of the row from the other table that will be copied from
     */
    /* ----------------------------------------------------------------------------*/
  __device__ 
  gdf_error copy_row(gdf_table const & other,
                const size_type my_row_index,
                const size_type other_row_index)
  {

    for(size_type i = 0; i < num_columns; ++i)
    {
      const gdf_dtype my_col_type = d_columns_types[i];
      const gdf_dtype other_col_type = other.d_columns_types[i];
    
      if(my_col_type != other_col_type){
        return GDF_DTYPE_MISMATCH;
      }

      switch(my_col_type)
      {
        case GDF_INT8:
          {
            using col_type = int8_t;
            col_type & my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            my_elem = other_elem;
            break;
          }
        case GDF_INT16:
          {
            using col_type = int16_t;
            col_type & my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            my_elem = other_elem;
            break;
          }
        case GDF_INT32:
          {
            using col_type = int32_t;
            col_type & my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            my_elem = other_elem;
            break;
          }
        case GDF_INT64:
          {
            using col_type = int64_t;
            col_type & my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            my_elem = other_elem;
            break;
          }
        case GDF_FLOAT32:
          {
            using col_type = float;
            col_type & my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            my_elem = other_elem;
            break;
          }
        case GDF_FLOAT64:
          {
            using col_type = double;
            col_type & my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            my_elem = other_elem;
            break;
          }
        case GDF_DATE32:
          {
            using col_type = int32_t;
            col_type & my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            my_elem = other_elem;
            break;
          }
        case GDF_DATE64:
          {
            using col_type = int64_t;
            col_type & my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            my_elem = other_elem;
            break;
          }
        case GDF_TIMESTAMP:
          {
            using col_type = int64_t;
            col_type & my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            my_elem = other_elem;
            break;
          }
        default:
          return GDF_UNSUPPORTED_DTYPE;
      }
    }
    return GDF_SUCCESS;
  }

    /* --------------------------------------------------------------------------*/
    /** 
     * @Synopsis  Checks for equality between a row in this table and another table.
     * 
     * @Param other The other table whose row is compared to this tables
     * @Param my_row_index The row index of this table to compare
     * @Param other_row_index The row index of the other table to compare
     * 
     * @Returns True if the elements in both rows are equivalent, otherwise False
     */
    /* ----------------------------------------------------------------------------*/
  __device__
  bool rows_equal(gdf_table const & other, 
                  const size_type my_row_index, 
                  const size_type other_row_index) const
  {

    for(size_type i = 0; i < num_columns; ++i)
    {
      const gdf_dtype my_col_type = d_columns_types[i];
      const gdf_dtype other_col_type = other.d_columns_types[i];
    
      if(my_col_type != other_col_type)
      {
        return false;
      }

      switch(my_col_type)
      {
        case GDF_INT8:
          {
            using col_type = int8_t;
            const col_type my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            if(my_elem != other_elem)
              return false;
            break;
          }
        case GDF_INT16:
          {
            using col_type = int16_t;
            const col_type my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            if(my_elem != other_elem)
              return false;
            break;
          }
        case GDF_INT32:
          {
            using col_type = int32_t;
            const col_type my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            if(my_elem != other_elem)
              return false;
            break;
          }
        case GDF_INT64:
          {
            using col_type = int64_t;
            const col_type my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            if(my_elem != other_elem)
              return false;
            break;
          }
        case GDF_FLOAT32:
          {
            using col_type = float;
            const col_type my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            if(my_elem != other_elem)
              return false;
            break;
          }
        case GDF_FLOAT64:
          {
            using col_type = double;
            const col_type my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            if(my_elem != other_elem)
              return false;
            break;
          }
        case GDF_DATE32:
          {
            using col_type = int32_t;
            const col_type my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            if(my_elem != other_elem)
              return false;
            break;
          }
        case GDF_DATE64:
          {
            using col_type = int64_t;
            const col_type my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            if(my_elem != other_elem)
              return false;
            break;
          }
        case GDF_TIMESTAMP:
          {
            using col_type = int64_t;
            const col_type my_elem = static_cast<col_type*>(d_columns_data[i])[my_row_index];
            const col_type other_elem = static_cast<col_type*>(other.d_columns_data[i])[other_row_index];
            if(my_elem != other_elem)
              return false;
            break;
          }
        default:
          return false;
      }
    }

    return true;
  }

  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis  This device function computes a hash value for a given row in the table
   * 
   * @Param row_index The row of the table to compute the hash value for
   * @Param num_columns_to_hash The number of columns in the row to hash. If 0, hashes all columns
   * @tparam hash_function The hash function that is used for each element in the row
   * @tparam dummy Used only to be able to resolve the result_type from the hash_function.
                   The actual type of dummy doesn't matter.
   * 
   * @Returns The hash value of the row
   */
  /* ----------------------------------------------------------------------------*/
  template <template <typename> class hash_function = default_hash,
            typename dummy = int>
  __device__ 
  typename hash_function<dummy>::result_type hash_row(size_type row_index, 
                                                      size_type num_columns_to_hash = 0) const
  {
    using hash_value_t = typename hash_function<dummy>::result_type;
    hash_value_t hash_value{0};

    // If num_columns_to_hash is zero, hash all columns
    if(0 == num_columns_to_hash)
      num_columns_to_hash = this->num_columns;

    for(size_type i = 0; i < num_columns_to_hash; ++i)
    {
      const gdf_dtype current_column_type = d_columns_types[i];

      switch(current_column_type)
      {
        case GDF_INT8:
          {
            using col_type = int8_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
            break;
          }
        case GDF_INT16:
          {
            using col_type = int16_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
            break;
          }
        case GDF_INT32:
          {
            using col_type = int32_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
            break;
          }
        case GDF_INT64:
          {
            using col_type = int64_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
            break;
          }
        case GDF_FLOAT32:
          {
            using col_type = float;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
            break;
          }
        case GDF_FLOAT64:
          {
            using col_type = double;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
            break;
          }
        case GDF_DATE32:
          {
            using col_type = int32_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
            break;
          }
        case GDF_DATE64:
          {
            using col_type = int64_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
            break;
          }
        case GDF_TIMESTAMP:
          {
            using col_type = int64_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
            break;
          }
        default:
          assert(false && "Attempted to hash unsupported GDF datatype");
      }
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
   * 
   * @Returns   
   */
  /* ----------------------------------------------------------------------------*/
  template <typename size_type>
  gdf_error gather(thrust::device_vector<size_type> const & row_gather_map,
          gdf_table<size_type> & gather_output_table)
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
      size_type column_width_bytes{0};
      gdf_status = get_column_byte_width(current_input_column, &column_width_bytes);
  
      if(GDF_SUCCESS != gdf_status)
        return gdf_status;
  
      // Scatter each column based on it's byte width
      switch(column_width_bytes)
      {
        case 1:
          {
            using column_type = int8_t;
            column_type * const input = static_cast<column_type*>(current_input_column->data);
            column_type * const output = static_cast<column_type*>(current_output_column->data);
            gdf_status = gather_column<column_type, size_type>(input, 
                                                     column_length,
                                                     row_gather_map, 
                                                     output,
                                                     column_streams[i]);
            break;
          }
        case 2:
          {
            using column_type = int16_t;
            column_type * input = static_cast<column_type*>(current_input_column->data);
            column_type * output = static_cast<column_type*>(current_output_column->data);
            gdf_status = gather_column<column_type, size_type>(input, 
                                                     column_length,
                                                     row_gather_map, 
                                                     output,
                                                     column_streams[i]);
            break;
          }
        case 4:
          {
            using column_type = int32_t;
            column_type * input = static_cast<column_type*>(current_input_column->data);
            column_type * output = static_cast<column_type*>(current_output_column->data);
            gdf_status = gather_column<column_type, size_type>(input, 
                                                     column_length,
                                                     row_gather_map, 
                                                     output,
                                                     column_streams[i]);
            break;
          }
        case 8:
          {
            using column_type = int64_t;
            column_type * input = static_cast<column_type*>(current_input_column->data);
            column_type * output = static_cast<column_type*>(current_output_column->data);
            gdf_status = gather_column<column_type, size_type>(input, 
                                                     column_length,
                                                     row_gather_map, 
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
  
    return gdf_status;
  }

  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis  An in-place gather operation that permutes the rows of the table
   * according to a map. permuted_table[i] = original_table[ row_gather_map[i] ]
   * 
   * @Param row_gather_map The map the determines the reordering of rows in the 
   table 
   * 
   * @Returns   
   */
  /* ----------------------------------------------------------------------------*/
  template <typename size_type>
  gdf_error gather(thrust::device_vector<size_type> const & row_gather_map) {
      return gather(row_gather_map, *this);
  }

  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis  Lexicographically sorts the rows of the gdf_table in-place
   * 
   * @Returns A permutation vector of the new ordering of the rows, e.g.,
   * sorted_table[i] == unsorted_table[ permuted_indices[i] ]
   */
  /* ----------------------------------------------------------------------------*/
  thrust::device_vector<size_type> sort(void) {

      cudaStream_t stream = NULL;

      // Functor that defines a `less` operator between rows of a set of
      // gdf_columns
      LesserRTTI<size_type> comparator(d_columns_data,
              reinterpret_cast<int*>(d_columns_types),
              num_columns);

      // Vector that will store the permutation of the rows after the sort
      thrust::device_vector<size_type> permuted_indices(column_length);
      thrust::sequence(thrust::cuda::par.on(stream),
              permuted_indices.begin(), permuted_indices.end());

      // Use the LesserRTTI functor to sort the rows of the table and the
      // permutation vector
      thrust::sort(thrust::cuda::par.on(stream),
              permuted_indices.begin(), permuted_indices.end(),
              [comparator] __host__ __device__ (size_type i1, size_type i2) {
              return comparator.less(i1, i2);
              });

      //thrust::host_vector<void*> host_columns = device_columns;
      //thrust::host_vector<gdf_dtype> host_types = device_types;

      gather(permuted_indices);

      return permuted_indices;
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
   * 
   * @Returns GDF_SUCCESS upon successful computation
   */
  /* ----------------------------------------------------------------------------*/
  template <typename column_type,
            typename size_type>
  gdf_error gather_column(column_type * const __restrict__ input_column,
                           size_type const num_rows,
                           thrust::device_vector<size_type> const & row_gather_map,
                           column_type * const __restrict__ output_column,
                           cudaStream_t stream = 0) const
  {
  
    gdf_error gdf_status{GDF_SUCCESS};

    // Gathering from one table to another
    if (input_column != output_column) {
      thrust::gather(thrust::cuda::par.on(stream),
                     row_gather_map.begin(),
                     row_gather_map.end(),
                     input_column,
                     output_column);
    } 
    // Gather is in-place
    else {
        thrust::device_vector<column_type> remapped_copy(num_rows);
        thrust::gather(thrust::cuda::par.on(stream),
                       row_gather_map.begin(),
                       row_gather_map.end(),
                       input_column,
                       remapped_copy.begin());
        thrust::copy(thrust::cuda::par.on(stream),
                remapped_copy.begin(),
                remapped_copy.end(),
                output_column);
    }
  
    CUDA_CHECK_LAST();
  
    return gdf_status;
  }

  void ** d_columns_data{nullptr};
  gdf_dtype * d_columns_types{nullptr};

  thrust::device_vector<void*> device_columns;
  thrust::device_vector<gdf_dtype> device_types;

  gdf_column ** host_columns{nullptr};
  const size_type num_columns;
  size_type column_length{0};

  size_type row_size_bytes{0};
  thrust::device_vector<byte_type> column_byte_widths;
  byte_type * d_column_byte_widths{nullptr};

  // Just use the first column as the build/probe column for now
  const size_type build_column_index{0};
  const size_type probe_column_index{0};

};

#endif
