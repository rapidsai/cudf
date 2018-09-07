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
  void get_dense_row(size_type row_index, byte_type * row_byte_buffer)
  {
    if(nullptr == row_byte_buffer)
    {
      printf("The buffer to store the row must be preallocated!\n");
      return;
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
            printf("Illegal column byte width.\n");
            return;
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
  void copy_row(gdf_table const & other,
                const size_type my_row_index,
                const size_type other_row_index)
  {

    for(size_type i = 0; i < num_columns; ++i)
    {
      const gdf_dtype my_col_type = d_columns_types[i];
      const gdf_dtype other_col_type = other.d_columns_types[i];
    
      if(my_col_type != other_col_type)
      {
        printf("Attempted to copy columns of different types.\n");
        return;
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
          printf("Attempted to copy column of unsupported GDF datatype\n");
          return;
      }
    }

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
        printf("Attempted to compare columns of different types.\n");
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
          printf("Attempted to compare columns of unsupported GDF datatype\n");
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

  void sort(void) {
      cudaStream_t stream = NULL;
      LesserRTTI<size_type> comparator(d_columns_data,
              reinterpret_cast<int*>(d_columns_types),
              num_columns);
      thrust::device_vector<size_type> indices(column_length);
      thrust::sequence(thrust::cuda::par.on(stream),
              indices.begin(), indices.end());
      thrust::sort(thrust::cuda::par.on(stream),
              indices.begin(), indices.end(),
              [comparator] __host__ __device__ (size_type i1, size_type i2) {
              return comparator.less(i1, i2);
              });
  }

private:

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
