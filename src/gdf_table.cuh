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

// TODO Inherit from managed class to allocate with managed memory?
template <typename T>
class gdf_table : public managed
{
public:

  using size_type = T;

  gdf_table(size_type num_cols, gdf_column ** gdf_columns) 
    : num_columns(num_cols), host_columns(gdf_columns)
  {

    column_length = host_columns[0]->size;

    // Copy the pointers to the column's data and types to the device 
    // as contiguous arrays
    device_columns.reserve(num_cols);
    device_types.reserve(num_cols);
    for(size_type i = 0; i < num_cols; ++i)
    {
      assert(column_length == host_columns[i]->size);

      device_columns.push_back(host_columns[i]->data);
      device_types.push_back(host_columns[i]->dtype);
    }

    d_columns_data = device_columns.data().get();
    d_columns_types = device_types.data().get();
  }

  ~gdf_table(){}

  __host__ __device__
  size_type get_column_length() const
  {
    return column_length;
  }

  gdf_dtype get_build_column_type() const
  {
    return host_columns[build_column_index]->dtype;
  }

  void * get_build_column_data() const
  {
    return host_columns[build_column_index]->data;
  }

  void * get_probe_column_data() const
  {
    return host_columns[build_column_index]->data;
  }

  __host__ 
  void print_row(const size_type row_index, char * msg = "") const
  {
    char row[256];
    sprintf(row,"(");
    for(size_type i = 0; i < num_columns; ++i)
    {
      const gdf_dtype col_type = d_columns_types[i];

      switch(col_type)
      {
        case GDF_INT8:
          {
            sprintf(row,"%d", static_cast<int8_t*>(d_columns_data[i])[row_index]);
            break;
          }
        case GDF_INT16:
          {
            sprintf(row,"%d", static_cast<int16_t*>(d_columns_data[i])[row_index]);
            break;
          }
        case GDF_INT32:
          {
            sprintf(row,"%d", static_cast<int32_t*>(d_columns_data[i])[row_index]);
            break;
          }
        case GDF_INT64:
          {
            sprintf(row,"%ld", static_cast<int64_t*>(d_columns_data[i])[row_index]);
            break;
          }
        case GDF_FLOAT32:
          {
            sprintf(row,"%f", static_cast<float*>(d_columns_data[i])[row_index]);
            break;
          }
        case GDF_FLOAT64:
          {
            sprintf(row,"%f", static_cast<double*>(d_columns_data[i])[row_index]);
            break;
          }
        case GDF_DATE32:
          {
            sprintf(row,"%d", static_cast<int32_t*>(d_columns_data[i])[row_index]);
            break;
          }
        case GDF_DATE64:
          {
            sprintf(row,"%ld", static_cast<int64_t*>(d_columns_data[i])[row_index]);
            break;
          }
        case GDF_TIMESTAMP:
          {
            sprintf(row,"%ld", static_cast<int64_t*>(d_columns_data[i])[row_index]);
            break;
          }
        default:
          assert(false && "Attempted to compare unsupported GDF datatype");
      }
      sprintf(row,", ");
    }
    sprintf(row,")\n");

    printf("%s %s", msg, row);

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
          }
        case GDF_INT16:
          {
            using col_type = int16_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
          }
        case GDF_INT32:
          {
            using col_type = int32_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
          }
        case GDF_INT64:
          {
            using col_type = int64_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
          }
        case GDF_FLOAT32:
          {
            using col_type = float;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
          }
        case GDF_FLOAT64:
          {
            using col_type = double;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
          }
        case GDF_DATE32:
          {
            using col_type = int32_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
          }
        case GDF_DATE64:
          {
            using col_type = int64_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
          }
        case GDF_TIMESTAMP:
          {
            using col_type = int64_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_t key_hash = hasher(current_value);
            hash_value = hasher.hash_combine(hash_value, key_hash);
          }
        default:
          assert(false && "Attempted to hash unsupported GDF datatype");
      }
    }

    return hash_value;
  }


private:

  void ** d_columns_data{nullptr};
  gdf_dtype * d_columns_types{nullptr};

  thrust::device_vector<void*> device_columns;
  thrust::device_vector<gdf_dtype> device_types;

  gdf_column ** host_columns;
  const size_type num_columns;
  size_type column_length;

  // Just use the first column as the build/probe column for now
  const size_type build_column_index{0};
  const size_type probe_column_index{0};

};

#endif
