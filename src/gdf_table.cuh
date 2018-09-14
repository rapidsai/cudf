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
#include <gdf/errorutils.h>
#include "hashmap/hash_functions.cuh"
#include "hashmap/managed.cuh"
#include "util/bit_util.cuh"

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
  __device__ gdf_valid_type operator()(const size_type mask_number)
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

template <typename T>
class gdf_table : public managed
{
public:

  using size_type = T;

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
    // to the device  as contiguous arrays
    device_columns_data.reserve(num_cols);
    device_columns_valids.reserve(num_cols);
    device_columns_types.reserve(num_cols);
    for(size_type i = 0; i < num_cols; ++i)
    {
      assert(nullptr != host_columns[i]);
      assert(column_length == host_columns[i]->size);
      if(column_length > 0)
      {
        assert(nullptr != host_columns[i]->data);
      }
      device_columns_data.push_back(host_columns[i]->data);
      device_columns_valids.push_back(host_columns[i]->valid);
      device_columns_types.push_back(host_columns[i]->dtype);
    }

    d_columns_data = device_columns_data.data().get();
    d_columns_valids = device_columns_valids.data().get();
    d_columns_types = device_columns_types.data().get();

    // Allocate storage sufficient to hold a validity bit for every row
    // in the table
    const size_type mask_size = gdf::util::valid_size(column_length);
    device_row_valid.resize(mask_size);

    // If a row contains a single NULL value, then the entire row is considered
    // to be NULL, therefore initialize the row-validity mask with the 
    // bit-wise AND of the validity mask of all the columns
    thrust::tabulate(device_row_valid.begin(),
                     device_row_valid.end(),
                     row_masker<size_type>(d_columns_valids, num_cols));

    d_row_valid = device_row_valid.data().get();

  }

  ~gdf_table(){}

  __host__ 
  size_type get_num_columns() const
  {
    return num_columns;
  }

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

  __device__ bool is_row_valid(size_type row_index) const
  {
    const bool row_valid = gdf::util::get_bit(d_row_valid, row_index);

    return row_valid;
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

    // If either row contains a NULL, then by definition, because NULL != x for all x,
    // the two rows are not equal
    bool valid = this->is_row_valid(my_row_index) && other.is_row_valid(other_row_index);
    if (false == valid) {
      return false;
    }

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
            hash_value_type key_hash = hasher(current_value);
            // Only combine hash values after the first column
            if(i > 0)
              hash_value = hasher.hash_combine(hash_value, key_hash);
            else
              hash_value = key_hash;
            break;
          }
        case GDF_INT16:
          {
            using col_type = int16_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_type key_hash = hasher(current_value);
            // Only combine hash values after the first column
            if(i > 0)
              hash_value = hasher.hash_combine(hash_value, key_hash);
            else
              hash_value = key_hash;
            break;
          }
        case GDF_INT32:
          {
            using col_type = int32_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_type key_hash = hasher(current_value);
            // Only combine hash values after the first column
            if(i > 0)
              hash_value = hasher.hash_combine(hash_value, key_hash);
            else
              hash_value = key_hash;
            break;
          }
        case GDF_INT64:
          {
            using col_type = int64_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_type key_hash = hasher(current_value);
            // Only combine hash values after the first column
            if(i > 0)
              hash_value = hasher.hash_combine(hash_value, key_hash);
            else
              hash_value = key_hash;
            break;
          }
        case GDF_FLOAT32:
          {
            using col_type = float;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_type key_hash = hasher(current_value);
            // Only combine hash values after the first column
            if(i > 0)
              hash_value = hasher.hash_combine(hash_value, key_hash);
            else
              hash_value = key_hash;
            break;
          }
        case GDF_FLOAT64:
          {
            using col_type = double;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_type key_hash = hasher(current_value);
            // Only combine hash values after the first column
            if(i > 0)
              hash_value = hasher.hash_combine(hash_value, key_hash);
            else
              hash_value = key_hash;
            break;
          }
        case GDF_DATE32:
          {
            using col_type = int32_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_type key_hash = hasher(current_value);
            // Only combine hash values after the first column
            if(i > 0)
              hash_value = hasher.hash_combine(hash_value, key_hash);
            else
              hash_value = key_hash;
            break;
          }
        case GDF_DATE64:
          {
            using col_type = int64_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_type key_hash = hasher(current_value);
            // Only combine hash values after the first column
            if(i > 0)
              hash_value = hasher.hash_combine(hash_value, key_hash);
            else
              hash_value = key_hash;
            break;
          }
        case GDF_TIMESTAMP:
          {
            using col_type = int64_t;
            hash_function<col_type> hasher;
            const col_type * current_column = static_cast<col_type*>(d_columns_data[i]);
            const col_type current_value = current_column[row_index];
            hash_value_type key_hash = hasher(current_value);
            // Only combine hash values after the first column
            if(i > 0)
              hash_value = hasher.hash_combine(hash_value, key_hash);
            else
              hash_value = key_hash;
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
  for(auto & s : column_streams)
  {
    cudaStreamCreate(&s);
  }

  // Scatter columns one by one
  for(size_type i = 0; i < num_columns; ++i)
  {
    gdf_column * const current_input_column = this->get_column(i);
    gdf_column * const current_output_column = scattered_output_table.get_column(i);
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

  return gdf_status;
}


private:

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

  thrust::scatter(thrust::cuda::par.on(stream),
                  input_column,
                  input_column + num_rows,
                  row_scatter_map,
                  output_column);

  CUDA_CHECK_LAST();

  return gdf_status;
}

  const size_type num_columns; /** The number of columns in the table */
  size_type column_length;     /** The number of rows in the table */

  gdf_column ** host_columns;  /** The set of gdf_columns that this table wraps */

  thrust::device_vector<void*> device_columns_data; /** Device array of pointers to each columns data */
  void ** d_columns_data{nullptr};                  /** Raw pointer to the device array's data */

  thrust::device_vector<gdf_valid_type*> device_columns_valids;  /** Device array of pointers to each columns validity bitmask*/
  gdf_valid_type** d_columns_valids{nullptr};                   /** Raw pointer to the device array's data */

  thrust::device_vector<gdf_valid_type> device_row_valid;  /** Device array of bitmask for the validity of each row. */
  gdf_valid_type * d_row_valid{nullptr};                   /** Raw pointer to device array's data */

  thrust::device_vector<gdf_dtype> device_columns_types; /** Device array of each columns data type */
  gdf_dtype * d_columns_types{nullptr};                 /** Raw pointer to the device array's data */

};

#endif
