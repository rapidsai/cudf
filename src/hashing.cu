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

#include <gdf/gdf.h>
#include <gdf/errorutils.h>
#include <thrust/tabulate.h>
#include <thrust/device_vector.h>

#include "join/joining.h"
#include "gdf_table.cuh"
#include "hashmap/hash_functions.cuh"

constexpr int HASH_KERNEL_BLOCK_SIZE = 256;
constexpr int HASH_KERNEL_ROWS_PER_THREAD = 1;

// convert to int dtype with the same size
gdf_dtype to_int_dtype(gdf_dtype type)
{
  switch (type) {
    case GDF_INT8:
    case GDF_INT16:
    case GDF_INT32:
    case GDF_INT64:
      return type;
    case GDF_FLOAT32:
      return GDF_INT32;
    case GDF_FLOAT64:
      return GDF_INT64;
    default:
      return GDF_invalid;
  }
}

__device__ __inline__
uint32_t hashed(void *ptr, int int_dtype, int index)
{
  // TODO: add switch to select the right hash class, currently we only support Murmur3 anyways
  switch (int_dtype) {
  case GDF_INT8:  { default_hash<int8_t> hasher; return hasher(((int8_t*)ptr)[index]); }
  case GDF_INT16: { default_hash<int16_t> hasher; return hasher(((int16_t*)ptr)[index]); }
  case GDF_INT32: { default_hash<int32_t> hasher; return hasher(((int32_t*)ptr)[index]); }
  case GDF_INT64: { default_hash<int64_t> hasher; return hasher(((int64_t*)ptr)[index]); }
  default:
    return 0;
  }
}

template<typename size_type>
__device__ __inline__
void hash_combine(size_type &seed, const uint32_t hash_val)
{
  seed ^= hash_val + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

// one thread handles multiple rows
// d_col_data[i]: column's data (on device)
// d_col_int_dtype[i]: column's dtype (converted to int) 
__global__ void hash_cols(int num_rows, int num_cols, void **d_col_data, gdf_dtype *d_col_int_dtype, int *d_output)
{
  for (int row = threadIdx.x + blockIdx.x * blockDim.x; row < num_rows; row += blockDim.x * gridDim.x) {
    uint32_t seed = 0;
    for (int col = 0; col < num_cols; col++) {
      uint32_t hash_val = hashed(d_col_data[col], d_col_int_dtype[col], row);
      hash_combine(seed, hash_val);
    }
    d_output[row] = seed;
  }
}

gdf_error gdf_hash(int num_cols, gdf_column **input, gdf_hash_func hash, gdf_column *output)
{
  // check that all columns have the same size
  for (int i = 0; i < num_cols; i++)
    if (i > 0 && input[i]->size != input[i-1]->size) return GDF_COLUMN_SIZE_MISMATCH;
  // check that the output dtype is int32
  // TODO: do we need to support int64 as well?
  if (output->dtype != GDF_INT32) return GDF_UNSUPPORTED_DTYPE;
  int64_t num_rows = input[0]->size;

  // copy data pointers to device
  void **d_col_data, **h_col_data;
  cudaMalloc(&d_col_data, num_cols * sizeof(void*));
  cudaMallocHost(&h_col_data, num_cols * sizeof(void*));
  for (int i = 0; i < num_cols; i++)
    h_col_data[i] = input[i]->data;
  cudaMemcpy(d_col_data, h_col_data, num_cols * sizeof(void*), cudaMemcpyDefault);

  // copy dtype (converted to int) to device
  gdf_dtype *d_col_int_dtype, *h_col_int_dtype;
  cudaMalloc(&d_col_int_dtype, num_cols * sizeof(gdf_dtype));
  cudaMallocHost(&h_col_int_dtype, num_cols * sizeof(gdf_dtype));
  for (int i = 0; i < num_cols; i++)
    h_col_int_dtype[i] = to_int_dtype(input[i]->dtype);
  cudaMemcpy(d_col_int_dtype, h_col_int_dtype, num_cols * sizeof(gdf_dtype), cudaMemcpyDefault);

  // launch a kernel
  const int rows_per_block = HASH_KERNEL_BLOCK_SIZE * HASH_KERNEL_ROWS_PER_THREAD;
  const int64_t grid = (num_rows + rows_per_block-1) / rows_per_block;
  hash_cols<<<grid, HASH_KERNEL_BLOCK_SIZE>>>(num_rows, num_cols, d_col_data, d_col_int_dtype, (int32_t*)output->data);

  // TODO: do we need to synchronize here
  cudaDeviceSynchronize();
  CUDA_CHECK_LAST();

  // free temp memory
  cudaFree(d_col_data);
  cudaFreeHost(h_col_data);
  cudaFree(d_col_int_dtype);
  cudaFreeHost(h_col_int_dtype);

  return GDF_SUCCESS;

}


template <template <typename> class hash_function,
          typename size_type>
struct table_row_hasher
{
  table_row_hasher(gdf_table<size_type> const & table_to_hash) 
    : the_table{table_to_hash}
  {}
  
  __device__
  hash_value_type operator()(size_type row_index) const
  {
    return the_table.template hash_row<hash_function>(row_index);
  }

  gdf_table<size_type> const & the_table;
};


/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Functor to map a hash value to a particular 'bin' or partition number
 * that uses the modulo operation.
 */
/* ----------------------------------------------------------------------------*/
template <typename hash_value_t,
          typename size_type,
          typename output_type>
struct modulo_partitioner
{
  __device__
  output_type operator()(hash_value_t hash_value, size_type num_partitions) const
  {
    return hash_value % num_partitions;
  }
};

template <template <typename> class hash_function,
          typename partitioner_type,
          typename size_type>
__global__ 
void compute_row_partition_numbers(gdf_table<size_type> const & the_table, 
                                     const size_type num_rows,
                                     const size_type num_partitions,
                                     const partitioner_type the_partitioner,
                                     size_type * row_partition_numbers,
                                     size_type * partition_sizes)
{
  size_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  // Compute the hash value for each row, store it to the array of hash values
  // and compute the partition to which the hash value belongs and increment
  // the counter for that partition
  while( row_number < num_rows)
  {
    // See here why template disambiguator is required: 
    // https://stackoverflow.com/questions/4077110/template-disambiguator
    const hash_value_type row_hash_value = the_table.template hash_row<hash_function>(row_number);

    const size_type partition_number = the_partitioner(row_hash_value, num_partitions);

    row_partition_numbers[row_number] = partition_number;

    atomicAdd(&(partition_sizes[partition_number]), size_type(1));

    row_number += blockDim.x * gridDim.x;
  }
}

template <typename size_type>
__global__
void compute_output_locations( size_type * row_partition_numbers,
                               size_type num_partitions,
                               size_type * output_locations)
{

}



/* --------------------------------------------------------------------------*/
/** 
 * @brief Partitions an input gdf_table into a specified number of partitions.
 * A hash value is computed for each row in a sub-set of the columns of the 
 * input table. Each hash value is placed in a bin from [0, number of partitions).
 * A copy of the input table is created where the rows are rearranged such that
 * rows with hash values in the same bin are contiguous.
 * 
 * @Param[in] input_table The table to partition
 * @Param[in] table_to_hash Sub-table of the input table with only the columns 
 * that will be hashed
 * @Param[in] num_partitions The number of partitions that table will be rearranged into
 * @Param[out] partition_offsets Preallocated array the size of the number of 
 * partitions. Where partition_offsets[i] indicates the starting position 
 * of partition 'i'
 * @Param[out] partitioned_output Preallocated gdf_columns to hold the rearrangement
 * of the input columns into the desired number of partitions
 * @tparam hash_function The hash function that will be used to hash the rows
 */
/* ----------------------------------------------------------------------------*/
template < template <typename> class hash_function,
           typename size_type>
void hash_partition_gdf_table(gdf_table<size_type> const & input_table,
                              gdf_table<size_type> const & table_to_hash,
                              const size_type num_partitions,
                              size_type * partition_offsets,
                              gdf_table<size_type> & partitioned_output)
{

  
  // Determines how the mapping between hash value and partition number is computed
  using partitioner_type = modulo_partitioner<hash_value_type, size_type, size_type>;

  // Compute the hash value for all the rows in the table to hash
  //table_row_hasher<hash_function, size_type> the_hasher(table_to_hash);
  //thrust::tabulate(row_partition_numbers.begin(), 
  //                 row_partition_numbers.end(), 
  //                 the_hasher);
  const size_type num_rows = table_to_hash.get_column_length();


  size_type * row_partition_numbers{nullptr};
  cudaMalloc(&row_partition_numbers, num_rows * sizeof(hash_value_type));

  size_type * partition_sizes{nullptr};
  cudaMalloc(&partition_sizes, num_partitions * sizeof(size_type));
  cudaMemsetAsync(partition_sizes, 0, num_partitions * sizeof(size_type));

  constexpr int rows_per_block = HASH_KERNEL_BLOCK_SIZE * HASH_KERNEL_ROWS_PER_THREAD;
  const int grid_size = (num_rows + rows_per_block - 1) / rows_per_block;


  // Computes which partition each row belongs to by hashing the row and performing
  // a partitioning operator on the hash value. Also computes the number of
  // rows in each partition
  compute_row_partition_numbers<hash_function>
  <<<grid_size, HASH_KERNEL_BLOCK_SIZE>>>(table_to_hash, 
                                          num_rows,
                                          num_partitions,
                                          partitioner_type(),
                                          row_partition_numbers,
                                          partition_sizes);

  // Compute exclusive scan of the partition sizes in-place to determine 
  // the starting point for each partition in the output
  thrust::exclusive_scan(partition_sizes, 
                         partition_sizes + num_partitions, 
                         partition_sizes);

  // Copy the result of the exlusive scan to the output offsets array
  // to indicate the starting point for each partition in the output
  cudaMemcpyAsync(partition_offsets, 
                  partition_sizes, 
                  num_partitions * sizeof(size_type),
                  cudaMemcpyDeviceToHost);

  



  cudaFree(row_partition_numbers);
  cudaFree(partition_sizes);
}

/* --------------------------------------------------------------------------*/
/**
 * @brief Computes the hash values of the specified rows in the input columns and
 * bins the hash values into the desired number of partitions. Rearranges the input
 * columns such that rows with hash values in the same bin are contiguous.
 *
 * @Param[in] num_input_cols The number of columns in the input columns
 * @Param[in] input[] The input set of columns
 * @Param[in] columns_to_hash[] Indices of the columns in the input set to hash
 * @Param[in] num_cols_to_hash The number of columns to hash
 * @Param[in] num_partitions The number of partitions to rearrange the input rows into
 * @Param[out] partitioned_output Preallocated gdf_columns to hold the rearrangement
 * of the input columns into the desired number of partitions
 * @Param[out] partition_offsets Preallocated array the size of the number of 
 * partitions. Where partition_offsets[i] indicates the starting position 
 * of partition 'i'
 * @Param[in] hash The hash function to use
 *
 * @Returns  If the operation was successful, returns GDF_SUCCESS
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_hash_partition(int num_input_cols,
                             gdf_column * input[],
                             int columns_to_hash[],
                             int num_cols_to_hash,
                             int num_partitions,
                             gdf_column * partitioned_output[],
                             int partition_offsets[],
                             gdf_hash_func hash)
{
  // Use int until gdf API is updated to use something other than int
  // for ordinal variables
  using size_type = int;

  // Ensure all the inputs are non-zero and not null
  if((0 == num_input_cols) 
      || (0 == num_cols_to_hash)
      || (0 == num_partitions)
      || (nullptr == input) 
      || (nullptr == partitioned_output)
      || (nullptr == columns_to_hash)
      || (nullptr == partition_offsets))
  {
    return GDF_INVALID_API_CALL;
  }

  // check that the columns data are not null, have matching types,
  // and the same number of rows
  for (size_type i = 0; i < num_input_cols; i++) {
    if( (nullptr == input[i]->data) 
        || (nullptr == partitioned_output[i]->data))
      return GDF_DATASET_EMPTY;

    if(input[i]->dtype != partitioned_output[i]->dtype) 
      return GDF_PARTITION_DTYPE_MISMATCH;

    if((input[0]->size != input[i]->size) 
        || (input[0]->size != partitioned_output[i]->size))
      return GDF_COLUMN_SIZE_MISMATCH;
  }

  // Wrap input and output columns in gdf_table
  std::unique_ptr< const gdf_table<size_type> > input_table{new gdf_table<size_type>(num_input_cols, input)};
  std::unique_ptr< gdf_table<size_type> > output_table{new gdf_table<size_type>(num_input_cols, partitioned_output)};

  // Create vector of pointers to columns that will be hashed
  std::vector<gdf_column *> gdf_columns_to_hash(num_cols_to_hash);
  for(size_type i = 0; i < num_cols_to_hash; ++i)
  {
    gdf_columns_to_hash[i] = input[columns_to_hash[i]];
  }
  // Create a separate table of the columns to be hashed
  std::unique_ptr< const gdf_table<size_type> > table_to_hash {new gdf_table<size_type>(num_cols_to_hash, 
                                                                                        gdf_columns_to_hash.data())};
  switch(hash)
  {
    case GDF_HASH_MURMUR3:
      {
        hash_partition_gdf_table<MurmurHash3_32>(*input_table, 
                                                 *table_to_hash,
                                                 num_partitions,
                                                 partition_offsets,
                                                 *output_table);
        break;
      }
    default:
      return GDF_INVALID_HASH_FUNCTION;
  }

  return GDF_SUCCESS;
}

