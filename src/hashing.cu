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
#include "int_fastdiv.h"

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


/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Functor to map a hash value to a particular 'bin' or partition number
 * that uses the FAST modulo operation implemented in int_fastdiv from here:
 * https://github.com/milakov/int_fastdiv
 */
/* ----------------------------------------------------------------------------*/
template <typename hash_value_t,
          typename size_type,
          typename output_type>
struct fast_modulo_partitioner
{

  fast_modulo_partitioner(int num_partitions) : fast_divisor{num_partitions}{}

  __host__ __device__
  output_type operator()(hash_value_t hash_value) const
  {
    // Using int_fastdiv casts 'hash_value' to an int, which can 
    // result in negative modulos, requiring taking the absolute value
    // Because of the casting it can also return results that are not
    // the same as using the normal % operator
    output_type partition_number = std::abs(hash_value % fast_divisor);

    return partition_number;
  }

  const int_fastdiv fast_divisor;
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
  modulo_partitioner(size_type num_partitions) : divisor{num_partitions}{}

  __host__ __device__
  output_type operator()(hash_value_t hash_value) const 
  {
    return hash_value % divisor;
  }

  const size_type divisor;
};

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  This function determines if a number is a power of 2.
 * 
 * @Param number The number to check.
 * 
 * @Returns True if the number is a power of 2.
 */
/* ----------------------------------------------------------------------------*/
template <typename T>
bool is_power_two( T number )
{
  return (0 == (number & (number - 1)));
}

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Functor to map a hash value to a particular 'bin' or partition number
 * that uses bitshifts. Only works when num_partitions is a power of 2.
 *
 * For n % d, if d is a power of two, then it can be computed more efficiently via 
 * a single bitwise AND as:
 * n & (d - 1)
 */
/* ----------------------------------------------------------------------------*/
template <typename hash_value_t,
          typename size_type,
          typename output_type>
struct bitwise_partitioner
{
  bitwise_partitioner(size_type num_partitions) : divisor{(num_partitions - 1)}
  {
    assert( is_power_two(num_partitions) );
  }

  __host__ __device__
  output_type operator()(hash_value_t hash_value) const 
  {
    return hash_value & (divisor);
  }

  const size_type divisor;
};

/* --------------------------------------------------------------------------*/
/** 
 * @brief Computes which partition each row of a gdf_table will belong to based
   on hashing each row, and applying a partition function to the hash value. 
   Also computes the number of rows that belong to each partition.
 * 
 * @Param[in] the_table The table whose rows will be partitioned
 * @Param[in] num_rows The number of rows in the table
 * @Param[in] num_partitions The number of partitions to divide the rows into
 * @Param[in] the_partitioner The functor that maps a rows hash value to a partition number
 * @Param[out] row_partition_numbers Array that holds which partition each row belongs to
 * @Param[out] partition_sizes The number of rows in each partition.
 */
/* ----------------------------------------------------------------------------*/
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
  // Accumulate histogram of the size of each partition in shared memory
  extern __shared__ size_type shared_partition_sizes[];

  size_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  // Initialize local histogram
  size_type i = threadIdx.x;
  while(i < num_partitions)
  {
    shared_partition_sizes[i] = 0;
    i += blockDim.x;
  }

  __syncthreads();

  // Compute the hash value for each row, store it to the array of hash values
  // and compute the partition to which the hash value belongs and increment
  // the counter for that partition
  while( row_number < num_rows)
  {
    // See here why template disambiguator is required: 
    // https://stackoverflow.com/questions/4077110/template-disambiguator
    const hash_value_type row_hash_value = the_table.template hash_row<hash_function>(row_number);

    const size_type partition_number = the_partitioner(row_hash_value);

    row_partition_numbers[row_number] = partition_number;

    atomicAdd(&(shared_partition_sizes[partition_number]), size_type(1));

    row_number += blockDim.x * gridDim.x;
  }

  __syncthreads();

  // Flush shared memory histogram to global memory
  i = threadIdx.x;
  while(i < num_partitions)
  {
    size_type old = atomicAdd(&(partition_sizes[i]), shared_partition_sizes[i]);
    i += blockDim.x;
  }
}



/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Functor used to map a row's partition number to it's final output location
 */
/* ----------------------------------------------------------------------------*/
template <typename size_type>
struct compute_row_output_location
{
  compute_row_output_location(size_type * const offsets) 
    : partition_offsets{offsets}{};

  __device__
  size_type operator()(size_type partition_number)
  {
    const size_type output_location = atomicAdd(&(partition_offsets[partition_number]), size_type(1));
    return output_location;
  }

  size_type * const __restrict__ partition_offsets;
};

/* --------------------------------------------------------------------------*/
/** 
 * @brief Scatters the values of a column into a new column based on a map that
   maps rows in the input column to rows in the output column. input_column[i]
   will be scattered to output_column[ row_output_locations[i] ]
 * 
 * @Param[in] input_column The input column whose rows will be scattered
 * @Param[in] num_rows The number of rows in the input and output columns
 * @Param[in] row_output_locations An array that maps rows in the input column
   to rows in the output column
 * @Param[out] output_column The rearrangement of the input column 
   based on the mapping determined by the row_output_locations array
 * 
 * @Returns   
 */
/* ----------------------------------------------------------------------------*/
template <typename column_type,
          typename size_type>
gdf_error scatter_column(column_type const * const __restrict__ input_column,
                         size_type const num_rows,
                         size_type const * const __restrict__ row_output_locations,
                         column_type * const __restrict__ output_column,
                         cudaStream_t stream = 0)
{

  gdf_error gdf_status{GDF_SUCCESS};

  thrust::scatter(thrust::cuda::par.on(stream),
                  input_column,
                  input_column + num_rows,
                  row_output_locations,
                  output_column);

  return gdf_status;
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Creates the partitioned output table by scattering the rows of the 
   input table to rows of the output table based on each rows output location.
   I.e., input_table[i] will be scattered to 
   partitioned_output_table[row_output_locations[i]]
 * 
 * @Param[in] input_table The input table to scatter
 * @Param[in] row_output_locations The mapping from input row locations to output row
   locations
 * @Param[out] partitioned_output_table The rearrangement of the input table based 
   on the mappings from the row_output_locations array
 * 
 * @Returns   
 */
/* ----------------------------------------------------------------------------*/
template <typename size_type>
gdf_error scatter_gdf_table(gdf_table<size_type> const & input_table,
                            size_type const * const row_output_locations,
                            gdf_table<size_type> & partitioned_output_table)
{
  gdf_error gdf_status{GDF_SUCCESS};

  const size_type num_columns = input_table.get_num_columns();
  const size_type num_rows = input_table.get_column_length();

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
    gdf_column * current_input_column = input_table.get_column(i);
    gdf_column * current_output_column = partitioned_output_table.get_column(i);
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
                                                   num_rows,
                                                   row_output_locations, 
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
                                                   num_rows,
                                                   row_output_locations, 
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
                                                   num_rows,
                                                   row_output_locations, 
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
                                                   num_rows,
                                                   row_output_locations, 
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
gdf_error hash_partition_gdf_table(gdf_table<size_type> const & input_table,
                                   gdf_table<size_type> const & table_to_hash,
                                   const size_type num_partitions,
                                   size_type * partition_offsets,
                                   gdf_table<size_type> & partitioned_output)
{


  const size_type num_rows = table_to_hash.get_column_length();

  // Allocate array to hold which partition each row belongs to
  size_type * row_partition_numbers{nullptr};
  CUDA_TRY( cudaMalloc(&row_partition_numbers, num_rows * sizeof(hash_value_type)) );

  // Array to hold the size of each partition
  size_type * partition_sizes{nullptr};
  CUDA_TRY(cudaMalloc(&partition_sizes, num_partitions * sizeof(size_type)));
  CUDA_TRY(cudaMemsetAsync(partition_sizes, 0, num_partitions * sizeof(size_type)));

  constexpr int rows_per_block = HASH_KERNEL_BLOCK_SIZE * HASH_KERNEL_ROWS_PER_THREAD;
  const int grid_size = (num_rows + rows_per_block - 1) / rows_per_block;





  // If the number of partitions is a power of two, we can compute the partition 
  // number of each row more efficiently with bitwise operations
  if( true == is_power_two(num_partitions) )
  {
    // Determines how the mapping between hash value and partition number is computed
    using partitioner_type = bitwise_partitioner<hash_value_type, size_type, size_type>;

    // Computes which partition each row belongs to by hashing the row and performing
    // a partitioning operator on the hash value. Also computes the number of
    // rows in each partition
    compute_row_partition_numbers<hash_function>
    <<<grid_size, HASH_KERNEL_BLOCK_SIZE, num_partitions * sizeof(size_type)>>>(table_to_hash, 
                                                                                num_rows,
                                                                                num_partitions,
                                                                                partitioner_type(num_partitions),
                                                                                row_partition_numbers,
                                                                                partition_sizes);

  }
  else
  {
    // Determines how the mapping between hash value and partition number is computed
    using partitioner_type = modulo_partitioner<hash_value_type, size_type, size_type>;

    // Computes which partition each row belongs to by hashing the row and performing
    // a partitioning operator on the hash value. Also computes the number of
    // rows in each partition
    compute_row_partition_numbers<hash_function>
    <<<grid_size, HASH_KERNEL_BLOCK_SIZE, num_partitions * sizeof(size_type)>>>(table_to_hash, 
                                                                                num_rows,
                                                                                num_partitions,
                                                                                partitioner_type(num_partitions),
                                                                                row_partition_numbers,
                                                                                partition_sizes);
  }


  CUDA_CHECK_LAST();

  // Compute exclusive scan of the partition sizes in-place to determine 
  // the starting point for each partition in the output
  size_type * scanned_partition_sizes{partition_sizes};
  thrust::exclusive_scan(thrust::cuda::par,
                         partition_sizes, 
                         partition_sizes + num_partitions, 
                         scanned_partition_sizes);
  CUDA_CHECK_LAST();

  // Copy the result of the exlusive scan to the output offsets array
  // to indicate the starting point for each partition in the output
  CUDA_TRY(cudaMemcpyAsync(partition_offsets, 
                           scanned_partition_sizes, 
                           num_partitions * sizeof(size_type),
                           cudaMemcpyDeviceToHost));


  // Compute the output location for each row in-place based on it's 
  // partition number such that each partition will be contiguous in memory
  size_type * row_output_locations{row_partition_numbers};
  thrust::transform(thrust::cuda::par,
                    row_partition_numbers,
                    row_partition_numbers + num_rows,
                    row_output_locations,
                    compute_row_output_location<size_type>(partition_sizes));


  CUDA_CHECK_LAST();

  // Creates the partitioned output table by scattering the rows of
  // the input table to rows of the output table based on each rows
  // output location
  scatter_gdf_table(input_table, 
                    row_output_locations, 
                    partitioned_output);

  CUDA_CHECK_LAST();

  CUDA_TRY(cudaFree(row_partition_numbers));
  CUDA_TRY(cudaFree(partition_sizes));

  return GDF_SUCCESS;
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

  const size_t num_rows{input[0]->size};

  // If the input is empty, return immediately
  if(0 == num_rows)
  {
    return GDF_SUCCESS;
  }

  // check that the columns data are not null, have matching types,
  // and the same number of rows
  for (size_type i = 0; i < num_input_cols; i++) {
    if( (nullptr == input[i]->data) 
        || (nullptr == partitioned_output[i]->data))
      return GDF_DATASET_EMPTY;

    if(input[i]->dtype != partitioned_output[i]->dtype) 
      return GDF_PARTITION_DTYPE_MISMATCH;

    if((num_rows != input[i]->size) 
        || (num_rows != partitioned_output[i]->size))
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

  gdf_error gdf_status{GDF_SUCCESS};

  switch(hash)
  {
    case GDF_HASH_MURMUR3:
      {
        gdf_status = hash_partition_gdf_table<MurmurHash3_32>(*input_table, 
                                                              *table_to_hash,
                                                              num_partitions,
                                                              partition_offsets,
                                                              *output_table);
        break;
      }
    case GDF_HASH_IDENTITY:
      {
        gdf_status = hash_partition_gdf_table<IdentityHash>(*input_table, 
                                                            *table_to_hash,
                                                            num_partitions,
                                                            partition_offsets,
                                                            *output_table);
        break;
      }
    default:
      gdf_status = GDF_INVALID_HASH_FUNCTION;
  }

  return gdf_status;
}

