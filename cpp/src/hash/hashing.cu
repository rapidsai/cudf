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

#include <thrust/tabulate.h>

#include "cudf.h"
#include "rmm/rmm.h"
#include "utilities/error_utils.h"
#include "join/joining.h"
#include "dataframe/cudf_table.cuh"
#include "hash/hash_functions.cuh"
#include "utilities/int_fastdiv.h"
#include "utilities/nvtx/nvtx_utils.h"

constexpr int BLOCK_SIZE = 256;
constexpr int ROWS_PER_THREAD = 1;

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
 * @Synopsis  This functor is used to compute the hash value for the rows
 * of a gdf_table
 */
/* ----------------------------------------------------------------------------*/
template <template <typename> class hash_function,
         typename size_type>
struct row_hasher
{
  row_hasher(gdf_table<size_type> const & table_to_hash)
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
 * @Synopsis  Computes the hash value of each row in the input set of columns.
 * 
 * @Param num_cols The number of columns in the input set
 * @Param input The list of columns whose rows will be hashed
 * @Param hash The hash function to use
 * @Param output The hash value of each row of the input
 * 
 * @Returns   
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_hash(int num_cols, gdf_column **input, gdf_hash_func hash, gdf_column *output)
{
  // Ensure inputs aren't null
  if((0 == num_cols)
     || (nullptr == input)
     || (nullptr == output))
  {
    return GDF_DATASET_EMPTY;
  }

  // check that the output dtype is int32
  // TODO: do we need to support int64 as well?
  if (output->dtype != GDF_INT32) 
  {
    return GDF_UNSUPPORTED_DTYPE;
  }

  // Return immediately for empty input/output
  if(nullptr != input[0]) {
    if(0 == input[0]->size){
      return GDF_SUCCESS;
    }
  }
  if(0 == output->size) {
    return GDF_SUCCESS;
  }
  else if(nullptr == output->data) {
    return GDF_DATASET_EMPTY;
  }

  using size_type = int64_t;

  // Wrap input columns in gdf_table
  std::unique_ptr< gdf_table<size_type> > input_table{new gdf_table<size_type>(num_cols, input)};

  const size_type num_rows = input_table->get_column_length();

  // Wrap output buffer in Thrust device_ptr
  hash_value_type * p_output = static_cast<hash_value_type*>(output->data);
  thrust::device_ptr<hash_value_type> row_hash_values = thrust::device_pointer_cast(p_output);


  // Compute the hash value for each row depending on the specified hash function
  switch(hash)
  {
    case GDF_HASH_MURMUR3:
      {
        thrust::tabulate(rmm::exec_policy(cudaStream_t{0}),
                        row_hash_values, 
                         row_hash_values + num_rows, 
                         row_hasher<MurmurHash3_32,size_type>(*input_table));
        break;
      }
    case GDF_HASH_IDENTITY:
      {
        thrust::tabulate(rmm::exec_policy(cudaStream_t{0}),
                         row_hash_values, 
                         row_hash_values + num_rows, 
                         row_hasher<IdentityHash,size_type>(*input_table));
        break;
      }
    default:
      return GDF_INVALID_HASH_FUNCTION;
  }

  CUDA_CHECK_LAST();

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
   Records the size of each partition for each thread block as well as the global
   size of each partition across all thread blocks.
 * 
 * @Param[in] the_table The table whose rows will be partitioned
 * @Param[in] num_rows The number of rows in the table
 * @Param[in] num_partitions The number of partitions to divide the rows into
 * @Param[in] the_partitioner The functor that maps a rows hash value to a partition number
 * @Param[out] row_partition_numbers Array that holds which partition each row belongs to
 * @Param[out] block_partition_sizes Array that holds the size of each partition for each block,
 * i.e., { {block0 partition0 size, block1 partition0 size, ...}, 
         {block0 partition1 size, block1 partition1 size, ...},
         ...
         {block0 partition(num_partitions-1) size, block1 partition(num_partitions -1) size, ...} }
 * @Param[out] global_partition_sizes The number of rows in each partition.
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
                                   size_type * block_partition_sizes,
                                   size_type * global_partition_sizes)
{
  // Accumulate histogram of the size of each partition in shared memory
  extern __shared__ size_type shared_partition_sizes[];

  size_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  // Initialize local histogram
  size_type partition_number = threadIdx.x;
  while(partition_number < num_partitions)
  {
    shared_partition_sizes[partition_number] = 0;
    partition_number += blockDim.x;
  }

  __syncthreads();

  // Compute the hash value for each row, store it to the array of hash values
  // and compute the partition to which the hash value belongs and increment
  // the shared memory counter for that partition
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
  partition_number = threadIdx.x;
  while(partition_number < num_partitions)
  {
    const size_type block_partition_size = shared_partition_sizes[partition_number];

    // Update global size of each partition
    atomicAdd(&global_partition_sizes[partition_number], block_partition_size);

    // Record the size of this partition in this block
    const size_type write_location = partition_number * gridDim.x + blockIdx.x;
    block_partition_sizes[write_location] = block_partition_size;
    partition_number += blockDim.x;
  }
}

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Given an array of partition numbers, computes the final output location
   for each element in the output such that all rows with the same partition are 
   contiguous in memory.
 * 
 * @Param row_partition_numbers The array that records the partition number for each row
 * @Param num_rows The number of rows
 * @Param num_partitions THe number of partitions
 * @Param[out] block_partition_offsets Array that holds the offset of each partition for each thread block,
 * i.e., { {block0 partition0 offset, block1 partition0 offset, ...}, 
         {block0 partition1 offset, block1 partition1 offset, ...},
         ...
         {block0 partition(num_partitions-1) offset, block1 partition(num_partitions -1) offset, ...} }
 */
/* ----------------------------------------------------------------------------*/
template <typename size_type>
__global__ 
void compute_row_output_locations(size_type * row_partition_numbers, 
                                  const size_type num_rows,
                                  const size_type num_partitions,
                                  size_type * block_partition_offsets)
{
  // Shared array that holds the offset of this blocks partitions in 
  // global memory
  extern __shared__ size_type shared_partition_offsets[];

  // Initialize array of this blocks offsets from global array
  size_type partition_number= threadIdx.x;
  while(partition_number < num_partitions)
  {
    shared_partition_offsets[partition_number] = block_partition_offsets[partition_number * gridDim.x + blockIdx.x];
    partition_number += blockDim.x;
  }
  __syncthreads();

  size_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  // Get each row's partition number, and get it's output location by 
  // incrementing block's offset counter for that partition number
  // and store the row's output location in-place
  while( row_number < num_rows )
  {
    // Get partition number of this row
    const size_type partition_number = row_partition_numbers[row_number];

    // Get output location based on partition number by incrementing the corresponding
    // partition offset for this block
    const size_type row_output_location = atomicAdd(&(shared_partition_offsets[partition_number]), size_type(1));

    // Store the row's output location in-place
    row_partition_numbers[row_number] = row_output_location;

    row_number += blockDim.x * gridDim.x;
  }
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

  constexpr int rows_per_block = BLOCK_SIZE * ROWS_PER_THREAD;
  const size_type grid_size = (num_rows + rows_per_block - 1) / rows_per_block;

  // Allocate array to hold which partition each row belongs to
  size_type * row_partition_numbers{nullptr};
  RMM_TRY( RMM_ALLOC((void**)&row_partition_numbers, num_rows * sizeof(hash_value_type), 0) ); // TODO: non-default stream?
  
  // Array to hold the size of each partition computed by each block
  //  i.e., { {block0 partition0 size, block1 partition0 size, ...}, 
  //          {block0 partition1 size, block1 partition1 size, ...},
  //          ...
  //          {block0 partition(num_partitions-1) size, block1 partition(num_partitions -1) size, ...} }
  size_type * block_partition_sizes{nullptr};
  RMM_TRY(RMM_ALLOC((void**)&block_partition_sizes, (grid_size * num_partitions) * sizeof(size_type), 0) );

  // Holds the total number of rows in each partition
  size_type * global_partition_sizes{nullptr};
  RMM_TRY( RMM_ALLOC((void**)&global_partition_sizes, num_partitions * sizeof(size_type), 0) );
  CUDA_TRY( cudaMemsetAsync(global_partition_sizes, 0, num_partitions * sizeof(size_type)) );

  // If the number of partitions is a power of two, we can compute the partition 
  // number of each row more efficiently with bitwise operations
  if( true == is_power_two(num_partitions) )
  {
    // Determines how the mapping between hash value and partition number is computed
    using partitioner_type = bitwise_partitioner<hash_value_type, size_type, size_type>;

    // Computes which partition each row belongs to by hashing the row and performing
    // a partitioning operator on the hash value. Also computes the number of
    // rows in each partition both for each thread block as well as across all blocks
    compute_row_partition_numbers<hash_function>
    <<<grid_size, BLOCK_SIZE, num_partitions * sizeof(size_type)>>>(table_to_hash, 
                                                                    num_rows,
                                                                    num_partitions,
                                                                    partitioner_type(num_partitions),
                                                                    row_partition_numbers,
                                                                    block_partition_sizes,
                                                                    global_partition_sizes);

  }
  else
  {
    // Determines how the mapping between hash value and partition number is computed
    using partitioner_type = modulo_partitioner<hash_value_type, size_type, size_type>;

    // Computes which partition each row belongs to by hashing the row and performing
    // a partitioning operator on the hash value. Also computes the number of
    // rows in each partition both for each thread block as well as across all blocks
    compute_row_partition_numbers<hash_function>
    <<<grid_size, BLOCK_SIZE, num_partitions * sizeof(size_type)>>>(table_to_hash, 
                                                                    num_rows,
                                                                    num_partitions,
                                                                    partitioner_type(num_partitions),
                                                                    row_partition_numbers,
                                                                    block_partition_sizes,
                                                                    global_partition_sizes);
  }


  CUDA_CHECK_LAST();

  
  // Compute exclusive scan of all blocks' partition sizes in-place to determine 
  // the starting point for each blocks portion of each partition in the output
  size_type * scanned_block_partition_sizes{block_partition_sizes};
  thrust::exclusive_scan(rmm::exec_policy(cudaStream_t{0}),
                         block_partition_sizes, 
                         block_partition_sizes + (grid_size * num_partitions), 
                         scanned_block_partition_sizes);
  CUDA_CHECK_LAST();


  // Compute exclusive scan of size of each partition to determine offset location
  // of each partition in final output. This can be done independently on a separate stream
  cudaStream_t s1{};
  cudaStreamCreate(&s1);
  size_type * scanned_global_partition_sizes{global_partition_sizes};
  thrust::exclusive_scan(rmm::exec_policy(s1),
                         global_partition_sizes, 
                         global_partition_sizes + num_partitions,
                         scanned_global_partition_sizes);
  CUDA_CHECK_LAST();

  // Copy the result of the exlusive scan to the output offsets array
  // to indicate the starting point for each partition in the output
  CUDA_TRY(cudaMemcpyAsync(partition_offsets, 
                           scanned_global_partition_sizes, 
                           num_partitions * sizeof(size_type),
                           cudaMemcpyDeviceToHost,
                           s1));

  // Compute the output location for each row in-place based on it's 
  // partition number such that each partition will be contiguous in memory
  size_type * row_output_locations{row_partition_numbers};
  compute_row_output_locations
  <<<grid_size, BLOCK_SIZE, num_partitions * sizeof(size_type)>>>(row_output_locations,
                                                                  num_rows,
                                                                  num_partitions,
                                                                  scanned_block_partition_sizes);

  CUDA_CHECK_LAST();

  // Creates the partitioned output table by scattering the rows of
  // the input table to rows of the output table based on each rows
  // output location
  gdf_error gdf_error_code = input_table.scatter(partitioned_output,
                                                 row_output_locations);

  if(GDF_SUCCESS != gdf_error_code){
    return gdf_error_code;
  }

  CUDA_CHECK_LAST();

  RMM_TRY(RMM_FREE(row_partition_numbers, 0));
  RMM_TRY(RMM_FREE(block_partition_sizes, 0));

  cudaStreamSynchronize(s1);
  cudaStreamDestroy(s1);
  RMM_TRY(RMM_FREE(global_partition_sizes, 0));

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

  const gdf_size_type num_rows{input[0]->size};

  // If the input is empty, return immediately
  if(0 == num_rows)
  {
    return GDF_SUCCESS;
  }

  // TODO Check if the num_rows is > MAX_ROWS (MAX_INT)

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

  PUSH_RANGE("LIBGDF_HASH_PARTITION", PARTITION_COLOR);

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

  POP_RANGE();

  return gdf_status;
}

