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
#include "utilities/error_utils.hpp"
#include "join/joining.h"
#include <table/device_table.cuh>
#include "hash/hash_functions.cuh"
#include "utilities/int_fastdiv.h"
#include "utilities/nvtx/nvtx_utils.h"
#include "copying/scatter.hpp"
#include "types.hpp"

constexpr int BLOCK_SIZE = 256;
constexpr int ROWS_PER_THREAD = 1;

namespace {

/**
 * @brief  This function determines if a number is a power of 2.
 *
 * @param number The number to check.
 *
 * @returns True if the number is a power of 2.
 */
template <typename T>
bool is_power_two(T number) {
  return (0 == (number & (number - 1)));
}

/**
 * @brief  Computes hash value of a row using initial values for each column.
 */
template <template <typename> class hash_function>
struct row_hasher_initial_values {
  row_hasher_initial_values(device_table const& table_to_hash,
                            hash_value_type *initial_hash_values)
      : the_table{table_to_hash}, initial_hash_values(initial_hash_values) {}

  __device__ hash_value_type operator()(gdf_size_type row_index) const {
    return hash_row<hash_function>(the_table, row_index, initial_hash_values);
  }

  device_table the_table;
  hash_value_type *initial_hash_values{nullptr};
};

/**
 * @brief  Computes hash value of a row 
 */
template <template <typename> class hash_function>
struct row_hasher {
  row_hasher(device_table const& table_to_hash) : the_table{table_to_hash} {}

  __device__ hash_value_type operator()(gdf_size_type row_index) const {
    return hash_row<hash_function>(the_table, row_index);
  }

  device_table the_table;
};
}  // namespace

/* --------------------------------------------------------------------------*/
/** 
 * @brief Computes the hash value of each row in the input set of columns.
 * 
 * @param[in] num_cols The number of columns in the input set
 * @param[in] input The list of columns whose rows will be hashed
 * @param[in] hash The hash function to use
 * @param[in] initial_hash_values Optional array in device memory specifying an initial hash value for each column
 * that will be combined with the hash of every element in the column. If this argument is `nullptr`,
 * then each element will be hashed as-is.
 * @param[out] output The hash value of each row of the input
 * 
 * @return    GDF_SUCCESS if the operation was successful, otherwise an
 *            appropriate error code.
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_hash(int num_cols,
                   gdf_column **input,
                   gdf_hash_func hash,
                   hash_value_type *initial_hash_values,
                   gdf_column *output)
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

  // Wrap input columns in device_table
  auto input_table = device_table::create(num_cols, input);

  const gdf_size_type num_rows = input_table->num_rows();

  // Wrap output buffer in Thrust device_ptr
  hash_value_type * p_output = static_cast<hash_value_type*>(output->data);
  thrust::device_ptr<hash_value_type> row_hash_values = thrust::device_pointer_cast(p_output);


  // Compute the hash value for each row depending on the specified hash function
  switch (hash) {
    case GDF_HASH_MURMUR3: {
      if (nullptr == initial_hash_values) {
        thrust::tabulate(rmm::exec_policy()->on(0), row_hash_values,
                         row_hash_values + num_rows,
                         row_hasher<MurmurHash3_32>(*input_table));

      } else {
        thrust::tabulate(rmm::exec_policy()->on(0), row_hash_values,
                         row_hash_values + num_rows,
                         row_hasher_initial_values<MurmurHash3_32>(
                             *input_table, initial_hash_values));
      }
      break;
    }
    case GDF_HASH_IDENTITY: {
      if (nullptr == initial_hash_values) {
        thrust::tabulate(rmm::exec_policy()->on(0), row_hash_values,
                         row_hash_values + num_rows,
                         row_hasher<IdentityHash>(*input_table));

      } else {
        thrust::tabulate(rmm::exec_policy()->on(0), row_hash_values,
                         row_hash_values + num_rows,
                         row_hasher_initial_values<IdentityHash>(
                             *input_table, initial_hash_values));
      }
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
 * @brief  Functor to map a hash value to a particular 'bin' or partition number
 * that uses the FAST modulo operation implemented in int_fastdiv from here:
 * https://github.com/milakov/int_fastdiv
 */
/* ----------------------------------------------------------------------------*/
template <typename hash_value_t,
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
 * @brief  Functor to map a hash value to a particular 'bin' or partition number
 * that uses the modulo operation.
 */
/* ----------------------------------------------------------------------------*/
template <typename hash_value_t>
struct modulo_partitioner
{
  modulo_partitioner(gdf_size_type num_partitions) : divisor{num_partitions}{}

  __host__ __device__
  gdf_size_type operator()(hash_value_t hash_value) const 
  {
    return hash_value % divisor;
  }

  const gdf_size_type divisor;
};


/* --------------------------------------------------------------------------*/
/** 
 * @brief  Functor to map a hash value to a particular 'bin' or partition number
 * that uses bitshifts. Only works when num_partitions is a power of 2.
 *
 * For n % d, if d is a power of two, then it can be computed more efficiently via 
 * a single bitwise AND as:
 * n & (d - 1)
 */
/* ----------------------------------------------------------------------------*/
template <typename hash_value_t>
struct bitwise_partitioner
{
  bitwise_partitioner(gdf_size_type num_partitions) : divisor{(num_partitions - 1)}
  {
    assert( is_power_two(num_partitions) );
  }

  __host__ __device__
  gdf_size_type operator()(hash_value_t hash_value) const 
  {
    return hash_value & (divisor);
  }

  const gdf_size_type divisor;
};

/* --------------------------------------------------------------------------*/
/** 
 * @brief Computes which partition each row of a device_table will belong to based
   on hashing each row, and applying a partition function to the hash value. 
   Records the size of each partition for each thread block as well as the global
   size of each partition across all thread blocks.
 * 
 * @param[in] the_table The table whose rows will be partitioned
 * @param[in] num_rows The number of rows in the table
 * @param[in] num_partitions The number of partitions to divide the rows into
 * @param[in] the_partitioner The functor that maps a rows hash value to a partition number
 * @param[out] row_partition_numbers Array that holds which partition each row belongs to
 * @param[out] block_partition_sizes Array that holds the size of each partition for each block,
 * i.e., { {block0 partition0 size, block1 partition0 size, ...}, 
         {block0 partition1 size, block1 partition1 size, ...},
         ...
         {block0 partition(num_partitions-1) size, block1 partition(num_partitions -1) size, ...} }
 * @param[out] global_partition_sizes The number of rows in each partition.
 */
/* ----------------------------------------------------------------------------*/
template <template <typename> class hash_function,
          typename partitioner_type>
__global__ 
void compute_row_partition_numbers(device_table the_table, 
                                   const gdf_size_type num_rows,
                                   const gdf_size_type num_partitions,
                                   const partitioner_type the_partitioner,
                                   gdf_size_type * row_partition_numbers,
                                   gdf_size_type * block_partition_sizes,
                                   gdf_size_type * global_partition_sizes)
{
  // Accumulate histogram of the size of each partition in shared memory
  extern __shared__ gdf_size_type shared_partition_sizes[];

  gdf_size_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  // Initialize local histogram
  gdf_size_type partition_number = threadIdx.x;
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
    const hash_value_type row_hash_value =
        hash_row<hash_function>(the_table, row_number);

    const gdf_size_type partition_number = the_partitioner(row_hash_value);

    row_partition_numbers[row_number] = partition_number;

    atomicAdd(&(shared_partition_sizes[partition_number]), gdf_size_type(1));

    row_number += blockDim.x * gridDim.x;
  }

  __syncthreads();

  // Flush shared memory histogram to global memory
  partition_number = threadIdx.x;
  while(partition_number < num_partitions)
  {
    const gdf_size_type block_partition_size = shared_partition_sizes[partition_number];

    // Update global size of each partition
    atomicAdd(&global_partition_sizes[partition_number], block_partition_size);

    // Record the size of this partition in this block
    const gdf_size_type write_location = partition_number * gridDim.x + blockIdx.x;
    block_partition_sizes[write_location] = block_partition_size;
    partition_number += blockDim.x;
  }
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Given an array of partition numbers, computes the final output location
   for each element in the output such that all rows with the same partition are 
   contiguous in memory.
 * 
 * @param row_partition_numbers The array that records the partition number for each row
 * @param num_rows The number of rows
 * @param num_partitions THe number of partitions
 * @param[out] block_partition_offsets Array that holds the offset of each partition for each thread block,
 * i.e., { {block0 partition0 offset, block1 partition0 offset, ...}, 
         {block0 partition1 offset, block1 partition1 offset, ...},
         ...
         {block0 partition(num_partitions-1) offset, block1 partition(num_partitions -1) offset, ...} }
 */
/* ----------------------------------------------------------------------------*/
__global__ 
void compute_row_output_locations(gdf_size_type * row_partition_numbers, 
                                  const gdf_size_type num_rows,
                                  const gdf_size_type num_partitions,
                                  gdf_size_type * block_partition_offsets)
{
  // Shared array that holds the offset of this blocks partitions in 
  // global memory
  extern __shared__ gdf_size_type shared_partition_offsets[];

  // Initialize array of this blocks offsets from global array
  gdf_size_type partition_number= threadIdx.x;
  while(partition_number < num_partitions)
  {
    shared_partition_offsets[partition_number] = block_partition_offsets[partition_number * gridDim.x + blockIdx.x];
    partition_number += blockDim.x;
  }
  __syncthreads();

  gdf_size_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  // Get each row's partition number, and get it's output location by 
  // incrementing block's offset counter for that partition number
  // and store the row's output location in-place
  while( row_number < num_rows )
  {
    // Get partition number of this row
    const gdf_size_type partition_number = row_partition_numbers[row_number];

    // Get output location based on partition number by incrementing the corresponding
    // partition offset for this block
    const gdf_size_type row_output_location = atomicAdd(&(shared_partition_offsets[partition_number]), gdf_size_type(1));

    // Store the row's output location in-place
    row_partition_numbers[row_number] = row_output_location;

    row_number += blockDim.x * gridDim.x;
  }
}



/* --------------------------------------------------------------------------*/
/** 
 * @brief Partitions an input device_table into a specified number of partitions.
 * A hash value is computed for each row in a sub-set of the columns of the 
 * input table. Each hash value is placed in a bin from [0, number of partitions).
 * A copy of the input table is created where the rows are rearranged such that
 * rows with hash values in the same bin are contiguous.
 * 
 * @param[in] input_table The table to partition
 * @param[in] table_to_hash Sub-table of the input table with only the columns 
 * that will be hashed
 * @param[in] num_partitions The number of partitions that table will be rearranged into
 * @param[out] partition_offsets Preallocated array the size of the number of 
 * partitions. Where partition_offsets[i] indicates the starting position 
 * of partition 'i'
 * @param[out] partitioned_output Preallocated gdf_columns to hold the rearrangement
 * of the input columns into the desired number of partitions
 * @tparam hash_function The hash function that will be used to hash the rows
 */
/* ----------------------------------------------------------------------------*/
template <template <typename> class hash_function>
gdf_error hash_partition_table(cudf::table const &input_table,
                               cudf::table const &table_to_hash,
                               const gdf_size_type num_partitions,
                               gdf_size_type *partition_offsets,
                               cudf::table &partitioned_output) {
  const gdf_size_type num_rows = table_to_hash.num_rows();

  constexpr gdf_size_type rows_per_block = BLOCK_SIZE * ROWS_PER_THREAD;
  const gdf_size_type grid_size = (num_rows + rows_per_block - 1) / rows_per_block;

  // Allocate array to hold which partition each row belongs to
  gdf_size_type * row_partition_numbers{nullptr};
  RMM_TRY( RMM_ALLOC((void**)&row_partition_numbers, num_rows * sizeof(hash_value_type), 0) ); // TODO: non-default stream?
  
  // Array to hold the size of each partition computed by each block
  //  i.e., { {block0 partition0 size, block1 partition0 size, ...}, 
  //          {block0 partition1 size, block1 partition1 size, ...},
  //          ...
  //          {block0 partition(num_partitions-1) size, block1 partition(num_partitions -1) size, ...} }
  gdf_size_type * block_partition_sizes{nullptr};
  RMM_TRY(RMM_ALLOC((void**)&block_partition_sizes, (grid_size * num_partitions) * sizeof(gdf_size_type), 0) );

  // Holds the total number of rows in each partition
  gdf_size_type * global_partition_sizes{nullptr};
  RMM_TRY( RMM_ALLOC((void**)&global_partition_sizes, num_partitions * sizeof(gdf_size_type), 0) );
  CUDA_TRY( cudaMemsetAsync(global_partition_sizes, 0, num_partitions * sizeof(gdf_size_type)) );

  auto d_table_to_hash = device_table::create(table_to_hash);

  // If the number of partitions is a power of two, we can compute the partition 
  // number of each row more efficiently with bitwise operations
  if( true == is_power_two(num_partitions) )
  {
    // Determines how the mapping between hash value and partition number is computed
    using partitioner_type = bitwise_partitioner<hash_value_type>;

    // Computes which partition each row belongs to by hashing the row and performing
    // a partitioning operator on the hash value. Also computes the number of
    // rows in each partition both for each thread block as well as across all blocks
    compute_row_partition_numbers<hash_function>
        <<<grid_size, BLOCK_SIZE, num_partitions * sizeof(gdf_size_type)>>>(
            *d_table_to_hash, num_rows, num_partitions,
            partitioner_type(num_partitions), row_partition_numbers,
            block_partition_sizes, global_partition_sizes);

  }
  else
  {
    // Determines how the mapping between hash value and partition number is computed
    using partitioner_type = modulo_partitioner<hash_value_type>;

    // Computes which partition each row belongs to by hashing the row and performing
    // a partitioning operator on the hash value. Also computes the number of
    // rows in each partition both for each thread block as well as across all blocks
    compute_row_partition_numbers<hash_function>
        <<<grid_size, BLOCK_SIZE, num_partitions * sizeof(gdf_size_type)>>>(
            *d_table_to_hash, num_rows, num_partitions,
            partitioner_type(num_partitions), row_partition_numbers,
            block_partition_sizes, global_partition_sizes);
  }


  CUDA_CHECK_LAST();

  
  // Compute exclusive scan of all blocks' partition sizes in-place to determine 
  // the starting point for each blocks portion of each partition in the output
  gdf_size_type * scanned_block_partition_sizes{block_partition_sizes};
  thrust::exclusive_scan(rmm::exec_policy()->on(0),
                         block_partition_sizes, 
                         block_partition_sizes + (grid_size * num_partitions), 
                         scanned_block_partition_sizes);
  CUDA_CHECK_LAST();


  // Compute exclusive scan of size of each partition to determine offset location
  // of each partition in final output. This can be done independently on a separate stream
  cudaStream_t s1{};
  cudaStreamCreate(&s1);
  gdf_size_type * scanned_global_partition_sizes{global_partition_sizes};
  thrust::exclusive_scan(rmm::exec_policy(s1)->on(s1),
                         global_partition_sizes, 
                         global_partition_sizes + num_partitions,
                         scanned_global_partition_sizes);
  CUDA_CHECK_LAST();

  // Copy the result of the exlusive scan to the output offsets array
  // to indicate the starting point for each partition in the output
  CUDA_TRY(cudaMemcpyAsync(partition_offsets, 
                           scanned_global_partition_sizes, 
                           num_partitions * sizeof(gdf_size_type),
                           cudaMemcpyDeviceToHost,
                           s1));

  // Compute the output location for each row in-place based on it's 
  // partition number such that each partition will be contiguous in memory
  gdf_size_type * row_output_locations{row_partition_numbers};
  compute_row_output_locations
  <<<grid_size, BLOCK_SIZE, num_partitions * sizeof(gdf_size_type)>>>(row_output_locations,
                                                                  num_rows,
                                                                  num_partitions,
                                                                  scanned_block_partition_sizes);

  CUDA_CHECK_LAST();

  // Creates the partitioned output table by scattering the rows of
  // the input table to rows of the output table based on each rows
  // output location
  cudf::detail::scatter(&input_table, row_output_locations,
                        &partitioned_output);

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
 * @param[in] num_input_cols The number of columns in the input columns
 * @param[in] input[] The input set of columns
 * @param[in] columns_to_hash[] Indices of the columns in the input set to hash
 * @param[in] num_cols_to_hash The number of columns to hash
 * @param[in] num_partitions The number of partitions to rearrange the input rows into
 * @param[out] partitioned_output Preallocated gdf_columns to hold the rearrangement
 * of the input columns into the desired number of partitions
 * @param[out] partition_offsets Preallocated array the size of the number of 
 * partitions. Where partition_offsets[i] indicates the starting position 
 * of partition 'i'
 * @param[in] hash The hash function to use
 *
 * @returns  If the operation was successful, returns GDF_SUCCESS
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
  for (gdf_size_type i = 0; i < num_input_cols; i++) {
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

  cudf::table input_table(input, num_input_cols);
  cudf::table output_table(partitioned_output, num_input_cols);

  // Create vector of pointers to columns that will be hashed
  std::vector<gdf_column *> gdf_columns_to_hash(num_cols_to_hash);
  for(gdf_size_type i = 0; i < num_cols_to_hash; ++i)
  {
    gdf_columns_to_hash[i] = input[columns_to_hash[i]];
  }

  // Create a separate table of the columns to be hashed
  cudf::table table_to_hash(gdf_columns_to_hash.data(),
                            gdf_columns_to_hash.size());

  gdf_error gdf_status{GDF_SUCCESS};

  switch (hash) {
    case GDF_HASH_MURMUR3: {
      gdf_status = hash_partition_table<MurmurHash3_32>(
          input_table, table_to_hash, num_partitions, partition_offsets,
          output_table);
      break;
    }
    case GDF_HASH_IDENTITY: {
      gdf_status = hash_partition_table<IdentityHash>(
          input_table, table_to_hash, num_partitions, partition_offsets,
          output_table);
      break;
    }
    default:
      gdf_status = GDF_INVALID_HASH_FUNCTION;
  }

  POP_RANGE();

  return gdf_status;
}

