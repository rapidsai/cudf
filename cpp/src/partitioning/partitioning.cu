/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cub/cub.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/scatter.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/partitioning.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>

namespace cudf {
namespace {
// Launch configuration for optimized hash partition
constexpr size_type OPTIMIZED_BLOCK_SIZE                     = 512;
constexpr size_type OPTIMIZED_ROWS_PER_THREAD                = 8;
constexpr size_type ELEMENTS_PER_THREAD                      = 2;
constexpr size_type THRESHOLD_FOR_OPTIMIZED_PARTITION_KERNEL = 1024;

// Launch configuration for fallback hash partition
constexpr size_type FALLBACK_BLOCK_SIZE      = 256;
constexpr size_type FALLBACK_ROWS_PER_THREAD = 1;

/**
 * @brief  Functor to map a hash value to a particular 'bin' or partition number
 * that uses the modulo operation.
 */
template <typename hash_value_t>
class modulo_partitioner {
 public:
  modulo_partitioner(size_type num_partitions) : divisor{num_partitions} {}

  __device__ size_type operator()(hash_value_t hash_value) const { return hash_value % divisor; }

 private:
  const size_type divisor;
};

template <typename T>
bool is_power_two(T number)
{
  return (0 == (number & (number - 1)));
}

/**
 * @brief  Functor to map a hash value to a particular 'bin' or partition number
 * that uses a bitwise mask. Only works when num_partitions is a power of 2.
 *
 * For n % d, if d is a power of two, then it can be computed more efficiently
 * via a single bitwise AND as: n & (d - 1)
 */
template <typename hash_value_t>
class bitwise_partitioner {
 public:
  bitwise_partitioner(size_type num_partitions) : mask{(num_partitions - 1)}
  {
    assert(is_power_two(num_partitions));
  }

  __device__ size_type operator()(hash_value_t hash_value) const
  {
    return hash_value & mask;  // hash_value & (num_partitions - 1)
  }

 private:
  const size_type mask;
};

/* --------------------------------------------------------------------------*/
/**
 * @brief Computes which partition each row of a device_table will belong to
 based on hashing each row, and applying a partition function to the hash value.
   Records the size of each partition for each thread block as well as the
 global size of each partition across all thread blocks.
 *
 * @param[in] the_table The table whose rows will be partitioned
 * @param[in] num_rows The number of rows in the table
 * @param[in] num_partitions The number of partitions to divide the rows into
 * @param[in] the_partitioner The functor that maps a rows hash value to a
 partition number
 * @param[out] row_partition_numbers Array that holds which partition each row
 belongs to
 * @param[out] row_partition_offset Array that holds the offset of each row in
 its partition of
 * the thread block
 * @param[out] block_partition_sizes Array that holds the size of each partition
 for each block,
 * i.e., { {block0 partition0 size, block1 partition0 size, ...},
         {block0 partition1 size, block1 partition1 size, ...},
         ...
         {block0 partition(num_partitions-1) size, block1
 partition(num_partitions -1) size, ...} }
 * @param[out] global_partition_sizes The number of rows in each partition.
 */
/* ----------------------------------------------------------------------------*/
template <class row_hasher_t, typename partitioner_type>
__global__ void compute_row_partition_numbers(row_hasher_t the_hasher,
                                              const size_type num_rows,
                                              const size_type num_partitions,
                                              const partitioner_type the_partitioner,
                                              size_type* __restrict__ row_partition_numbers,
                                              size_type* __restrict__ row_partition_offset,
                                              size_type* __restrict__ block_partition_sizes,
                                              size_type* __restrict__ global_partition_sizes)
{
  // Accumulate histogram of the size of each partition in shared memory
  extern __shared__ size_type shared_partition_sizes[];

  size_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  // Initialize local histogram
  size_type partition_number = threadIdx.x;
  while (partition_number < num_partitions) {
    shared_partition_sizes[partition_number] = 0;
    partition_number += blockDim.x;
  }

  __syncthreads();

  // Compute the hash value for each row, store it to the array of hash values
  // and compute the partition to which the hash value belongs and increment
  // the shared memory counter for that partition
  while (row_number < num_rows) {
    const hash_value_type row_hash_value = the_hasher(row_number);

    const size_type partition_number = the_partitioner(row_hash_value);

    row_partition_numbers[row_number] = partition_number;

    row_partition_offset[row_number] =
      atomicAdd(&(shared_partition_sizes[partition_number]), size_type(1));

    row_number += blockDim.x * gridDim.x;
  }

  __syncthreads();

  // Flush shared memory histogram to global memory
  partition_number = threadIdx.x;
  while (partition_number < num_partitions) {
    const size_type block_partition_size = shared_partition_sizes[partition_number];

    // Update global size of each partition
    atomicAdd(&global_partition_sizes[partition_number], block_partition_size);

    // Record the size of this partition in this block
    const size_type write_location        = partition_number * gridDim.x + blockIdx.x;
    block_partition_sizes[write_location] = block_partition_size;
    partition_number += blockDim.x;
  }
}

/* --------------------------------------------------------------------------*/
/**
 * @brief  Given an array of partition numbers, computes the final output
 location for each element in the output such that all rows with the same
 partition are contiguous in memory.
 *
 * @param row_partition_numbers The array that records the partition number for
 each row
 * @param num_rows The number of rows
 * @param num_partitions THe number of partitions
 * @param[out] block_partition_offsets Array that holds the offset of each
 partition for each thread block,
 * i.e., { {block0 partition0 offset, block1 partition0 offset, ...},
         {block0 partition1 offset, block1 partition1 offset, ...},
         ...
         {block0 partition(num_partitions-1) offset, block1
 partition(num_partitions -1) offset, ...} }
 */
/* ----------------------------------------------------------------------------*/
__global__ void compute_row_output_locations(size_type* __restrict__ row_partition_numbers,
                                             const size_type num_rows,
                                             const size_type num_partitions,
                                             size_type* __restrict__ block_partition_offsets)
{
  // Shared array that holds the offset of this blocks partitions in
  // global memory
  extern __shared__ size_type shared_partition_offsets[];

  // Initialize array of this blocks offsets from global array
  size_type partition_number = threadIdx.x;
  while (partition_number < num_partitions) {
    shared_partition_offsets[partition_number] =
      block_partition_offsets[partition_number * gridDim.x + blockIdx.x];
    partition_number += blockDim.x;
  }
  __syncthreads();

  size_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  // Get each row's partition number, and get it's output location by
  // incrementing block's offset counter for that partition number
  // and store the row's output location in-place
  while (row_number < num_rows) {
    // Get partition number of this row
    const size_type partition_number = row_partition_numbers[row_number];

    // Get output location based on partition number by incrementing the
    // corresponding partition offset for this block
    const size_type row_output_location =
      atomicAdd(&(shared_partition_offsets[partition_number]), size_type(1));

    // Store the row's output location in-place
    row_partition_numbers[row_number] = row_output_location;

    row_number += blockDim.x * gridDim.x;
  }
}

/* --------------------------------------------------------------------------*/
/**
 * @brief Move one column from the input table to the hashed table.
 *
 * @param[in] input_buf Data buffer of the column in the input table
 * @param[out] output_buf Preallocated data buffer of the column in the output
 * table
 * @param[in] num_rows The number of rows in each column
 * @param[in] num_partitions The number of partitions to divide the rows into
 * @param[in] row_partition_numbers Array that holds which partition each row
 * belongs to
 * @param[in] row_partition_offset Array that holds the offset of each row in
 * its partition of the thread block.
 * @param[in] block_partition_sizes Array that holds the size of each partition
 * for each block
 * @param[in] scanned_block_partition_sizes The scan of block_partition_sizes
 */
/* ----------------------------------------------------------------------------*/
template <typename InputIter, typename DataType>
__global__ void copy_block_partitions(InputIter input_iter,
                                      DataType* __restrict__ output_buf,
                                      const size_type num_rows,
                                      const size_type num_partitions,
                                      size_type const* __restrict__ row_partition_numbers,
                                      size_type const* __restrict__ row_partition_offset,
                                      size_type const* __restrict__ block_partition_sizes,
                                      size_type const* __restrict__ scanned_block_partition_sizes)
{
  extern __shared__ char shared_memory[];
  auto block_output = reinterpret_cast<DataType*>(shared_memory);
  auto partition_offset_shared =
    reinterpret_cast<size_type*>(block_output + OPTIMIZED_BLOCK_SIZE * OPTIMIZED_ROWS_PER_THREAD);
  auto partition_offset_global = partition_offset_shared + num_partitions + 1;

  typedef cub::BlockScan<size_type, OPTIMIZED_BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // use ELEMENTS_PER_THREAD=2 to support upto 1024 partitions
  size_type temp_histo[ELEMENTS_PER_THREAD];

  for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    if (ELEMENTS_PER_THREAD * threadIdx.x + i < num_partitions) {
      temp_histo[i] =
        block_partition_sizes[blockIdx.x + (ELEMENTS_PER_THREAD * threadIdx.x + i) * gridDim.x];
    } else {
      temp_histo[i] = 0;
    }
  }

  __syncthreads();

  BlockScan(temp_storage).InclusiveSum(temp_histo, temp_histo);

  __syncthreads();

  if (threadIdx.x == 0) { partition_offset_shared[0] = 0; }

  // Calculate the offset in shared memory of each partition in this thread
  // block
  for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    if (ELEMENTS_PER_THREAD * threadIdx.x + i < num_partitions) {
      partition_offset_shared[ELEMENTS_PER_THREAD * threadIdx.x + i + 1] = temp_histo[i];
    }
  }

  // Fetch the offset in the output buffer of each partition in this thread
  // block
  for (size_type ipartition = threadIdx.x; ipartition < num_partitions; ipartition += blockDim.x) {
    partition_offset_global[ipartition] =
      scanned_block_partition_sizes[ipartition * gridDim.x + blockIdx.x];
  }

  __syncthreads();

  // Fetch the input data to shared memory
  for (size_type row_number = threadIdx.x + blockIdx.x * blockDim.x; row_number < num_rows;
       row_number += blockDim.x * gridDim.x) {
    size_type const ipartition = row_partition_numbers[row_number];

    block_output[partition_offset_shared[ipartition] + row_partition_offset[row_number]] =
      input_iter[row_number];
  }

  __syncthreads();

  // Copy data from shared memory to output using 32 threads for each partition
  constexpr int nthreads_partition = 32;
  static_assert(OPTIMIZED_BLOCK_SIZE % nthreads_partition == 0,
                "BLOCK_SIZE must be divisible by number of threads");

  for (size_type ipartition = threadIdx.x / nthreads_partition; ipartition < num_partitions;
       ipartition += OPTIMIZED_BLOCK_SIZE / nthreads_partition) {
    size_type const nelements_partition =
      partition_offset_shared[ipartition + 1] - partition_offset_shared[ipartition];

    for (size_type row_offset = threadIdx.x % nthreads_partition; row_offset < nelements_partition;
         row_offset += nthreads_partition) {
      output_buf[partition_offset_global[ipartition] + row_offset] =
        block_output[partition_offset_shared[ipartition] + row_offset];
    }
  }
}

template <typename InputIter, typename OutputIter>
void copy_block_partitions_impl(InputIter const input,
                                OutputIter output,
                                size_type num_rows,
                                size_type num_partitions,
                                size_type const* row_partition_numbers,
                                size_type const* row_partition_offset,
                                size_type const* block_partition_sizes,
                                size_type const* scanned_block_partition_sizes,
                                size_type grid_size,
                                cudaStream_t stream)
{
  // We need 3 chunks of shared memory:
  // 1. BLOCK_SIZE * ROWS_PER_THREAD elements of size_type for copying to output
  // 2. num_partitions + 1 elements of size_type for per-block partition offsets
  // 3. num_partitions + 1 elements of size_type for global partition offsets
  int const smem = OPTIMIZED_BLOCK_SIZE * OPTIMIZED_ROWS_PER_THREAD * sizeof(*output) +
                   (num_partitions + 1) * sizeof(size_type) * 2;

  copy_block_partitions<<<grid_size, OPTIMIZED_BLOCK_SIZE, smem, stream>>>(
    input,
    output,
    num_rows,
    num_partitions,
    row_partition_numbers,
    row_partition_offset,
    block_partition_sizes,
    scanned_block_partition_sizes);
}

rmm::device_vector<size_type> compute_gather_map(size_type num_rows,
                                                 size_type num_partitions,
                                                 size_type const* row_partition_numbers,
                                                 size_type const* row_partition_offset,
                                                 size_type const* block_partition_sizes,
                                                 size_type const* scanned_block_partition_sizes,
                                                 size_type grid_size,
                                                 cudaStream_t stream)
{
  auto sequence = thrust::make_counting_iterator(0);
  rmm::device_vector<size_type> gather_map(num_rows);

  copy_block_partitions_impl(sequence,
                             gather_map.data().get(),
                             num_rows,
                             num_partitions,
                             row_partition_numbers,
                             row_partition_offset,
                             block_partition_sizes,
                             scanned_block_partition_sizes,
                             grid_size,
                             stream);

  return gather_map;
}

struct copy_block_partitions_dispatcher {
  template <typename DataType, std::enable_if_t<is_fixed_width<DataType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input,
                                     const size_type num_partitions,
                                     size_type const* row_partition_numbers,
                                     size_type const* row_partition_offset,
                                     size_type const* block_partition_sizes,
                                     size_type const* scanned_block_partition_sizes,
                                     size_type grid_size,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    rmm::device_buffer output(input.size() * sizeof(DataType), stream, mr);

    copy_block_partitions_impl(input.data<DataType>(),
                               static_cast<DataType*>(output.data()),
                               input.size(),
                               num_partitions,
                               row_partition_numbers,
                               row_partition_offset,
                               block_partition_sizes,
                               scanned_block_partition_sizes,
                               grid_size,
                               stream);

    return std::make_unique<column>(input.type(), input.size(), std::move(output));
  }

  template <typename DataType, std::enable_if_t<not is_fixed_width<DataType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input,
                                     const size_type num_partitions,
                                     size_type const* row_partition_numbers,
                                     size_type const* row_partition_offset,
                                     size_type const* block_partition_sizes,
                                     size_type const* scanned_block_partition_sizes,
                                     size_type grid_size,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    // Use move_to_output_buffer to create an equivalent gather map
    auto gather_map = compute_gather_map(input.size(),
                                         num_partitions,
                                         row_partition_numbers,
                                         row_partition_offset,
                                         block_partition_sizes,
                                         scanned_block_partition_sizes,
                                         grid_size,
                                         stream);

    // Use gather instead for non-fixed width types
    return type_dispatcher(input.type(),
                           detail::column_gatherer{},
                           input,
                           gather_map.begin(),
                           gather_map.end(),
                           false,
                           stream,
                           mr);
  }
};

// NOTE hash_has_nulls must be true if table_to_hash has nulls
template <bool hash_has_nulls>
std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition_table(
  table_view const& input,
  table_view const& table_to_hash,
  size_type num_partitions,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  auto const num_rows = table_to_hash.num_rows();

  bool const use_optimization{num_partitions <= THRESHOLD_FOR_OPTIMIZED_PARTITION_KERNEL};
  auto const block_size = use_optimization ? OPTIMIZED_BLOCK_SIZE : FALLBACK_BLOCK_SIZE;
  auto const rows_per_thread =
    use_optimization ? OPTIMIZED_ROWS_PER_THREAD : FALLBACK_ROWS_PER_THREAD;
  auto const rows_per_block = block_size * rows_per_thread;

  // NOTE grid_size is non-const to workaround lambda capture bug in gcc 5.4
  auto grid_size = util::div_rounding_up_safe(num_rows, rows_per_block);

  // Allocate array to hold which partition each row belongs to
  auto row_partition_numbers = rmm::device_vector<size_type>(num_rows);

  // Array to hold the size of each partition computed by each block
  //  i.e., { {block0 partition0 size, block1 partition0 size, ...},
  //          {block0 partition1 size, block1 partition1 size, ...},
  //          ...
  //          {block0 partition(num_partitions-1) size, block1
  //          partition(num_partitions -1) size, ...} }
  auto block_partition_sizes = rmm::device_vector<size_type>(grid_size * num_partitions);

  auto scanned_block_partition_sizes = rmm::device_vector<size_type>(grid_size * num_partitions);

  // Holds the total number of rows in each partition
  auto global_partition_sizes = rmm::device_vector<size_type>(num_partitions, size_type{0});

  auto row_partition_offset = rmm::device_vector<size_type>(num_rows);

  auto const device_input = table_device_view::create(table_to_hash, stream);
  auto const hasher       = row_hasher<MurmurHash3_32, hash_has_nulls>(*device_input);

  // If the number of partitions is a power of two, we can compute the partition
  // number of each row more efficiently with bitwise operations
  if (is_power_two(num_partitions)) {
    // Determines how the mapping between hash value and partition number is
    // computed
    using partitioner_type = bitwise_partitioner<hash_value_type>;

    // Computes which partition each row belongs to by hashing the row and
    // performing a partitioning operator on the hash value. Also computes the
    // number of rows in each partition both for each thread block as well as
    // across all blocks
    compute_row_partition_numbers<<<grid_size,
                                    block_size,
                                    num_partitions * sizeof(size_type),
                                    stream>>>(hasher,
                                              num_rows,
                                              num_partitions,
                                              partitioner_type(num_partitions),
                                              row_partition_numbers.data().get(),
                                              row_partition_offset.data().get(),
                                              block_partition_sizes.data().get(),
                                              global_partition_sizes.data().get());
  } else {
    // Determines how the mapping between hash value and partition number is
    // computed
    using partitioner_type = modulo_partitioner<hash_value_type>;

    // Computes which partition each row belongs to by hashing the row and
    // performing a partitioning operator on the hash value. Also computes the
    // number of rows in each partition both for each thread block as well as
    // across all blocks
    compute_row_partition_numbers<<<grid_size,
                                    block_size,
                                    num_partitions * sizeof(size_type),
                                    stream>>>(hasher,
                                              num_rows,
                                              num_partitions,
                                              partitioner_type(num_partitions),
                                              row_partition_numbers.data().get(),
                                              row_partition_offset.data().get(),
                                              block_partition_sizes.data().get(),
                                              global_partition_sizes.data().get());
  }

  // Compute exclusive scan of all blocks' partition sizes in-place to determine
  // the starting point for each blocks portion of each partition in the output
  thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                         block_partition_sizes.begin(),
                         block_partition_sizes.end(),
                         scanned_block_partition_sizes.data().get());

  // Compute exclusive scan of size of each partition to determine offset
  // location of each partition in final output.
  // TODO This can be done independently on a separate stream
  size_type* scanned_global_partition_sizes{global_partition_sizes.data().get()};
  thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                         global_partition_sizes.begin(),
                         global_partition_sizes.end(),
                         scanned_global_partition_sizes);

  // Copy the result of the exclusive scan to the output offsets array
  // to indicate the starting point for each partition in the output
  std::vector<size_type> partition_offsets(num_partitions);
  CUDA_TRY(cudaMemcpyAsync(partition_offsets.data(),
                           scanned_global_partition_sizes,
                           num_partitions * sizeof(size_type),
                           cudaMemcpyDeviceToHost,
                           stream));

  // When the number of partitions is less than a threshold, we can apply an
  // optimization using shared memory to copy values to the output buffer.
  // Otherwise, fallback to using scatter.
  if (use_optimization) {
    std::vector<std::unique_ptr<column>> output_cols(input.num_columns());

    // NOTE these pointers are non-const to workaround lambda capture bug in
    // gcc 5.4
    auto row_partition_numbers_ptr{row_partition_numbers.data().get()};
    auto row_partition_offset_ptr{row_partition_offset.data().get()};
    auto block_partition_sizes_ptr{block_partition_sizes.data().get()};
    auto scanned_block_partition_sizes_ptr{scanned_block_partition_sizes.data().get()};

    // Copy input to output by partition per column
    std::transform(input.begin(), input.end(), output_cols.begin(), [=](auto const& col) {
      return cudf::type_dispatcher(col.type(),
                                   copy_block_partitions_dispatcher{},
                                   col,
                                   num_partitions,
                                   row_partition_numbers_ptr,
                                   row_partition_offset_ptr,
                                   block_partition_sizes_ptr,
                                   scanned_block_partition_sizes_ptr,
                                   grid_size,
                                   mr,
                                   stream);
    });

    if (has_nulls(input)) {
      // Use copy_block_partitions to compute a gather map
      auto gather_map = compute_gather_map(num_rows,
                                           num_partitions,
                                           row_partition_numbers_ptr,
                                           row_partition_offset_ptr,
                                           block_partition_sizes_ptr,
                                           scanned_block_partition_sizes_ptr,
                                           grid_size,
                                           stream);

      // Handle bitmask using gather to take advantage of ballot_sync
      detail::gather_bitmask(
        input, gather_map.begin(), output_cols, detail::gather_bitmask_op::DONT_CHECK, mr, stream);
    }

    auto output{std::make_unique<table>(std::move(output_cols))};
    return std::make_pair(std::move(output), std::move(partition_offsets));
  } else {
    // Compute a scatter map from input to output such that the output rows are
    // sorted by partition number
    auto row_output_locations{row_partition_numbers.data().get()};
    auto scanned_block_partition_sizes_ptr{scanned_block_partition_sizes.data().get()};
    compute_row_output_locations<<<grid_size,
                                   block_size,
                                   num_partitions * sizeof(size_type),
                                   stream>>>(
      row_output_locations, num_rows, num_partitions, scanned_block_partition_sizes_ptr);

    // Use the resulting scatter map to materialize the output
    auto output = detail::scatter(
      input, row_partition_numbers.begin(), row_partition_numbers.end(), input, false, mr, stream);

    return std::make_pair(std::move(output), std::move(partition_offsets));
  }
}

struct dispatch_map_type {
  /**
   * @brief Partitions the table `t` according to the `partition_map`.
   *
   * Algorithm:
   * - Compute the histogram of the size each partition
   * - Compute the exclusive scan of the histogram to get the offset for each
   * partition in the final partitioned output
   * - Use a transform iterator to materialize the scatter map of the rows from
   * `t` into the final output.
   *
   * @note JH: It would likely be more efficient to avoid the atomic increments
   * in the transform iterator. It would probably be faster to compute a
   * per-thread block histogram and compute an exclusive scan of all of the
   * per-block histograms (like in hash partition). But I'm purposefully trying
   * to reduce memory pressure by avoiding intermediate materializations. Plus,
   * atomics resolve in L2 and should be pretty fast since all the offsets will
   * fit in L2.
   *
   */
  template <typename MapType>
  std::enable_if_t<is_index_type<MapType>(),
                   std::pair<std::unique_ptr<table>, std::vector<size_type>>>
  operator()(table_view const& t,
             column_view const& partition_map,
             size_type num_partitions,
             rmm::mr::device_memory_resource* mr,
             cudaStream_t stream) const
  {
    // Build a histogram of the number of rows in each partition
    rmm::device_vector<size_type> histogram(num_partitions + 1);
    std::size_t temp_storage_bytes{};
    std::size_t const num_levels = num_partitions + 1;
    size_type const lower_level  = 0;
    size_type const upper_level  = num_partitions;
    cub::DeviceHistogram::HistogramEven(nullptr,
                                        temp_storage_bytes,
                                        partition_map.begin<MapType>(),
                                        histogram.data().get(),
                                        num_levels,
                                        lower_level,
                                        upper_level,
                                        partition_map.size(),
                                        stream);

    rmm::device_buffer temp_storage(temp_storage_bytes, stream);

    cub::DeviceHistogram::HistogramEven(temp_storage.data(),
                                        temp_storage_bytes,
                                        partition_map.begin<MapType>(),
                                        histogram.data().get(),
                                        num_levels,
                                        lower_level,
                                        upper_level,
                                        partition_map.size(),
                                        stream);

    // `histogram` was created with an extra entry at the end such that an
    // exclusive scan will put the total number of rows at the end
    thrust::exclusive_scan(
      rmm::exec_policy()->on(stream), histogram.begin(), histogram.end(), histogram.begin());

    // Copy offsets to host
    std::vector<size_type> partition_offsets(histogram.size());
    thrust::copy(histogram.begin(), histogram.end(), partition_offsets.begin());

    // Unfortunately need to materialize the scatter map because
    // `detail::scatter` requires multiple passes through the iterator
    rmm::device_vector<MapType> scatter_map(partition_map.size());

    // For each `partition_map[i]`, atomically increment the corresponding
    // partition offset to determine `i`s location in the output
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      partition_map.begin<MapType>(),
                      partition_map.end<MapType>(),
                      scatter_map.begin(),
                      [offsets = histogram.data().get()] __device__(auto partition_number) {
                        return atomicAdd(&offsets[partition_number], 1);
                      });

    // Scatter the rows into their partitions
    auto scattered =
      cudf::detail::scatter(t, scatter_map.begin(), scatter_map.end(), t, false, mr, stream);

    return std::make_pair(std::move(scattered), std::move(partition_offsets));
  }

  template <typename MapType>
  std::enable_if_t<not is_index_type<MapType>(),
                   std::pair<std::unique_ptr<table>, std::vector<size_type>>>
  operator()(table_view const& t,
             column_view const& partition_map,
             size_type num_partitions,
             rmm::mr::device_memory_resource* mr,
             cudaStream_t stream) const
  {
    CUDF_FAIL("Unexpected, non-integral partition map.");
  }
};
}  // namespace

namespace detail {
namespace local {
std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition(
  table_view const& input,
  std::vector<size_type> const& columns_to_hash,
  int num_partitions,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream = 0)
{
  auto table_to_hash = input.select(columns_to_hash);

  // Return empty result if there are no partitions or nothing to hash
  if (num_partitions <= 0 || input.num_rows() == 0 || table_to_hash.num_columns() == 0) {
    return std::make_pair(empty_like(input), std::vector<size_type>{});
  }

  if (has_nulls(table_to_hash)) {
    return hash_partition_table<true>(input, table_to_hash, num_partitions, mr, stream);
  } else {
    return hash_partition_table<false>(input, table_to_hash, num_partitions, mr, stream);
  }
}
}  // namespace local

std::pair<std::unique_ptr<table>, std::vector<size_type>> partition(
  table_view const& t,
  column_view const& partition_map,
  size_type num_partitions,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream = 0)
{
  CUDF_EXPECTS(t.num_rows() == partition_map.size(),
               "Size mismatch between table and partition map.");
  CUDF_EXPECTS(not partition_map.has_nulls(), "Unexpected null values in partition_map.");

  if (num_partitions == 0 or t.num_rows() == 0) {
    return std::make_pair(empty_like(t), std::vector<size_type>{});
  }

  return cudf::type_dispatcher(
    partition_map.type(), dispatch_map_type{}, t, partition_map, num_partitions, mr, stream);
}
}  // namespace detail

// Partition based on hash values
std::pair<std::unique_ptr<table>, std::vector<size_type>> hash_partition(
  table_view const& input,
  std::vector<size_type> const& columns_to_hash,
  int num_partitions,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::local::hash_partition(input, columns_to_hash, num_partitions, mr);
}

// Partition based on an explicit partition map
std::pair<std::unique_ptr<table>, std::vector<size_type>> partition(
  table_view const& t,
  column_view const& partition_map,
  size_type num_partitions,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::partition(t, partition_map, num_partitions, mr);
}

}  // namespace cudf
