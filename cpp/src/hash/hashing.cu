/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/detail/hashing.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/nvtx_utils.hpp>
#include <cudf/detail/utilities/cuda.cuh>

#include <hash/hash_functions.cuh>

#include <thrust/tabulate.h>

namespace cudf {

namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int ROWS_PER_THREAD = 1;

template <template <typename> class hash_function>
struct row_hasher_initial_values {
  row_hasher_initial_values(table_device_view const& table_to_hash,
                            hash_value_type *initial_hash)
      : _table{table_to_hash}, _initial_hash(initial_hash) {}

  __device__ hash_value_type operator()(cudf::size_type row_index) const {
    return 0; // TODO
    //return hash_row<true, hash_function>(_table, row_index, _initial_hash);
  }

  table_device_view const& _table;
  hash_value_type *_initial_hash{nullptr};
};

template <template <typename> class hash_function>
struct row_hasher {
  row_hasher(table_device_view const& table_to_hash) : _table{table_to_hash} {}

  __device__ hash_value_type operator()(cudf::size_type row_index) const {
    return 0; // TODO
    //return hash_row<true, hash_function>(_table, row_index);
  }

  table_device_view const& _table;
};

/** 
 * @brief  Functor to map a hash value to a particular 'bin' or partition number
 * that uses the modulo operation.
 */
template <typename hash_value_t>
struct modulo_partitioner
{
  modulo_partitioner(size_type num_partitions) : divisor{num_partitions}{}

  __host__ __device__
  size_type operator()(hash_value_t hash_value) const 
  {
    return hash_value % divisor;
  }

  const size_type divisor;
};

/** 
 * @brief  Functor to map a hash value to a particular 'bin' or partition number
 * that uses bitshifts. Only works when num_partitions is a power of 2.
 *
 * For n % d, if d is a power of two, then it can be computed more efficiently via 
 * a single bitwise AND as:
 * n & (d - 1)
 */
template <typename hash_value_t>
struct bitwise_partitioner
{
  bitwise_partitioner(size_type num_partitions) : divisor{(num_partitions - 1)}
  {
    assert( is_power_two(num_partitions) );
  }

  __host__ __device__
  size_type operator()(hash_value_t hash_value) const 
  {
    return hash_value & (divisor);
  }

  const size_type divisor;
};

/** 
 * @brief Computes which partition each row of a device_table will belong to based
   on hashing each row, and applying a partition function to the hash value. 
   Records the size of each partition for each thread block as well as the global
   size of each partition across all thread blocks.
 */
template <template <typename> class hash_function,
          typename partitioner_type>
__global__ 
void compute_row_partition_numbers(table_device_view the_table, 
                                   const size_type num_rows,
                                   const size_type num_partitions,
                                   const partitioner_type the_partitioner,
                                   size_type *row_partition_numbers,
                                   size_type *block_partition_sizes,
                                   size_type *global_partition_sizes)
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
    const hash_value_type row_hash_value =
        0; // TODO
        //hash_row<true, hash_function>(the_table, row_number);

    const size_type partition_number = the_partitioner(row_hash_value);

    row_partition_numbers[row_number] = partition_number;

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
    const size_type write_location = partition_number * gridDim.x + blockIdx.x;
    block_partition_sizes[write_location] = block_partition_size;
    partition_number += blockDim.x;
  }
}

template <template <typename> class hash_function>
std::vector<std::unique_ptr<experimental::table>>
hash_partition_table(table_view const& input,
                     table_view const &table_to_hash,
                     const cudf::size_type num_partitions,
                     cudaStream_t stream)
{
  const cudf::size_type num_rows = table_to_hash.num_rows();
  constexpr cudf::size_type rows_per_block = BLOCK_SIZE * ROWS_PER_THREAD;
  const cudf::size_type grid_size = util::div_rounding_up_safe(num_rows, rows_per_block);

  auto device_input = table_device_view::create(input, stream);
  auto row_partition_numbers = rmm::device_vector<size_type>(num_rows);
  auto block_partition_sizes = rmm::device_vector<size_type>(grid_size * num_partitions);
  auto global_partition_sizes = rmm::device_vector<size_type>(num_partitions);
  CUDA_TRY(cudaMemsetAsync(global_partition_sizes.data().get(), 0, num_partitions * sizeof(size_type), stream));

  // If the number of partitions is a power of two, we can compute the partition 
  // number of each row more efficiently with bitwise operations
  if (is_power_two(num_partitions)) {
    // Determines how the mapping between hash value and partition number is computed
    using partitioner_type = bitwise_partitioner<hash_value_type>;

    // Computes which partition each row belongs to by hashing the row and performing
    // a partitioning operator on the hash value. Also computes the number of
    // rows in each partition both for each thread block as well as across all blocks
    compute_row_partition_numbers<hash_function>
        <<<grid_size, BLOCK_SIZE, num_partitions * sizeof(cudf::size_type), stream>>>(
            *device_input, num_rows, num_partitions,
            partitioner_type(num_partitions), row_partition_numbers.data().get(),
            block_partition_sizes.data().get(), global_partition_sizes.data().get());

  } else {
    // Determines how the mapping between hash value and partition number is computed
    using partitioner_type = modulo_partitioner<hash_value_type>;

    // Computes which partition each row belongs to by hashing the row and performing
    // a partitioning operator on the hash value. Also computes the number of
    // rows in each partition both for each thread block as well as across all blocks
    compute_row_partition_numbers<hash_function>
        <<<grid_size, BLOCK_SIZE, num_partitions * sizeof(cudf::size_type), stream>>>(
            *device_input, num_rows, num_partitions,
            partitioner_type(num_partitions), row_partition_numbers.data().get(),
            block_partition_sizes.data().get(), global_partition_sizes.data().get());
  }

  // TODO

  // build output tables from partitioned row indices
  std::vector<std::unique_ptr<experimental::table>> output(num_partitions);

  return output;
}

}  // namespace

namespace detail {

std::vector<std::unique_ptr<experimental::table>>
hash_partition(table_view const& input,
               std::vector<size_type> const& columns_to_hash,
               int num_partitions,
               hash_func::Type hash,
               rmm::mr::device_memory_resource* mr,
               cudaStream_t stream)
{
  CUDF_EXPECTS(columns_to_hash.size() > 0, "Need at least one column to hash");
  CUDF_EXPECTS(num_partitions > 0, "Need at least one partition");

  auto table_to_hash = input.select(columns_to_hash);
  std::vector<std::unique_ptr<experimental::table>> output;

  cudf::nvtx::range_push("CUDF_HASH_PARTITION", cudf::nvtx::PARTITION_COLOR);

  switch (hash) {
    case hash_func::MURMUR3:
      output = hash_partition_table<MurmurHash3_32>(
          input, table_to_hash, num_partitions, stream);
      break;
    case hash_func::IDENTITY:
      output = hash_partition_table<IdentityHash>(
          input, table_to_hash, num_partitions, stream);
      break;
    default:
      CUDF_FAIL("Invalid hash function");
  }

  cudf::nvtx::range_pop();

  return output;
}

std::unique_ptr<column> hash(table_view const& input,
                             hash_func::Type hash,
                             std::vector<uint32_t> const& initial_hash,
                             rmm::mr::device_memory_resource* mr,
                             cudaStream_t stream)
{
  // TODO this should be UINT32
  auto hash_column = make_numeric_column(data_type(INT32), input.num_rows());
  auto hash_view = hash_column->mutable_view();

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    return hash_column;
  }

  auto device_input = table_device_view::create(input, stream);

  // Compute the hash value for each row depending on the specified hash function
  if (!initial_hash.empty()) {
    CUDF_EXPECTS(initial_hash.size() == input.num_columns(),
      "Expected same size of initial hash values as number of columns");
    auto device_initial_hash = rmm::device_vector<uint32_t>(initial_hash);

    switch (hash) {
      case hash_func::MURMUR3:
        thrust::tabulate(rmm::exec_policy(stream)->on(stream),
                         hash_view.begin<int32_t>(), hash_view.end<int32_t>(),
                         row_hasher_initial_values<MurmurHash3_32>(
                             *device_input, device_initial_hash.data().get()));
        break;
      case hash_func::IDENTITY:
        thrust::tabulate(rmm::exec_policy(stream)->on(stream),
                         hash_view.begin<int32_t>(), hash_view.end<int32_t>(),
                         row_hasher_initial_values<IdentityHash>(
                             *device_input, device_initial_hash.data().get()));
        break;
      default:
        CUDF_FAIL("Invalid hash function");
    }
  } else {
    switch (hash) {
      case hash_func::MURMUR3:
        thrust::tabulate(rmm::exec_policy(stream)->on(stream),
                         hash_view.begin<int32_t>(), hash_view.end<int32_t>(),
                         row_hasher<MurmurHash3_32>(*device_input));
        break;
      case hash_func::IDENTITY:
        thrust::tabulate(rmm::exec_policy(stream)->on(stream),
                         hash_view.begin<int32_t>(), hash_view.end<int32_t>(),
                         row_hasher<IdentityHash>(*device_input));
        break;
      default:
        CUDF_FAIL("Invalid hash function");
    }
  }

  return hash_column;
}

}  // namespace detail

std::vector<std::unique_ptr<experimental::table>>
hash_partition(table_view const& input,
               std::vector<size_type> const& columns_to_hash,
               int num_partitions,
               hash_func::Type hash,
               rmm::mr::device_memory_resource* mr)
{
  return detail::hash_partition(input, columns_to_hash, num_partitions, hash, mr);
}

std::unique_ptr<column> hash(table_view const& input,
                             hash_func::Type hash,
                             std::vector<uint32_t> const& initial_hash,
                             rmm::mr::device_memory_resource* mr)
{
  return detail::hash(input, hash, initial_hash, mr);
}

}  // namespace cudf
