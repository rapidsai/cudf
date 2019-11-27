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
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/detail/scatter.hpp>

#include <thrust/tabulate.h>

namespace cudf {

namespace {

/** 
 * @brief  Functor to map a hash value to a particular 'bin' or partition number
 * that uses the modulo operation.
 */
template <typename hash_value_t>
struct modulo_partitioner
{
  modulo_partitioner(size_type num_partitions) : divisor{num_partitions} {}

  __host__ __device__
  size_type operator()(hash_value_t hash_value) const {
    return hash_value % divisor;
  }

  const size_type divisor;
};

template <typename T>
bool is_power_two(T number) {
  return (0 == (number & (number - 1)));
}

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
  bitwise_partitioner(size_type num_partitions) : divisor{(num_partitions - 1)} {
    assert(is_power_two(num_partitions));
  }

  __host__ __device__
  size_type operator()(hash_value_t hash_value) const {
    return hash_value & (divisor);
  }

  const size_type divisor;
};

/** 
 * @brief Computes which partition each row of a device_table will belong to based
 * on hashing each row, and applying a partition function to the hash value.
 */
template <typename hasher_type, typename partitioner_type>
__global__ 
void compute_row_partition_numbers(hasher_type hasher, 
                                   const size_type num_rows,
                                   const size_type num_partitions,
                                   const partitioner_type the_partitioner,
                                   size_type *row_partition_numbers)
{
  size_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  // Compute the hash value for each row, store it to the array of hash values
  // and compute the partition to which the hash value belongs
  while (row_number < num_rows) {
    const hash_value_type row_hash_value = hasher(row_number);

    const size_type partition_number = the_partitioner(row_hash_value);

    row_partition_numbers[row_number] = partition_number;

    row_number += blockDim.x * gridDim.x;
  }
}

template <bool has_nulls>
std::vector<std::unique_ptr<experimental::table>>
hash_partition_table(table_view const& input,
                     table_view const &table_to_hash,
                     const size_type num_partitions,
                     rmm::mr::device_memory_resource* mr,
                     cudaStream_t stream)
{
  constexpr size_type block_size = 256;
  constexpr size_type rows_per_thread = 1;
  constexpr size_type rows_per_block = block_size * rows_per_thread;

  const size_type num_rows = table_to_hash.num_rows();
  const size_type grid_size = util::div_rounding_up_safe(num_rows, rows_per_block);

  auto device_input = table_device_view::create(input, stream);
  auto row_partition_numbers = rmm::device_vector<size_type>(num_rows);
  auto block_partition_sizes = rmm::device_vector<size_type>(grid_size * num_partitions);
  auto global_partition_sizes = rmm::device_vector<size_type>(num_partitions);
  CUDA_TRY(cudaMemsetAsync(global_partition_sizes.data().get(), 0, num_partitions * sizeof(size_type), stream));

  auto hasher = experimental::row_hasher<MurmurHash3_32, has_nulls>(*device_input);

  // If the number of partitions is a power of two, we can compute the partition 
  // number of each row more efficiently with bitwise operations
  if (is_power_two(num_partitions)) {
    // Determines how the mapping between hash value and partition number is computed
    using partitioner_type = bitwise_partitioner<hash_value_type>;

    // Computes which partition each row belongs to by hashing the row and performing
    // a partitioning operator on the hash value. Also computes the number of
    // rows in each partition both for each thread block as well as across all blocks
    compute_row_partition_numbers
      <<<grid_size, block_size, num_partitions * sizeof(size_type), stream>>>(
        hasher, num_rows, num_partitions, partitioner_type(num_partitions),
        row_partition_numbers.data().get());

  } else {
    // Determines how the mapping between hash value and partition number is computed
    using partitioner_type = modulo_partitioner<hash_value_type>;

    // Computes which partition each row belongs to by hashing the row and performing
    // a partitioning operator on the hash value. Also computes the number of
    // rows in each partition both for each thread block as well as across all blocks
    compute_row_partition_numbers
      <<<grid_size, block_size, num_partitions * sizeof(size_type), stream>>>(
        hasher, num_rows, num_partitions, partitioner_type(num_partitions),
        row_partition_numbers.data().get());
  }

  auto partition_map = column_view(data_type(INT32), num_rows, row_partition_numbers.data().get());
  return experimental::detail::scatter_to_tables(input, partition_map, mr, stream);
}

}  // namespace

namespace detail {

std::vector<std::unique_ptr<experimental::table>>
hash_partition(table_view const& input,
               std::vector<size_type> const& columns_to_hash,
               int num_partitions,
               rmm::mr::device_memory_resource* mr,
               cudaStream_t stream)
{
  CUDF_EXPECTS(columns_to_hash.size() > 0, "Need at least one column to hash");
  CUDF_EXPECTS(num_partitions > 0, "Need at least one partition");

  auto table_to_hash = input.select(columns_to_hash);
  bool const nullable = has_nulls(table_to_hash);

  cudf::nvtx::range_push("CUDF_HASH_PARTITION", cudf::nvtx::PARTITION_COLOR);

  std::vector<std::unique_ptr<experimental::table>> output;
  if (nullable) {
    output = hash_partition_table<true>(
        input, table_to_hash, num_partitions, mr, stream);
  } else {
    output = hash_partition_table<false>(
        input, table_to_hash, num_partitions, mr, stream);
  }

  cudf::nvtx::range_pop();

  return output;
}

std::unique_ptr<column> hash(table_view const& input,
                             std::vector<uint32_t> const& initial_hash,
                             rmm::mr::device_memory_resource* mr,
                             cudaStream_t stream)
{
  // TODO this should be UINT32
  auto output = make_numeric_column(data_type(INT32), input.num_rows());

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    return output;
  }

  bool const nullable = has_nulls(input);
  auto const device_input = table_device_view::create(input, stream);
  auto output_view = output->mutable_view();

  // Compute the hash value for each row depending on the specified hash function
  if (!initial_hash.empty()) {
    CUDF_EXPECTS(initial_hash.size() == size_t(input.num_columns()),
      "Expected same size of initial hash values as number of columns");
    auto device_initial_hash = rmm::device_vector<uint32_t>(initial_hash);

    if (nullable) {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
          output_view.begin<int32_t>(), output_view.end<int32_t>(),
          experimental::row_hasher_initial_values<MurmurHash3_32, true>(
              *device_input, device_initial_hash.data().get()));
    } else {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
          output_view.begin<int32_t>(), output_view.end<int32_t>(),
          experimental::row_hasher_initial_values<MurmurHash3_32, false>(
              *device_input, device_initial_hash.data().get()));
    }
  } else {
    if (nullable) {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
          output_view.begin<int32_t>(), output_view.end<int32_t>(),
          experimental::row_hasher<MurmurHash3_32, true>(*device_input));
    } else {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
          output_view.begin<int32_t>(), output_view.end<int32_t>(),
          experimental::row_hasher<MurmurHash3_32, false>(*device_input));
    }
  }

  return output;
}

}  // namespace detail

std::vector<std::unique_ptr<experimental::table>>
hash_partition(table_view const& input,
               std::vector<size_type> const& columns_to_hash,
               int num_partitions,
               rmm::mr::device_memory_resource* mr)
{
  return detail::hash_partition(input, columns_to_hash, num_partitions, mr);
}

std::unique_ptr<column> hash(table_view const& input,
                             std::vector<uint32_t> const& initial_hash,
                             rmm::mr::device_memory_resource* mr)
{
  return detail::hash(input, initial_hash, mr);
}

}  // namespace cudf
