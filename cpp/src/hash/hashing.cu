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
#include <cudf/copying.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/nvtx_utils.hpp>
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
class modulo_partitioner
{
 public:
  modulo_partitioner(size_type num_partitions) : divisor{num_partitions} {}

  __device__
  size_type operator()(hash_value_t hash_value) const {
    return hash_value % divisor;
  }

 private:
  const size_type divisor;
};

template <typename T>
bool is_power_two(T number) {
  return (0 == (number & (number - 1)));
}

/** 
 * @brief  Functor to map a hash value to a particular 'bin' or partition number
 * that uses a bitwise mask. Only works when num_partitions is a power of 2.
 *
 * For n % d, if d is a power of two, then it can be computed more efficiently via 
 * a single bitwise AND as:
 * n & (d - 1)
 */
template <typename hash_value_t>
class bitwise_partitioner
{
 public:
  bitwise_partitioner(size_type num_partitions) : mask{(num_partitions - 1)} {
    assert(is_power_two(num_partitions));
  }

  __device__
  size_type operator()(hash_value_t hash_value) const {
    return hash_value & mask; // hash_value & (num_partitions - 1)
  }

 private:
  const size_type mask;
};

template <bool has_nulls>
std::vector<std::unique_ptr<experimental::table>>
hash_partition_table(table_view const& input,
                     table_view const &table_to_hash,
                     const size_type num_partitions,
                     rmm::mr::device_memory_resource* mr,
                     cudaStream_t stream)
{
  auto const num_rows = table_to_hash.num_rows();
  auto row_partition_numbers = rmm::device_vector<size_type>(num_rows);

  // Make an iterator to compute the hash over each row
  auto const device_input = table_device_view::create(table_to_hash, stream);
  auto const hasher = experimental::row_hasher<MurmurHash3_32, has_nulls>(*device_input);
  auto const hash_iterator = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), hasher);

  // If the number of partitions is a power of two, we can compute the partition 
  // number of each row more efficiently with bitwise operations
  if (is_power_two(num_partitions)) {
    // Compute which partition each row belongs to using bitwise partitioner
    auto partitioner = bitwise_partitioner<hash_value_type>{num_partitions};
    thrust::transform(rmm::exec_policy(stream)->on(stream), hash_iterator,
      hash_iterator + num_rows, row_partition_numbers.begin(), partitioner);

  } else {
    // Compute which partition each row belongs to using modulo partitioner
    auto partitioner = modulo_partitioner<hash_value_type>{num_partitions};
    thrust::transform(rmm::exec_policy(stream)->on(stream), hash_iterator,
      hash_iterator + num_rows, row_partition_numbers.begin(), partitioner);
  }

  // Scatter input rows to output partitions given the partition map
  auto const partition_map = column_view(data_type(INT32), num_rows,
    row_partition_numbers.data().get());
  auto output = experimental::detail::scatter_to_tables(input, partition_map, mr, stream);

  // Pad with empty tables if we have less than num_partitions
  if (output.size() < static_cast<size_t>(num_partitions)) {
    std::generate_n(std::back_inserter(output), num_partitions - output.size(),
      [&input]() { return experimental::empty_like(input); });
  }

  return output;
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
  std::vector<std::unique_ptr<experimental::table>> output;
  auto table_to_hash = input.select(columns_to_hash);

  // Return empty vector if there are no partitions or anything to hash
  if (num_partitions <= 0 || input.num_rows() == 0 || table_to_hash.num_columns() == 0) {
    return output;
  }

  cudf::nvtx::range_push("CUDF_HASH_PARTITION", cudf::nvtx::PARTITION_COLOR);

  if (has_nulls(table_to_hash)) {
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
