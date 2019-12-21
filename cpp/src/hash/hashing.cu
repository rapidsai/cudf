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
#include <cudf/detail/gather.cuh>

#include <thrust/tabulate.h>

namespace cudf {

namespace {

template <typename T>
bool is_power_two(T number) {
  return (0 == (number & (number - 1)));
}

template <bool has_nulls>
std::pair<std::unique_ptr<experimental::table>, std::vector<size_type>>
hash_partition_table(table_view const& input,
                     table_view const &table_to_hash,
                     const size_type num_partitions,
                     rmm::mr::device_memory_resource* mr,
                     cudaStream_t stream)
{
  auto const num_rows = table_to_hash.num_rows();
  auto row_partitions = rmm::device_vector<size_type>(num_rows);

  // Make an iterator to compute the hash over each row
  auto const device_input = table_device_view::create(input, stream);
  auto const hasher = experimental::row_hasher<MurmurHash3_32, has_nulls>(*device_input);
  auto const hash_iterator = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), hasher);

  // Compute the partition number by taking the modulo of the hash and the
  // number of partitions. When the number of partitions is a power of two,
  // the modulo can be implemented more efficiently as a bitwise and.
  if (is_power_two(num_partitions)) {
    thrust::transform(rmm::exec_policy(stream)->on(stream),
      hash_iterator, hash_iterator + num_rows, row_partitions.begin(),
      [mask = num_partitions - 1] __device__ (hash_value_type hash_value) {
        return hash_value & mask; // hash_value & (num_partitions - 1)
      });
  } else {
    thrust::transform(rmm::exec_policy(stream)->on(stream),
      hash_iterator, hash_iterator + num_rows, row_partitions.begin(),
      [num_partitions] __device__ (hash_value_type hash_value) {
        return hash_value % num_partitions;
      });
  }

  // Initialize gather maps and offsets to sequence
  auto gather_map = rmm::device_vector<size_type>(num_rows);
  auto d_offsets = rmm::device_vector<size_type>(num_rows);
  thrust::sequence(rmm::exec_policy(stream)->on(stream),
    gather_map.begin(), gather_map.end());
  thrust::sequence(rmm::exec_policy(stream)->on(stream),
    d_offsets.begin(), d_offsets.end());

  // Sort sequence using partition map as key to generate gather maps
  thrust::sort_by_key(rmm::exec_policy(stream)->on(stream),
    row_partitions.begin(), row_partitions.end(), gather_map.begin());

  // Reduce unique partitions to extract gather map offsets from sequence
  auto end = thrust::unique_by_key(rmm::exec_policy(stream)->on(stream),
    row_partitions.begin(), row_partitions.end(), d_offsets.begin());

  auto output = experimental::detail::gather(input, gather_map.begin(), gather_map.end(),
    false, false, false, mr, stream);

  thrust::host_vector<size_type> unique_partitions(row_partitions.begin(), end.first);
  thrust::host_vector<size_type> unique_offsets(d_offsets.begin(), end.second);

  std::vector<size_type> offsets(num_partitions, input.num_rows());

  // Pad offsets to insert empty partitions absent from unique_partitions
  size_type next_partition = 0;
  for (size_t i = 0; i < unique_partitions.size(); ++i) {
    auto const partition = next_partition;
    next_partition = unique_partitions[i] + 1;
    auto const count = next_partition - partition;
    auto const offset = unique_offsets[i];
    std::fill_n(&offsets[partition], count, offset);
  }

  return std::make_pair(std::move(output), std::move(offsets));
}

struct nvtx_raii {
  nvtx_raii(char const* name, nvtx::color color) { nvtx::range_push(name, color); }
  ~nvtx_raii() { nvtx::range_pop(); }
};

}  // namespace

namespace detail {

std::pair<std::unique_ptr<experimental::table>, std::vector<size_type>>
hash_partition(table_view const& input,
               std::vector<size_type> const& columns_to_hash,
               int num_partitions,
               rmm::mr::device_memory_resource* mr,
               cudaStream_t stream)
{
  // Push/pop nvtx range around the scope of this function
  nvtx_raii("CUDF_HASH_PARTITION", nvtx::PARTITION_COLOR);

  auto table_to_hash = input.select(columns_to_hash);

  // Return empty result if there are no partitions or nothing to hash
  if (num_partitions <= 0 || input.num_rows() == 0 || table_to_hash.num_columns() == 0) {
    return std::make_pair(experimental::empty_like(input), std::vector<size_type>{});
  }

  if (has_nulls(table_to_hash)) {
    return hash_partition_table<true>(
        input, table_to_hash, num_partitions, mr, stream);
  } else {
    return hash_partition_table<false>(
        input, table_to_hash, num_partitions, mr, stream);
  }
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

std::pair<std::unique_ptr<experimental::table>, std::vector<size_type>>
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
