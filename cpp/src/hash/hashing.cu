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

#include <hash/hash_functions.cuh>

#include <thrust/tabulate.h>

namespace cudf {

namespace {

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

template <template <typename> class hash_function>
std::vector<std::unique_ptr<experimental::table>>
hash_partition_table(table_view const& input,
                     table_view const &table_to_hash,
                     const cudf::size_type num_partitions) {
  std::vector<std::unique_ptr<experimental::table>> output(num_partitions);

  // TODO

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
          input, table_to_hash, num_partitions);
      break;
    case hash_func::IDENTITY:
      output = hash_partition_table<IdentityHash>(
          input, table_to_hash, num_partitions);
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
