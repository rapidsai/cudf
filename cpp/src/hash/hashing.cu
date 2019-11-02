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

#include <thrust/tabulate.h>

namespace cudf {

namespace {

/*template <template <typename> class hash_function>
struct row_hasher_initial_values {
  row_hasher_initial_values(device_table const& table_to_hash,
                            hash_value_type *initial_hash_values)
      : the_table{table_to_hash}, initial_hash_values(initial_hash_values) {}

  __device__ hash_value_type operator()(cudf::size_type row_index) const {
    return hash_row<true,hash_function>(the_table, row_index, initial_hash_values);
  }

  device_table the_table;
  hash_value_type *initial_hash_values{nullptr};
};

template <template <typename> class hash_function>
struct row_hasher {
  row_hasher(device_table const& table_to_hash) : the_table{table_to_hash} {}

  __device__ hash_value_type operator()(cudf::size_type row_index) const {
    return hash_row<true,hash_function>(the_table, row_index);
  }

  device_table the_table;
};*/

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
  // TODO
  return std::vector<std::unique_ptr<experimental::table>>{};
}

std::unique_ptr<column> hash(table_view const& input,
                             hash_func::Type hash,
                             std::vector<size_type> const& initial_hash_values,
                             rmm::mr::device_memory_resource* mr,
                             cudaStream_t stream)
{
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    return make_numeric_column(data_type(INT32), 0);
  }

  // Compute the hash value for each row depending on the specified hash function
  if (!initial_hash_values.empty()) {
    CUDF_EXPECTS(initial_hash_values.size() == input.num_columns(),
      "Expected same size of initial hash values as number of columns");

    switch (hash) {
      case hash_func::MURMUR3:
        /*thrust::tabulate(rmm::exec_policy(stream)->on(stream), row_hash_values,
                         row_hash_values + num_rows,
                         row_hasher<MurmurHash3_32>(*input_table));*/
        break;
      case hash_func::IDENTITY:
        /*thrust::tabulate(rmm::exec_policy(stream)->on(stream), row_hash_values,
                         row_hash_values + num_rows,
                         row_hasher<IdentityHash>(*input_table));*/
        break;
      default:
        CUDF_FAIL("Invalid hash function");
    }
  } else {
    switch (hash) {
      case hash_func::MURMUR3:
        /*thrust::tabulate(rmm::exec_policy(stream)->on(stream), row_hash_values,
                         row_hash_values + num_rows,
                         row_hasher_initial_values<MurmurHash3_32>(
                             *input_table, initial_hash_values));*/
        break;
      case hash_func::IDENTITY:
        /*thrust::tabulate(rmm::exec_policy(stream)->on(stream), row_hash_values,
                         row_hash_values + num_rows,
                         row_hasher_initial_values<IdentityHash>(
                             *input_table, initial_hash_values));*/
        break;
      default:
        CUDF_FAIL("Invalid hash function");
    }
  }

  // TODO
  return make_numeric_column(data_type(INT32), 0);
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
                             std::vector<size_type> const& initial_hash_values,
                             rmm::mr::device_memory_resource* mr)
{
  return detail::hash(input, hash, initial_hash_values, mr);
}

}  // namespace cudf
