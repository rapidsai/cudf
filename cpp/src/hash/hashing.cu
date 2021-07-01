/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/hashing.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/tabulate.h>

#include <algorithm>

namespace cudf {
namespace detail {
namespace {

template <typename IterType>
std::vector<column_view> to_leaf_columns(IterType iter_begin, IterType iter_end)
{
  std::vector<column_view> leaf_columns;
  std::for_each(iter_begin, iter_end, [&leaf_columns](column_view const& col) {
    if (is_nested(col.type())) {
      CUDF_EXPECTS(col.type().id() == type_id::STRUCT, "unsupported nested type");
      auto child_columns = to_leaf_columns(col.child_begin(), col.child_end());
      leaf_columns.insert(leaf_columns.end(), child_columns.begin(), child_columns.end());
    } else {
      leaf_columns.emplace_back(col);
    }
  });
  return leaf_columns;
}

}  // namespace

template <template <typename> class hash_function>
std::unique_ptr<column> serial_murmur_hash3_32(table_view const& input,
                                               uint32_t seed,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  auto output = make_numeric_column(
    data_type(type_id::INT32), input.num_rows(), mask_state::UNALLOCATED, stream, mr);

  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  table_view const leaf_table(to_leaf_columns(input.begin(), input.end()));
  auto const device_input = table_device_view::create(leaf_table, stream);
  auto output_view        = output->mutable_view();

  if (has_nulls(leaf_table)) {
    thrust::tabulate(rmm::exec_policy(stream),
                     output_view.begin<int32_t>(),
                     output_view.end<int32_t>(),
                     [device_input = *device_input, seed] __device__(auto row_index) {
                       return thrust::reduce(
                         thrust::seq,
                         device_input.begin(),
                         device_input.end(),
                         seed,
                         [rindex = row_index] __device__(auto hash, auto column) {
                           return cudf::type_dispatcher(
                             column.type(),
                             element_hasher_with_seed<hash_function, true>{hash, hash},
                             column,
                             rindex);
                         });
                     });
  } else {
    thrust::tabulate(rmm::exec_policy(stream),
                     output_view.begin<int32_t>(),
                     output_view.end<int32_t>(),
                     [device_input = *device_input, seed] __device__(auto row_index) {
                       return thrust::reduce(
                         thrust::seq,
                         device_input.begin(),
                         device_input.end(),
                         seed,
                         [rindex = row_index] __device__(auto hash, auto column) {
                           return cudf::type_dispatcher(
                             column.type(),
                             element_hasher_with_seed<hash_function, false>{hash, hash},
                             column,
                             rindex);
                         });
                     });
  }

  return output;
}

std::unique_ptr<column> hash(table_view const& input,
                             hash_id hash_function,
                             cudf::host_span<uint32_t const> initial_hash,
                             uint32_t seed,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  switch (hash_function) {
    case (hash_id::HASH_MURMUR3): return murmur_hash3_32(input, initial_hash, stream, mr);
    case (hash_id::HASH_MD5): return md5_hash(input, stream, mr);
    case (hash_id::HASH_SERIAL_MURMUR3):
      return serial_murmur_hash3_32<MurmurHash3_32>(input, seed, stream, mr);
    case (hash_id::HASH_SPARK_MURMUR3):
      return serial_murmur_hash3_32<SparkMurmurHash3_32>(input, seed, stream, mr);
    default: return nullptr;
  }
}

}  // namespace detail

std::unique_ptr<column> hash(table_view const& input,
                             hash_id hash_function,
                             cudf::host_span<uint32_t const> initial_hash,
                             uint32_t seed,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::hash(input, hash_function, initial_hash, seed, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
