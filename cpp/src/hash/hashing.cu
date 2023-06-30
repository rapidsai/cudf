/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <cudf/detail/utilities/algorithm.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/execution_policy.h>
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

std::unique_ptr<column> hash(table_view const& input,
                             hash_id hash_function,
                             uint32_t seed,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  switch (hash_function) {
    case (hash_id::HASH_MURMUR3): return murmur_hash3_32(input, seed, stream, mr);
    case (hash_id::HASH_SPARK_MURMUR3): return spark_murmur_hash3_32(input, seed, stream, mr);
    case (hash_id::HASH_MD5): return md5_hash(input, stream, mr);
    default: CUDF_FAIL("Unsupported hash function.");
  }
}

}  // namespace detail

std::unique_ptr<column> hash(table_view const& input,
                             hash_id hash_function,
                             uint32_t seed,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::hash(input, hash_function, seed, stream, mr);
}

}  // namespace cudf
