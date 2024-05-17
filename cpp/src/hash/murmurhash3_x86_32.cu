/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/tabulate.h>

namespace cudf {
namespace hashing {
namespace detail {

std::unique_ptr<column> murmurhash3_x86_32(table_view const& input,
                                           uint32_t seed,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  auto output = make_numeric_column(data_type(type_to_id<hash_value_type>()),
                                    input.num_rows(),
                                    mask_state::UNALLOCATED,
                                    stream,
                                    mr);

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  bool const nullable   = has_nulls(input);
  auto const row_hasher = cudf::experimental::row::hash::row_hasher(input, stream);
  auto output_view      = output->mutable_view();

  // Compute the hash value for each row
  thrust::tabulate(rmm::exec_policy(stream),
                   output_view.begin<hash_value_type>(),
                   output_view.end<hash_value_type>(),
                   row_hasher.device_hasher<cudf::experimental::type_identity, MurmurHash3_x86_32>(
                     nullable, seed));

  return output;
}

}  // namespace detail

std::unique_ptr<column> murmurhash3_x86_32(table_view const& input,
                                           uint32_t seed,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::murmurhash3_x86_32(input, seed, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
