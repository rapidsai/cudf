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
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/tabulate.h>

namespace cudf {
namespace detail {

std::unique_ptr<column> murmur_hash3_32(table_view const& input,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  // TODO this should be UINT32
  auto output = make_numeric_column(
    data_type(type_id::INT32), input.num_rows(), mask_state::UNALLOCATED, stream, mr);

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  bool const nullable     = has_nulls(input);
  auto const device_input = table_device_view::create(input, stream);
  auto output_view        = output->mutable_view();

  // Compute the hash value for each row
  thrust::tabulate(
    rmm::exec_policy(stream),
    output_view.begin<int32_t>(),
    output_view.end<int32_t>(),
    row_hasher<MurmurHash3_32, nullate::DYNAMIC>(nullate::DYNAMIC{nullable}, *device_input));

  return output;
}

}  // namespace detail
}  // namespace cudf
