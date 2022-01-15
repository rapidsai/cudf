/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/row_operator2.cuh>
#include <cudf/table/row_operator3.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <cudf/detail/structs/utilities.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cudf {
namespace detail {

/**
 * @copydoc
 * sorted_order(table_view&,std::vector<order>,std::vector<null_order>,rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <bool stable = false>
std::unique_ptr<column> sorted_order2(
  table_view input,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  if (input.num_rows() == 0 or input.num_columns() == 0) {
    return cudf::make_numeric_column(data_type(type_to_id<size_type>()), 0);
  }

  std::unique_ptr<column> sorted_indices = cudf::make_numeric_column(
    data_type(type_to_id<size_type>()), input.num_rows(), mask_state::UNALLOCATED, stream, mr);
  mutable_column_view mutable_indices_view = sorted_indices->mutable_view();
  thrust::sequence(rmm::exec_policy(stream),
                   mutable_indices_view.begin<size_type>(),
                   mutable_indices_view.end<size_type>(),
                   0);

  auto device_table = table_device_view::create(input, stream);
  // auto const comparator = row_lexicographic_comparator2<true>(*device_table, *device_table);
  auto const comparator =
    row_lexicographic_comparator3(nullate::DYNAMIC{true}, *device_table, *device_table);

  thrust::sort(rmm::exec_policy(stream),
               mutable_indices_view.begin<size_type>(),
               mutable_indices_view.end<size_type>(),
               comparator);
  // protection for temporary d_column_order and d_null_precedence
  stream.synchronize();

  return sorted_indices;
}

}  // namespace detail
}  // namespace cudf
