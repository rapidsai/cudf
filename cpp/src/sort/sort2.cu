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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
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
namespace experimental {

/**
 * @copydoc
 * sorted_order(table_view&,std::vector<order>,std::vector<null_order>,rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<column> sorted_order2(table_view input,
                                      std::vector<order> const& column_order,
                                      std::vector<null_order> const& null_precedence,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
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

  auto comp =
    cudf::experimental::row_lex_operator(input, input, column_order, null_precedence, stream);

  thrust::sort(rmm::exec_policy(stream),
               mutable_indices_view.begin<size_type>(),
               mutable_indices_view.end<size_type>(),
               comp.device_comparator<nullate::DYNAMIC>());
  // protection for temporary d_column_order and d_null_precedence
  stream.synchronize();

  return sorted_indices;
}

}  // namespace experimental
}  // namespace detail
}  // namespace cudf
