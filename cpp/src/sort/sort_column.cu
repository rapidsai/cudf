/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "common_sort_impl.cuh"
#include "sort_column_impl.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/resource_ref.hpp>

#include <thrust/sequence.h>

namespace cudf {
namespace detail {

/**
 * @copydoc
 * sorted_order(column_view&,order,null_order,rmm::cuda_stream_view,rmm::device_async_resource_ref)
 */
template <>
std::unique_ptr<column> sorted_order<sort_method::UNSTABLE>(column_view const& input,
                                                            order column_order,
                                                            null_order null_precedence,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr)
{
  auto sorted_indices = cudf::make_numeric_column(
    data_type(type_to_id<size_type>()), input.size(), mask_state::UNALLOCATED, stream, mr);
  mutable_column_view indices_view = sorted_indices->mutable_view();
  thrust::sequence(
    rmm::exec_policy(stream), indices_view.begin<size_type>(), indices_view.end<size_type>(), 0);
  cudf::type_dispatcher<dispatch_storage_type>(input.type(),
                                               column_sorted_order_fn<sort_method::UNSTABLE>{},
                                               input,
                                               indices_view,
                                               column_order == order::ASCENDING,
                                               null_precedence,
                                               stream);
  return sorted_indices;
}

}  // namespace detail
}  // namespace cudf
