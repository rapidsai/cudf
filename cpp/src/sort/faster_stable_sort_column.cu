/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include "faster_sort_column_impl.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/sequence.h>

namespace cudf {
namespace detail {

template <>
void faster_sorted_order<sort_method::STABLE>(column_view const& input,
                                              mutable_column_view& indices,
                                              bool ascending,
                                              rmm::cuda_stream_view stream)
{
  auto col_temp = column(input, stream);
  auto d_col    = col_temp.mutable_view();
  thrust::sequence(
    rmm::exec_policy_nosync(stream), indices.begin<size_type>(), indices.end<size_type>(), 0);
  auto dispatch_fn = faster_sorted_order_fn<sort_method::STABLE>{};
  cudf::type_dispatcher<dispatch_storage_type>(
    input.type(), dispatch_fn, d_col, indices, ascending, stream);
}

}  // namespace detail
}  // namespace cudf
