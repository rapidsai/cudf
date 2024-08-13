/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "groupby/sort/group_single_pass_reduction_util.cuh"

#include <cudf/detail/gather.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/replace.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/gather.h>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_argmin(column_view const& values,
                                     size_type num_groups,
                                     cudf::device_span<size_type const> group_labels,
                                     column_view const& key_sort_order,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  auto indices = type_dispatcher(values.type(),
                                 group_reduction_dispatcher<aggregation::ARGMIN>{},
                                 values,
                                 num_groups,
                                 group_labels,
                                 stream,
                                 mr);

  // // scalar value of ARGMIN_SENTINEL
  // auto sentinel = cudf::numeric_scalar<size_type>{cudf::detail::ARGMIN_SENTINEL};

  // auto new_indices = cudf::replace_nulls(*indices, sentinel, stream, mr);

  // // replace indices with ARGMIN_SENTINEL if bitmask is set
  auto indices_view = indices->mutable_view();

  thrust::gather_if(rmm::exec_policy(stream),
                    indices_view.begin<size_type>(),    // map first
                    indices_view.end<size_type>(),      // map last
                    indices_view.begin<size_type>(),    // stencil
                    key_sort_order.begin<size_type>(),  // input
                    indices_view.begin<size_type>(),    // result
                    [] __device__(auto i) {
                      if (i == cudf::detail::ARGMIN_SENTINEL) printf("not gather i: %d\n", i); 
                    return (i != cudf::detail::ARGMIN_SENTINEL); });
  return indices;
}

//   // new a mutable_view column with the same size as indices and fill it with ARGMIN_SENTINEL
//   auto result = make_numeric_column(data_type{type_to_id<size_type>()},
//                                     indices->size(),
//                                     mask_state::UNALLOCATED,
//                                     stream,
//                                     mr);
//   thrust::fill(rmm::exec_policy(stream),
//                result->mutable_view().begin<size_type>(),
//                result->mutable_view().end<size_type>(),
//                cudf::detail::ARGMIN_SENTINEL);

//   // get the null mask of indices
//   auto indices_view = indices->mutable_view();
//   auto indices_null_mask = cudf::detail::copy_bitmask(indices_view, stream, mr);
//   auto bitmask = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
//                                 indices->size(),
//                                 cudf::copy_bitmask(indices_view),
//                                 indices_view.null_count(),
//                                 stream);
  
//   // The functor returns the index of minimum in the sorted values.
//   // We need the index of minimum in the original unsorted values.
//   // So use indices to gather the sort order used to sort `values`.
//   // The values in data buffer of indices corresponding to null values was
//   // initialized to ARGMIN_SENTINEL. Using gather_if.
//   // This can't use gather because nulls in gathered column will not store ARGMIN_SENTINEL.
//   thrust::gather_if(rmm::exec_policy(stream),
//                     indices_view.begin<size_type>(),    // map first
//                     indices_view.end<size_type>(),      // map last
//                     bitmask->view().begin<bool>(),      // stencil
//                     key_sort_order.begin<size_type>(),  // input
//                     result->mutable_view().begin<size_type>(),    // result
//                     [] __device__(bool i) { return (i != true); });

//   return result;
// }

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
