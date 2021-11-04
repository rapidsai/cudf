/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <groupby/sort/group_util.cuh>

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_argminmax_struct(aggregation::Kind K,
                                               column_view const& values,
                                               size_type num_groups,
                                               cudf::device_span<size_type const> group_labels,
                                               column_view const& key_sort_order,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(K == aggregation::ARGMIN || aggregation::ARGMAX,
               "Only groupby ARGMIN/ARGMAX are supported for STRUCT type.");

  auto result = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, num_groups, mask_state::UNALLOCATED, stream, mr);

  if (values.is_empty()) { return result; }

  // When finding ARGMIN, we need to consider nulls as larger than non-null elements.
  // Thing is opposite for ARGMAX.
  auto const null_precedence  = (K == aggregation::ARGMIN) ? null_order::AFTER : null_order::BEFORE;
  auto const flattened_values = structs::detail::flatten_nested_columns(
    table_view{{values}}, {}, std::vector<null_order>{null_precedence});
  auto const d_flattened_values_ptr = table_device_view::create(flattened_values, stream);
  auto const flattened_null_precedences =
    (K == aggregation::ARGMIN)
      ? cudf::detail::make_device_uvector_async(flattened_values.null_orders(), stream)
      : rmm::device_uvector<null_order>(0, stream);

  // Perform segmented reduction to find ARGMIN/ARGMAX.
  auto const do_reduction = [&](auto const& inp_iter, auto const& out_iter, auto const& binop) {
    thrust::reduce_by_key(rmm::exec_policy(stream),
                          group_labels.data(),
                          group_labels.data() + group_labels.size(),
                          inp_iter,
                          thrust::make_discard_iterator(),
                          out_iter,
                          thrust::equal_to<size_type>{},
                          binop);
  };

  auto const count_iter   = thrust::make_counting_iterator<size_type>(0);
  auto const result_begin = result->mutable_view().template begin<size_type>();
  if (values.has_nulls()) {
    auto const binop = row_arg_minmax_fn<true>(values.size(),
                                               *d_flattened_values_ptr,
                                               flattened_null_precedences.data(),
                                               K == aggregation::ARGMIN);
    do_reduction(count_iter, result_begin, binop);

    // Generate bitmask for the output by segmented reduction of the input bitmask.
    auto const d_values_ptr = column_device_view::create(values, stream);
    auto validity           = rmm::device_uvector<bool>(num_groups, stream);
    do_reduction(cudf::detail::make_validity_iterator(*d_values_ptr),
                 validity.begin(),
                 thrust::logical_or<bool>{});

    auto [null_mask, null_count] = cudf::detail::valid_if(
      validity.begin(), validity.end(), thrust::identity<bool>{}, stream, mr);
    result->set_null_mask(std::move(null_mask), null_count);
  } else {
    auto const binop = row_arg_minmax_fn<false>(values.size(),
                                                *d_flattened_values_ptr,
                                                flattened_null_precedences.data(),
                                                K == aggregation::ARGMIN);
    do_reduction(count_iter, result_begin, binop);
  }

  // result now stores the indices of minimum elements in the sorted values.
  // We need the indices of minimum elements in the original unsorted values.
  thrust::gather(rmm::exec_policy(stream),
                 result_begin,
                 result_begin + num_groups,
                 key_sort_order.template begin<size_type>(),
                 result_begin);

  return result;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
