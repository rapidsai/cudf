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
#include <cudf/copying.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/scan.h>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> minmax_scan_struct(aggregation::Kind K,
                                           column_view const& values,
                                           size_type num_groups,
                                           device_span<size_type const> group_labels,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  if (values.is_empty()) { return cudf::empty_like(values); }

  // When finding MIN, we need to consider nulls as larger than non-null elements.
  // Thing is opposite when finding MAX.
  auto const null_precedence  = (K == aggregation::MIN) ? null_order::AFTER : null_order::BEFORE;
  auto const flattened_values = structs::detail::flatten_nested_columns(
    table_view{{values}}, {}, std::vector<null_order>{null_precedence});
  auto const d_flattened_values_ptr = table_device_view::create(flattened_values, stream);
  auto const flattened_null_precedences =
    (K == aggregation::MIN)
      ? cudf::detail::make_device_uvector_async(flattened_values.null_orders(), stream)
      : rmm::device_uvector<null_order>(0, stream);

  // Create a gather map contaning indices of the prefix min/max elements.
  auto gather_map      = rmm::device_uvector<size_type>(values.size(), stream);
  auto const map_begin = gather_map.begin();

  // Perform segmented scan.
  auto const do_scan = [&](auto const& inp_iter, auto const& out_iter, auto const& binop) {
    thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                  group_labels.begin(),
                                  group_labels.end(),
                                  inp_iter,
                                  out_iter,
                                  thrust::equal_to<size_type>{},
                                  binop);
  };

  // Find the indices of the prefix min/max elements within each group.
  auto const count_iter = thrust::make_counting_iterator<size_type>(0);
  if (values.has_nulls()) {
    auto const binop = row_arg_minmax_fn<true>(values.size(),
                                               *d_flattened_values_ptr,
                                               flattened_null_precedences.data(),
                                               K == aggregation::MIN);
    do_scan(count_iter, map_begin, binop);
  } else {
    auto const binop = row_arg_minmax_fn<false>(values.size(),
                                                *d_flattened_values_ptr,
                                                flattened_null_precedences.data(),
                                                K == aggregation::MIN);
    do_scan(count_iter, map_begin, binop);
  }

  auto gather_map_view =
    column_view(data_type{type_to_id<offset_type>()}, gather_map.size(), gather_map.data());

  // Gather the children elements of the prefix min/max struct elements first.
  auto scanned_children =
    cudf::detail::gather(
      table_view(std::vector<column_view>{values.child_begin(), values.child_end()}),
      gather_map_view,
      cudf::out_of_bounds_policy::DONT_CHECK,
      cudf::detail::negative_index_policy::NOT_ALLOWED,
      stream,
      mr)
      ->release();

  // After gathering the children elements, we need to push down nulls from the root structs
  // column to them.
  if (values.has_nulls()) {
    for (std::unique_ptr<column>& child : scanned_children) {
      structs::detail::superimpose_parent_nulls(
        values.null_mask(), values.null_count(), *child, stream, mr);
    }
  }

  return make_structs_column(values.size(),
                             std::move(scanned_children),
                             values.null_count(),
                             cudf::detail::copy_bitmask(values, stream, mr));
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
