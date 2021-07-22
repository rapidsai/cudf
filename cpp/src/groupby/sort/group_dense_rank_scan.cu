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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/logical.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {
template <bool has_nested_nulls>
std::unique_ptr<column> generate_dense_ranks(column_view const& order_by,
                                             device_span<size_type const> group_labels,
                                             device_span<size_type const> group_offsets,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  auto const flat_order =
    order_by.type().id() == type_id::STRUCT
      ? table_view{std::vector<column_view>{order_by.child_begin(), order_by.child_end()}}
      : table_view{{order_by}};
  auto const d_flat_order = table_device_view::create(flat_order, stream);
  row_equality_comparator<has_nested_nulls> comparator(*d_flat_order, *d_flat_order, true);
  auto ranks = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, order_by.size(), mask_state::UNALLOCATED, stream, mr);
  auto mutable_ranks = ranks->mutable_view();

  if (order_by.type().id() == type_id::STRUCT && order_by.has_nulls()) {
    auto const d_col_order = column_device_view::create(order_by, stream);
    thrust::tabulate(rmm::exec_policy(stream),
                     mutable_ranks.begin<size_type>(),
                     mutable_ranks.end<size_type>(),
                     [comparator,
                      d_col_order = *d_col_order,
                      labels      = group_labels.data(),
                      offsets     = group_offsets.data()] __device__(size_type row_index) {
                       if (row_index == offsets[labels[row_index]]) { return true; }
                       bool const lhs_is_null{d_col_order.is_null(row_index)};
                       bool const rhs_is_null{d_col_order.is_null(row_index - 1)};
                       if (lhs_is_null && rhs_is_null) {
                         return false;
                       } else if (lhs_is_null != rhs_is_null) {
                         return true;
                       }
                       return !comparator(row_index, row_index - 1);
                     });

  } else {
    thrust::tabulate(
      rmm::exec_policy(stream),
      mutable_ranks.begin<size_type>(),
      mutable_ranks.end<size_type>(),
      [comparator, labels = group_labels.data(), offsets = group_offsets.data()] __device__(
        size_type row_index) {
        return row_index == offsets[labels[row_index]] || !comparator(row_index, row_index - 1);
      });
  }

  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                group_labels.begin(),
                                group_labels.end(),
                                mutable_ranks.begin<size_type>(),
                                mutable_ranks.begin<size_type>());
  return ranks;
}
}  // namespace
std::unique_ptr<column> dense_rank_scan(column_view const& order_by,
                                        device_span<size_type const> group_labels,
                                        device_span<size_type const> group_offsets,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  if ((order_by.type().id() == type_id::STRUCT &&
       has_nested_nulls(
         table_view{std::vector<column_view>{order_by.child_begin(), order_by.child_end()}})) ||
      (order_by.type().id() != type_id::STRUCT && order_by.has_nulls())) {
    return generate_dense_ranks<true>(order_by, group_labels, group_offsets, stream, mr);
  }
  return generate_dense_ranks<false>(order_by, group_labels, group_offsets, stream, mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
