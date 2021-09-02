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
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <structs/utilities.hpp>

namespace cudf {
namespace groupby {
namespace detail {
namespace {
template <bool has_nulls>
std::unique_ptr<column> generate_ranks(table_view const& order_by,
                                       device_span<size_type const> group_labels,
                                       device_span<size_type const> group_offsets,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto const flattener = cudf::structs::detail::flatten_nested_columns(
    order_by, {}, {}, structs::detail::column_nullability::MATCH_INCOMING);
  auto const d_flat_order = table_device_view::create(std::get<0>(flattener), stream);
  row_equality_comparator<has_nulls> comparator(*d_flat_order, *d_flat_order, true);
  auto ranks = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, order_by.num_rows(), mask_state::UNALLOCATED, stream, mr);
  auto mutable_ranks = ranks->mutable_view();

  thrust::tabulate(rmm::exec_policy(stream),
                   mutable_ranks.begin<size_type>(),
                   mutable_ranks.end<size_type>(),
                   [comparator,
                    labels  = group_labels.data(),
                    offsets = group_offsets.data()] __device__(size_type row_index) {
                     auto group_start = offsets[labels[row_index]];
                     return row_index != group_start && comparator(row_index, row_index - 1)
                              ? 0
                              : row_index - group_start + 1;
                   });

  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                group_labels.begin(),
                                group_labels.end(),
                                mutable_ranks.begin<size_type>(),
                                mutable_ranks.begin<size_type>(),
                                thrust::equal_to<size_type>{},
                                DeviceMax{});

  return ranks;
}
}  // namespace
std::unique_ptr<column> rank_scan(column_view const& order_by,
                                  device_span<size_type const> group_labels,
                                  device_span<size_type const> group_offsets,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  auto const superimposed = structs::detail::superimpose_parent_nulls(order_by, stream, mr);
  table_view const order_table{{std::get<0>(superimposed)}};
  if (has_nested_nulls(table_view{{order_by}})) {
    return generate_ranks<true>(order_table, group_labels, group_offsets, stream, mr);
  }
  return generate_ranks<false>(order_table, group_labels, group_offsets, stream, mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
