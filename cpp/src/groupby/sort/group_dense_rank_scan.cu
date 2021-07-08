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

namespace cudf {
namespace groupby {
namespace detail {
namespace {
template <typename Comparator>
void generate_dense_rank_comparisons(mutable_column_view out,
                                     device_span<size_type const> group_labels,
                                     cudf::device_span<size_type const> group_offsets,
                                     Comparator comp,
                                     rmm::cuda_stream_view stream)
{
  thrust::tabulate(rmm::exec_policy(stream),
                   out.begin<size_type>(),
                   out.end<size_type>(),
                   [comp, labels = group_labels.data(), offsets = group_offsets.data()] __device__(
                     size_type row_index) {
                     return row_index == offsets[labels[row_index]] ||
                            !comp(row_index, row_index - 1);
                   });
}
}  // namespace
std::unique_ptr<column> dense_rank_scan(table_view const& order_by,
                                        cudf::device_span<size_type const> group_labels,
                                        cudf::device_span<size_type const> group_offsets,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  auto d_order_by    = table_device_view::create(order_by);
  auto ranks         = make_fixed_width_column(cudf::data_type{cudf::type_to_id<size_type>()},
                                       order_by.num_rows(),
                                       mask_state::ALL_VALID,
                                       stream,
                                       mr);
  auto mutable_ranks = ranks->mutable_view();

  if (has_nested_nulls(order_by)) {
    row_equality_comparator<true> row_comparator(*d_order_by, *d_order_by, true);
    generate_dense_rank_comparisons(
      mutable_ranks, group_labels, group_offsets, row_comparator, stream);
  } else {
    row_equality_comparator<false> row_comparator(*d_order_by, *d_order_by, true);
    generate_dense_rank_comparisons(
      mutable_ranks, group_labels, group_offsets, row_comparator, stream);
  }

  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                group_labels.begin(),
                                group_labels.end(),
                                mutable_ranks.begin<size_type>(),
                                mutable_ranks.begin<size_type>());
  return ranks;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
