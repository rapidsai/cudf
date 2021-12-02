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
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace groupby {
namespace detail {
namespace {
/**
 * @brief generate grouped row ranks or dense ranks using a row comparison then scan the results
 *
 * @tparam has_nulls if the order_by column has nulls
 * @tparam value_resolver flag value resolver function with boolean first and row number arguments
 * @tparam scan_operator scan function ran on the flag values
 * @param order_by input column to generate ranks for
 * @param group_labels ID of group that the corresponding value belongs to
 * @param group_offsets group index offsets with group ID indices
 * @param resolver flag value resolver
 * @param scan_op scan operation ran on the flag results
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return std::unique_ptr<column> rank values
 */
template <bool has_nulls, typename value_resolver, typename scan_operator>
std::unique_ptr<column> rank_generator(column_view const& order_by,
                                       device_span<size_type const> group_labels,
                                       device_span<size_type const> group_offsets,
                                       value_resolver resolver,
                                       scan_operator scan_op,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto const flattened = cudf::structs::detail::flatten_nested_columns(
    table_view{{order_by}}, {}, {}, structs::detail::column_nullability::MATCH_INCOMING);
  auto const d_flat_order = table_device_view::create(flattened, stream);
  row_equality_comparator<has_nulls> comparator(*d_flat_order, *d_flat_order, true);
  auto ranks         = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                       flattened.flattened_columns().num_rows(),
                                       mask_state::UNALLOCATED,
                                       stream,
                                       mr);
  auto mutable_ranks = ranks->mutable_view();

  thrust::tabulate(
    rmm::exec_policy(stream),
    mutable_ranks.begin<size_type>(),
    mutable_ranks.end<size_type>(),
    [comparator, resolver, labels = group_labels.data(), offsets = group_offsets.data()] __device__(
      size_type row_index) {
      auto group_start = offsets[labels[row_index]];
      return resolver(row_index == group_start || !comparator(row_index, row_index - 1),
                      row_index - group_start);
    });

  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                group_labels.begin(),
                                group_labels.end(),
                                mutable_ranks.begin<size_type>(),
                                mutable_ranks.begin<size_type>(),
                                thrust::equal_to{},
                                scan_op);

  return ranks;
}
}  // namespace

std::unique_ptr<column> rank_scan(column_view const& order_by,
                                  device_span<size_type const> group_labels,
                                  device_span<size_type const> group_offsets,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  if (has_nested_nulls(table_view{{order_by}})) {
    return rank_generator<true>(
      order_by,
      group_labels,
      group_offsets,
      [] __device__(bool equality, auto row_index) { return equality ? row_index + 1 : 0; },
      DeviceMax{},
      stream,
      mr);
  }
  return rank_generator<false>(
    order_by,
    group_labels,
    group_offsets,
    [] __device__(bool equality, auto row_index) { return equality ? row_index + 1 : 0; },
    DeviceMax{},
    stream,
    mr);
}

std::unique_ptr<column> dense_rank_scan(column_view const& order_by,
                                        device_span<size_type const> group_labels,
                                        device_span<size_type const> group_offsets,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  if (has_nested_nulls(table_view{{order_by}})) {
    return rank_generator<true>(
      order_by,
      group_labels,
      group_offsets,
      [] __device__(bool equality, auto row_index) { return equality; },
      DeviceSum{},
      stream,
      mr);
  }
  return rank_generator<false>(
    order_by,
    group_labels,
    group_offsets,
    [] __device__(bool equality, auto row_index) { return equality; },
    DeviceSum{},
    stream,
    mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
