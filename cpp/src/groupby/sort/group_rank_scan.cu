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
#include "cudf/binaryop.hpp"
#include "cudf/column/column_device_view.cuh"
#include "cudf/table/table_device_view.cuh"
#include "cudf/table/table_view.hpp"
#include "thrust/logical.h"

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
  auto const superimposed = structs::detail::superimpose_parent_nulls(order_by, stream, mr);
  table_view const order_table{{std::get<0>(superimposed)}};
  auto const flattener = cudf::structs::detail::flatten_nested_columns(
    order_table, {}, {}, structs::detail::column_nullability::MATCH_INCOMING);
  auto const d_flat_order = table_device_view::create(std::get<0>(flattener), stream);
  row_equality_comparator<has_nulls> comparator(*d_flat_order, *d_flat_order, true);
  auto ranks         = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                       order_table.num_rows(),
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
                                thrust::equal_to<size_type>{},
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

template <bool has_nulls>
void equality_row(table_view lhs,
                  table_view rhs,
                  mutable_column_view out,
                  rmm::cuda_stream_view stream)
{
  auto d_lhs = table_device_view::create(lhs);
  auto d_rhs = table_device_view::create(rhs);

  row_equality_comparator<has_nulls> comparator(*d_lhs, *d_rhs, true);

  thrust::tabulate(
    rmm::exec_policy(stream),
    out.begin<size_type>(),
    out.end<size_type>(),
    [comparator] __device__(size_type row_index) { return comparator(row_index, row_index); });
}

template <bool has_nulls>
void nequal_row(table_view lhs,
                table_view rhs,
                mutable_column_view out,
                rmm::cuda_stream_view stream)
{
  auto d_lhs = table_device_view::create(lhs);
  auto d_rhs = table_device_view::create(rhs);

  row_equality_comparator<has_nulls> comparator(*d_lhs, *d_rhs, true);

  thrust::tabulate(
    rmm::exec_policy(stream),
    out.begin<size_type>(),
    out.end<size_type>(),
    [comparator] __device__(size_type row_index) { return !comparator(row_index, row_index); });
}

void and_merge(table_view lhs,
               table_view rhs,
               mutable_column_view out,
               binary_operator op,
               rmm::cuda_stream_view stream)
{
  std::vector<column_view> comp_views{};
  std::vector<std::unique_ptr<column>> child_comparisons{};
  std::for_each(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(lhs.num_columns()),
    [&](auto child_index) {
      auto res = binary_operation(
        lhs.column(child_index), rhs.column(child_index), binary_operator::LESS, out.type());
      comp_views.push_back(res->view());
      child_comparisons.push_back(std::move(res));

    });


  table_view comp_table{comp_views};
  auto const d_comp_table = table_device_view::create(comp_table);

  // merge
  thrust::tabulate(rmm::exec_policy(stream),
                   out.begin<size_type>(),
                   out.end<size_type>(),
                   [d_comp_table = *d_comp_table] __device__(size_type row_index) {
                     return thrust::all_of(thrust::seq,
                                           d_comp_table.begin(),
                                           d_comp_table.end(),
                                           [row_index] __device__(column_device_view col) {
                                             return col.data<bool>()[row_index];
                                             // return col.element(row_index);
                                           });
                   });
}

template <bool has_nulls>
void lt_row(table_view lhs, table_view rhs, mutable_column_view out, rmm::cuda_stream_view stream)
{
  auto d_lhs = table_device_view::create(lhs);
  auto d_rhs = table_device_view::create(rhs);

  row_lexicographic_comparator<has_nulls> comparator(*d_lhs, *d_rhs, nullptr, nullptr);

  thrust::tabulate(
    rmm::exec_policy(stream),
    out.begin<size_type>(),
    out.end<size_type>(),
    [comparator] __device__(size_type row_index) { return !comparator(row_index, row_index); });
}

std::unique_ptr<column> struct_binary_operation2(column_view const& lhs,
                                                 column_view const& rhs,
                                                 binary_operator op,
                                                 data_type output_type,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  auto const lhs_superimposed = structs::detail::superimpose_parent_nulls(lhs);
  auto const lhs_flattener    = cudf::structs::detail::flatten_nested_columns(
    table_view{{std::get<0>(lhs_superimposed)}},
    {},
    {},
    structs::detail::column_nullability::MATCH_INCOMING);
  table_view lhs_flat = std::get<0>(lhs_flattener);
  // auto const d_lhs_flat = table_device_view::create(lhs_flat, stream);

  auto const rhs_superimposed = structs::detail::superimpose_parent_nulls(rhs);
  auto const rhs_flattener    = cudf::structs::detail::flatten_nested_columns(
    table_view{{std::get<0>(rhs_superimposed)}},
    {},
    {},
    structs::detail::column_nullability::MATCH_INCOMING);
  table_view rhs_flat = std::get<0>(rhs_flattener);
  // auto const d_rhs_flat = table_device_view::create(rhs_flat, stream);

  auto out =
    cudf::detail::make_fixed_width_column_for_output(lhs, rhs, op, output_type, stream, mr);
  auto out_view = out->mutable_view();

  auto lhs_has_nulls = has_nested_nulls(lhs_flat);
  auto rhs_has_nulls = has_nested_nulls(rhs_flat);

  switch (op) {
    case binary_operator::EQUAL: and_merge(lhs_flat, rhs_flat, out_view, op, stream); break;
    case binary_operator::NOT_EQUAL: and_merge(lhs_flat, rhs_flat, out_view, op, stream); break;
    case binary_operator::LESS: lt_row<true>(lhs_flat, rhs_flat, out_view, stream); break;
    // case binary_operator::GREATER: break;
    // case binary_operator::LESS_EQUAL: break;
    case binary_operator::GREATER_EQUAL: lt_row<true>(rhs_flat, lhs_flat, out_view, stream); break;
    // case binary_operator::NULL_EQUALS: break;
    default: CUDF_FAIL("Unsupported operator for these types");
  }
  return out;
}
}  // namespace detail

}  // namespace groupby
}  // namespace cudf
