/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/table/row_operators.cuh>

#include <rmm/exec_policy.hpp>
#include <structs/utilities.hpp>
#include "cudf/binaryop.hpp"
#include "cudf/column/column_device_view.cuh"
#include "cudf/table/table_device_view.cuh"
#include "cudf/table/table_view.hpp"
#include "thrust/logical.h"

namespace cudf {
namespace structs {
namespace detail {

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

std::unique_ptr<column> struct_binary_operation(column_view const& lhs,
                                                column_view const& rhs,
                                                binary_operator op,
                                                data_type output_type,
                                                rmm::cuda_stream_view stream,
                                                rmm::mr::device_memory_resource* mr)
{
  auto const lhs_superimposed = superimpose_parent_nulls(lhs);
  auto const lhs_flattener    = flatten_nested_columns(
    table_view{{std::get<0>(lhs_superimposed)}}, {}, {}, column_nullability::MATCH_INCOMING);
  table_view lhs_flat = std::get<0>(lhs_flattener);
  // auto const d_lhs_flat = table_device_view::create(lhs_flat, stream);

  auto const rhs_superimposed = superimpose_parent_nulls(rhs);
  auto const rhs_flattener    = flatten_nested_columns(
    table_view{{std::get<0>(rhs_superimposed)}}, {}, {}, column_nullability::MATCH_INCOMING);
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
}  // namespace structs
}  // namespace cudf

// binary_op_compare() {
//     type_dispatcher()
// }
