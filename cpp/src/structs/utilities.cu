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

#include <cudf/binaryop.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <structs/utilities.hpp>

namespace cudf {
namespace structs {
namespace detail {
namespace {
rmm::device_uvector<order> get_orders(binary_operator op,
                                      uint32_t const num_columns,
                                      rmm::cuda_stream_view stream)
{
  std::vector<order> op_modifier(
    num_columns,
    (op == binary_operator::LESS || op == binary_operator::GREATER_EQUAL) ? order::ASCENDING
                                                                          : order::DESCENDING);
  return cudf::detail::make_device_uvector_async(op_modifier, stream);
}

template <typename comparator>
void struct_compare_tabulation(mutable_column_view& out,
                               comparator compare,
                               binary_operator op,
                               rmm::cuda_stream_view stream)
{
  auto d_out = mutable_column_device_view::create(out, stream);

  (op == binary_operator::EQUAL || op == binary_operator::LESS || op == binary_operator::GREATER)
    ? thrust::tabulate(rmm::exec_policy(stream),
                       d_out->begin<bool>(),
                       d_out->end<bool>(),
                       [compare] __device__(auto i) { return compare(i, i); })
    : thrust::tabulate(rmm::exec_policy(stream),
                       d_out->begin<bool>(),
                       d_out->end<bool>(),
                       [compare] __device__(auto i) { return not compare(i, i); });
}

}  // namespace

std::unique_ptr<column> struct_binary_op(column_view const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  auto out      = make_fixed_width_column_for_output(lhs, rhs, op, output_type, stream, mr);
  auto out_view = out->mutable_view();
  struct_binary_operation(out_view, lhs, rhs, op, stream);

  return out;
}

void struct_binary_operation(mutable_column_view& out,
                             column_view const& lhs,
                             column_view const& rhs,
                             binary_operator op,
                             rmm::cuda_stream_view stream)
{
  bool const has_struct_nulls = contains_struct_nulls(lhs) || contains_struct_nulls(rhs);
  auto const lhs_superimposed = superimpose_parent_nulls(lhs);
  auto const rhs_superimposed = superimpose_parent_nulls(rhs);
  auto const lhs_flattener    = flatten_nested_columns(
    table_view{{std::get<0>(lhs_superimposed)}},
    {},
    {},
    has_struct_nulls ? column_nullability::FORCE : column_nullability::MATCH_INCOMING);
  auto const rhs_flattener = flatten_nested_columns(
    table_view{{std::get<0>(rhs_superimposed)}},
    {},
    {},
    has_struct_nulls ? column_nullability::FORCE : column_nullability::MATCH_INCOMING);

  table_view lhs_flat = std::get<0>(lhs_flattener);
  table_view rhs_flat = std::get<0>(rhs_flattener);

  auto d_lhs     = table_device_view::create(lhs_flat);
  auto d_rhs     = table_device_view::create(rhs_flat);
  bool has_nulls = has_nested_nulls(lhs_flat) || has_nested_nulls(rhs_flat);

  if (op == binary_operator::EQUAL || op == binary_operator::NOT_EQUAL) {
    if (has_nulls) {
      auto equal = row_equality_comparator<true>{*d_lhs, *d_rhs, true};
      struct_compare_tabulation(out, equal, op, stream);
    } else {
      auto equal = row_equality_comparator<false>{*d_lhs, *d_rhs, true};
      struct_compare_tabulation(out, equal, op, stream);
    }
  } else if (op == binary_operator::LESS || op == binary_operator::LESS_EQUAL ||
             op == binary_operator::GREATER || op == binary_operator::GREATER_EQUAL) {
    if (has_nulls) {
      auto compare = row_lexicographic_comparator<true>{
        *d_lhs, *d_rhs, get_orders(op, lhs_flat.num_columns(), stream).data()};
      struct_compare_tabulation(out, compare, op, stream);
    } else {
      auto compare = row_lexicographic_comparator<false>{
        *d_lhs, *d_rhs, get_orders(op, lhs_flat.num_columns(), stream).data()};
      struct_compare_tabulation(out, compare, op, stream);
    }
    //  } else if (op == binary_operator::NULL_EQUALS) {
  } else {
    CUDF_FAIL("Unsupported operator for these types");
  }
}

}  // namespace detail
}  // namespace structs
}  // namespace cudf
