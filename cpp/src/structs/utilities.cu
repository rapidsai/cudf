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

template <bool has_nulls>
struct StructComparatorFunctor {
  template <typename output_type, typename compare_op>
  void negate_result(mutable_column_view out, compare_op comparator, rmm::cuda_stream_view stream)
  {
    if (out.has_nulls()) {
      auto d_out = mutable_column_device_view::create(out);
      thrust::tabulate(rmm::exec_policy(stream),
                       out.begin<output_type>(),
                       out.end<output_type>(),
                       [comparator, d_out = *d_out] __device__(size_type row_index) {
                         return d_out.is_valid_nocheck(row_index) &&
                                !comparator(row_index, row_index);
                       });
    } else {
      thrust::tabulate(
        rmm::exec_policy(stream),
        out.begin<output_type>(),
        out.end<output_type>(),
        [comparator] __device__(size_type row_index) { return !comparator(row_index, row_index); });
    }
  }

  template <typename output_type, typename compare_op>
  void direct_result(mutable_column_view out, compare_op comparator, rmm::cuda_stream_view stream)
  {
    if (out.has_nulls()) {
      auto d_out = mutable_column_device_view::create(out);
      thrust::tabulate(rmm::exec_policy(stream),
                       out.begin<output_type>(),
                       out.end<output_type>(),
                       [comparator, d_out = *d_out] __device__(size_type row_index) {
                         return d_out.is_valid_nocheck(row_index) &&
                                comparator(row_index, row_index);
                       });
    } else {
      thrust::tabulate(
        rmm::exec_policy(stream),
        out.begin<output_type>(),
        out.end<output_type>(),
        [comparator] __device__(size_type row_index) { return comparator(row_index, row_index); });
    }
  }

  rmm::device_uvector<order> get_descending_orders(uint32_t const num_columns,
                                                   rmm::cuda_stream_view stream)
  {
    std::vector<order> op_modifier(num_columns, order::DESCENDING);
    return cudf::detail::make_device_uvector_async(op_modifier, stream);
  }

  template <typename T, std::enable_if_t<is_numeric<T>()>* = nullptr>
  void __host__ operator()(table_view const& lhs,
                           table_view const& rhs,
                           mutable_column_view& out,
                           binary_operator op,
                           rmm::cuda_stream_view stream)
  {
    auto d_lhs = table_device_view::create(lhs);
    auto d_rhs = table_device_view::create(rhs);

    switch (op) {
      case binary_operator::EQUAL:
        direct_result<T, row_equality_comparator<has_nulls>>(
          out, row_equality_comparator<has_nulls>{*d_lhs, *d_rhs, true}, stream);
        break;
      case binary_operator::NOT_EQUAL:
        negate_result<T, row_equality_comparator<has_nulls>>(
          out, row_equality_comparator<has_nulls>{*d_lhs, *d_rhs, true}, stream);
        break;
      case binary_operator::LESS:
        direct_result<T, row_lexicographic_comparator<has_nulls>>(
          out, row_lexicographic_comparator<has_nulls>{*d_lhs, *d_rhs}, stream);
        break;
      case binary_operator::GREATER:
        direct_result<T, row_lexicographic_comparator<has_nulls>>(
          out,
          row_lexicographic_comparator<has_nulls>{
            *d_lhs, *d_rhs, get_descending_orders(lhs.num_columns(), stream).data()},
          stream);
        break;
      case binary_operator::LESS_EQUAL:
        negate_result<T, row_lexicographic_comparator<has_nulls>>(
          out,
          row_lexicographic_comparator<has_nulls>{
            *d_lhs, *d_rhs, get_descending_orders(lhs.num_columns(), stream).data()},
          stream);
        break;
      case binary_operator::GREATER_EQUAL:
        negate_result<T, row_lexicographic_comparator<has_nulls>>(
          out, row_lexicographic_comparator<has_nulls>{*d_lhs, *d_rhs}, stream);
        break;
      // case binary_operator::NULL_EQUALS: break;
      default: CUDF_FAIL("Unsupported operator for these types");
    }
  }

  template <typename T, std::enable_if_t<!is_numeric<T>()>* = nullptr>
  void __host__ operator()(table_view const& lhs,
                           table_view const& rhs,
                           mutable_column_view& out,
                           binary_operator op,
                           rmm::cuda_stream_view stream)
  {
    CUDF_FAIL("unsupported output type");
  }
};

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

  if (has_nested_nulls(lhs_flat) || has_nested_nulls(rhs_flat)) {
    type_dispatcher(
      out.type(), StructComparatorFunctor<true>{}, lhs_flat, rhs_flat, out, op, stream);
  } else {
    type_dispatcher(
      out.type(), StructComparatorFunctor<false>{}, lhs_flat, rhs_flat, out, op, stream);
  }
}

}  // namespace detail
}  // namespace structs
}  // namespace cudf
