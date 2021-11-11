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

#include <binaryop/compiled/binary_ops.cuh>
#include <binaryop/compiled/struct_binary_ops.cuh>

#include <cudf/detail/structs/utilities.hpp>
#include <cudf/table/row_operators.cuh>

namespace cudf::binops::compiled {
void dispatch_equality_op(mutable_column_view& out,
                          column_view const& lhs,
                          column_view const& rhs,
                          bool is_lhs_scalar,
                          bool is_rhs_scalar,
                          binary_operator op,
                          rmm::cuda_stream_view stream)
{
  if (is_struct(lhs.type()) && is_struct(rhs.type())) {
    auto const nullability =
      structs::detail::contains_null_structs(lhs) || structs::detail::contains_null_structs(rhs)
        ? structs::detail::column_nullability::FORCE
        : structs::detail::column_nullability::MATCH_INCOMING;
    auto const lhs_flattened =
      structs::detail::flatten_nested_columns(table_view{{lhs}}, {}, {}, nullability);
    auto const rhs_flattened =
      structs::detail::flatten_nested_columns(table_view{{rhs}}, {}, {}, nullability);
    auto d_lhs = table_device_view::create(lhs_flattened);
    auto d_rhs = table_device_view::create(rhs_flattened);

    auto const do_compare = [&](auto const& comp) {
      detail::struct_compare(
        out, comp, is_lhs_scalar, is_rhs_scalar, op == binary_operator::NOT_EQUAL, stream);
    };
    switch (op) {
      case binary_operator::EQUAL:
      case binary_operator::NOT_EQUAL:
        has_nested_nulls(lhs_flattened) || has_nested_nulls(rhs_flattened)
          ? do_compare(row_equality_comparator<true>{*d_lhs, *d_rhs})
          : do_compare(row_equality_comparator<false>{*d_lhs, *d_rhs});
        break;
      default: CUDF_FAIL("Unsupported operator for these types");
    }
  } else {
    auto common_dtype = get_common_type(out.type(), lhs.type(), rhs.type());
    auto lhsd         = column_device_view::create(lhs, stream);
    auto rhsd         = column_device_view::create(rhs, stream);
    auto outd         = mutable_column_device_view::create(out, stream);
    // Execute it on every element
    for_each(stream,
             out.size(),
             [op,
              outd = *outd,
              lhsd = *lhsd,
              rhsd = *rhsd,
              is_lhs_scalar,
              is_rhs_scalar,
              common_dtype] __device__(size_type i) {
               // clang-format off
      // Similar enabled template types should go together (better performance)
      switch (op) {
      case binary_operator::EQUAL:         device_type_dispatcher<ops::Equal>{outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, common_dtype}(i); break;
      case binary_operator::NOT_EQUAL:     device_type_dispatcher<ops::NotEqual>{outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, common_dtype}(i); break;
      case binary_operator::NULL_EQUALS:   device_type_dispatcher<ops::NullEquals>{outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, common_dtype}(i); break;
      default:;
      }
               // clang-format on
             });
  }
}
}  // namespace cudf::binops::compiled
