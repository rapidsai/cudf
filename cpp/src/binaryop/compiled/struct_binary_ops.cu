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

// #include <binaryop/compiled/binary_ops.hpp>
#include <binaryop/compiled/struct_binary_ops.cuh>
// #include <binaryop/compiled/binary_ops.cuh>
// #include <cudf/binaryop.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/row_operators.cuh>

namespace cudf::binops::compiled::detail {
// namespace {

// }
void struct_lexicographic_compare(mutable_column_view& out,
                                  column_view const& lhs,
                                  column_view const& rhs,
                                  bool is_lhs_scalar,
                                  bool is_rhs_scalar,
                                  order op_order,
                                  bool flip_output,
                                  rmm::cuda_stream_view stream)
{
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
  auto compare_orders =
    cudf::detail::make_device_uvector_async(std::vector<order>(lhs.size(), op_order), stream);

  auto const do_compare = [&](auto const& comp) {
    struct_compare(out, comp, is_lhs_scalar, is_rhs_scalar, flip_output, stream);
  };
  has_nested_nulls(lhs_flattened) || has_nested_nulls(rhs_flattened)
    ? do_compare(row_lexicographic_comparator<true>{*d_lhs, *d_rhs, compare_orders.data()})
    : do_compare(row_lexicographic_comparator<false>{*d_lhs, *d_rhs, compare_orders.data()});
}
}  // namespace cudf::binops::compiled::detail