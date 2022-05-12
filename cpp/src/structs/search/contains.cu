/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/detail/structs/utilities.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/structs/detail/contains.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace structs {
namespace detail {

bool contains(structs_column_view const& haystack,
              scalar const& needle,
              rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(haystack.type() == needle.type(), "scalar and column types must match");

  auto const scalar_table = static_cast<struct_scalar const*>(&needle)->view();
  CUDF_EXPECTS(haystack.num_children() == scalar_table.num_columns(),
               "struct scalar and structs column must have the same number of children");
  for (size_type i = 0; i < haystack.num_children(); ++i) {
    CUDF_EXPECTS(haystack.child(i).type() == scalar_table.column(i).type(),
                 "scalar and column children types must match");
  }

  // Prepare to flatten the structs column and scalar.
  auto const has_null_elements = has_nested_nulls(table_view{std::vector<column_view>{
                                   haystack.child_begin(), haystack.child_end()}}) ||
                                 has_nested_nulls(scalar_table);
  auto const flatten_nullability = has_null_elements
                                     ? structs::detail::column_nullability::FORCE
                                     : structs::detail::column_nullability::MATCH_INCOMING;

  // Flatten the input structs column, only materialize the bitmask if there is null in the input.
  auto const haystack_flattened =
    structs::detail::flatten_nested_columns(table_view{{haystack}}, {}, {}, flatten_nullability);
  auto const needle_flattened =
    structs::detail::flatten_nested_columns(scalar_table, {}, {}, flatten_nullability);

  // The struct scalar only contains the struct member columns.
  // Thus, if there is any null in the input, we must exclude the first column in the flattened
  // table of the input column from searching because that column is the materialized bitmask of
  // the input structs column.
  auto const haystack_flattened_content  = haystack_flattened.flattened_columns();
  auto const haystack_flattened_children = table_view{std::vector<column_view>{
    haystack_flattened_content.begin() + static_cast<size_type>(has_null_elements),
    haystack_flattened_content.end()}};

  auto const d_haystack_children_ptr =
    table_device_view::create(haystack_flattened_children, stream);
  auto const d_needle_ptr = table_device_view::create(needle_flattened, stream);

  auto const start_iter = thrust::make_counting_iterator<size_type>(0);
  auto const end_iter   = start_iter + haystack.size();
  auto const comp       = row_equality_comparator(nullate::DYNAMIC{has_null_elements},
                                            *d_haystack_children_ptr,
                                            *d_needle_ptr,
                                            null_equality::EQUAL);
  auto const found_iter = thrust::find_if(
    rmm::exec_policy(stream), start_iter, end_iter, [comp] __device__(auto const idx) {
      return comp(idx, 0);  // compare haystack[idx] == val[0].
    });

  return found_iter != end_iter;
}

}  // namespace detail
}  // namespace structs
}  // namespace cudf
