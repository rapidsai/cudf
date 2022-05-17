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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>

namespace cudf {
namespace detail {
namespace {

std::unique_ptr<column> search_ordered(table_view const& haystack,
                                       table_view const& needles,
                                       bool find_first,
                                       std::vector<order> const& column_order,
                                       std::vector<null_order> const& null_precedence,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(
    column_order.empty() or static_cast<std::size_t>(haystack.num_columns()) == column_order.size(),
    "Mismatch between number of columns and column order.");
  CUDF_EXPECTS(null_precedence.empty() or
                 static_cast<std::size_t>(haystack.num_columns()) == null_precedence.size(),
               "Mismatch between number of columns and null precedence.");

  // Allocate result column
  auto result = make_numeric_column(
    data_type{type_to_id<size_type>()}, needles.num_rows(), mask_state::UNALLOCATED, stream, mr);
  auto const out_it = result->mutable_view().data<size_type>();

  // Handle empty inputs
  if (haystack.num_rows() == 0) {
    CUDF_CUDA_TRY(
      cudaMemsetAsync(out_it, 0, needles.num_rows() * sizeof(size_type), stream.value()));
    return result;
  }

  // This utility will ensure all corresponding dictionary columns have matching keys.
  // It will return any new dictionary columns created as well as updated table_views.
  auto const matched = dictionary::detail::match_dictionaries({haystack, needles}, stream);

  // Prepare to flatten the structs column
  auto const has_null_elements   = has_nested_nulls(haystack) or has_nested_nulls(needles);
  auto const flatten_nullability = has_null_elements
                                     ? structs::detail::column_nullability::FORCE
                                     : structs::detail::column_nullability::MATCH_INCOMING;

  // 0-table_view, 1-column_order, 2-null_precedence, 3-validity_columns
  auto const t_flattened = structs::detail::flatten_nested_columns(
    matched.second.front(), column_order, null_precedence, flatten_nullability);
  auto const values_flattened =
    structs::detail::flatten_nested_columns(matched.second.back(), {}, {}, flatten_nullability);

  auto const t_d      = table_device_view::create(t_flattened, stream);
  auto const values_d = table_device_view::create(values_flattened, stream);
  auto const& lhs     = find_first ? *t_d : *values_d;
  auto const& rhs     = find_first ? *values_d : *t_d;

  auto const& column_order_flattened    = t_flattened.orders();
  auto const& null_precedence_flattened = t_flattened.null_orders();
  auto const column_order_dv = detail::make_device_uvector_async(column_order_flattened, stream);
  auto const null_precedence_dv =
    detail::make_device_uvector_async(null_precedence_flattened, stream);

  auto const count_it = thrust::make_counting_iterator<size_type>(0);
  auto const comp     = row_lexicographic_comparator(nullate::DYNAMIC{has_null_elements},
                                                 lhs,
                                                 rhs,
                                                 column_order_dv.data(),
                                                 null_precedence_dv.data());

  if (find_first) {
    thrust::lower_bound(rmm::exec_policy(stream),
                        count_it,
                        count_it + haystack.num_rows(),
                        count_it,
                        count_it + needles.num_rows(),
                        out_it,
                        comp);
  } else {
    thrust::upper_bound(rmm::exec_policy(stream),
                        count_it,
                        count_it + haystack.num_rows(),
                        count_it,
                        count_it + needles.num_rows(),
                        out_it,
                        comp);
  }
  return result;
}
}  // namespace

std::unique_ptr<column> lower_bound(table_view const& haystack,
                                    table_view const& needles,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  return search_ordered(haystack, needles, true, column_order, null_precedence, stream, mr);
}

std::unique_ptr<column> upper_bound(table_view const& haystack,
                                    table_view const& needles,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  return search_ordered(haystack, needles, false, column_order, null_precedence, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> lower_bound(table_view const& haystack,
                                    table_view const& needles,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::lower_bound(
    haystack, needles, column_order, null_precedence, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> upper_bound(table_view const& haystack,
                                    table_view const& needles,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::upper_bound(
    haystack, needles, column_order, null_precedence, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
