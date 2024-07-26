/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/search.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

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
                                       rmm::device_async_resource_ref mr)
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
  auto const matched = dictionary::detail::match_dictionaries(
    {haystack, needles}, stream, rmm::mr::get_current_device_resource());
  auto const& matched_haystack = matched.second.front();
  auto const& matched_needles  = matched.second.back();

  auto const comparator = cudf::experimental::row::lexicographic::two_table_comparator(
    matched_haystack, matched_needles, column_order, null_precedence, stream);
  auto const has_nulls = has_nested_nulls(matched_haystack) or has_nested_nulls(matched_needles);

  auto const haystack_it = cudf::experimental::row::lhs_iterator(0);
  auto const needles_it  = cudf::experimental::row::rhs_iterator(0);

  if (cudf::detail::has_nested_columns(haystack) || cudf::detail::has_nested_columns(needles)) {
    auto const d_comparator = comparator.less<true>(nullate::DYNAMIC{has_nulls});
    if (find_first) {
      thrust::lower_bound(rmm::exec_policy(stream),
                          haystack_it,
                          haystack_it + haystack.num_rows(),
                          needles_it,
                          needles_it + needles.num_rows(),
                          out_it,
                          d_comparator);
    } else {
      thrust::upper_bound(rmm::exec_policy(stream),
                          haystack_it,
                          haystack_it + haystack.num_rows(),
                          needles_it,
                          needles_it + needles.num_rows(),
                          out_it,
                          d_comparator);
    }
  } else {
    auto const d_comparator = comparator.less<false>(nullate::DYNAMIC{has_nulls});
    if (find_first) {
      thrust::lower_bound(rmm::exec_policy(stream),
                          haystack_it,
                          haystack_it + haystack.num_rows(),
                          needles_it,
                          needles_it + needles.num_rows(),
                          out_it,
                          d_comparator);
    } else {
      thrust::upper_bound(rmm::exec_policy(stream),
                          haystack_it,
                          haystack_it + haystack.num_rows(),
                          needles_it,
                          needles_it + needles.num_rows(),
                          out_it,
                          d_comparator);
    }
  }
  return result;
}
}  // namespace

std::unique_ptr<column> lower_bound(table_view const& haystack,
                                    table_view const& needles,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  return search_ordered(haystack, needles, true, column_order, null_precedence, stream, mr);
}

std::unique_ptr<column> upper_bound(table_view const& haystack,
                                    table_view const& needles,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  return search_ordered(haystack, needles, false, column_order, null_precedence, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> lower_bound(table_view const& haystack,
                                    table_view const& needles,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::lower_bound(haystack, needles, column_order, null_precedence, stream, mr);
}

std::unique_ptr<column> upper_bound(table_view const& haystack,
                                    table_view const& needles,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::upper_bound(haystack, needles, column_order, null_precedence, stream, mr);
}

}  // namespace cudf
