/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <sort/sort_impl.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/functional.h>

namespace cudf {
namespace detail {
std::unique_ptr<column> sorted_order(table_view const& input,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  return sorted_order<false>(input, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> sort_by_key(table_view const& values,
                                   table_view const& keys,
                                   std::vector<order> const& column_order,
                                   std::vector<null_order> const& null_precedence,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(values.num_rows() == keys.num_rows(),
               "Mismatch in number of rows for values and keys");

  auto sorted_order = detail::sorted_order(
    keys, column_order, null_precedence, stream, rmm::mr::get_current_device_resource());

  return detail::gather(values,
                        sorted_order->view(),
                        out_of_bounds_policy::DONT_CHECK,
                        detail::negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}

struct inplace_column_sort_fn {
  template <typename T, std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>
  void operator()(mutable_column_view& col, bool ascending, rmm::cuda_stream_view stream) const
  {
    CUDF_EXPECTS(!col.has_nulls(), "Nulls not supported for in-place sort");
    if (ascending) {
      thrust::sort(rmm::exec_policy(stream), col.begin<T>(), col.end<T>(), thrust::less<T>());
    } else {
      thrust::sort(rmm::exec_policy(stream), col.begin<T>(), col.end<T>(), thrust::greater<T>());
    }
  }

  template <typename T, std::enable_if_t<!cudf::is_fixed_width<T>()>* = nullptr>
  void operator()(mutable_column_view&, bool, rmm::cuda_stream_view) const
  {
    CUDF_FAIL("Column type must be relationally comparable and fixed-width");
  }
};

std::unique_ptr<table> sort(table_view const& input,
                            std::vector<order> const& column_order,
                            std::vector<null_order> const& null_precedence,
                            rmm::cuda_stream_view stream,
                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  // fast-path sort conditions: single, non-floating-point, fixed-width column with no nulls
  if (input.num_columns() == 1 && !input.column(0).has_nulls() &&
      cudf::is_fixed_width(input.column(0).type()) &&
      !cudf::is_floating_point(input.column(0).type())) {
    auto output    = std::make_unique<column>(input.column(0), stream, mr);
    auto view      = output->mutable_view();
    bool ascending = (column_order.empty() ? true : column_order.front() == order::ASCENDING);
    cudf::type_dispatcher<dispatch_storage_type>(
      output->type(), inplace_column_sort_fn{}, view, ascending, stream);
    std::vector<std::unique_ptr<column>> columns;
    columns.emplace_back(std::move(output));
    return std::make_unique<table>(std::move(columns));
  }
  return detail::sort_by_key(
    input, input, column_order, null_precedence, cudf::default_stream_value, mr);
}

}  // namespace detail

std::unique_ptr<column> sorted_order(table_view const& input,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::sorted_order(input, column_order, null_precedence, cudf::default_stream_value, mr);
}

std::unique_ptr<table> sort(table_view const& input,
                            std::vector<order> const& column_order,
                            std::vector<null_order> const& null_precedence,
                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::sort(input, column_order, null_precedence, cudf::default_stream_value, mr);
}

std::unique_ptr<table> sort_by_key(table_view const& values,
                                   table_view const& keys,
                                   std::vector<order> const& column_order,
                                   std::vector<null_order> const& null_precedence,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::sort_by_key(
    values, keys, column_order, null_precedence, cudf::default_stream_value, mr);
}

}  // namespace cudf
