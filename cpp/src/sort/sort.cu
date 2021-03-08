/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/unary.hpp>

#include <sort/sort_impl.cuh>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

// Convert null_mask to BOOL8 columns and flatten the struct children in order.
void flatten_struct_column(column_view const& col,
                           std::vector<column_view>& all_columns,
                           std::vector<std::unique_ptr<column>>& validity_as_column)
{
  for (auto it = col.child_begin(); it != col.child_end(); it++) {
    auto const& child = *it;
    if (child.type().id() == type_id::STRUCT) {
      if (child.nullable()) {
        validity_as_column.push_back(cudf::is_valid(child));
        all_columns.push_back(validity_as_column.back()->view());
      }
      flatten_struct_column(child, all_columns, validity_as_column);
    } else {
      all_columns.push_back(child);
    }
  }
}

// Flatten the nested structs for comparator due to recursion limitation in device code.
std::pair<table_view, std::vector<std::unique_ptr<column>>> flatten_nested_columns(
  table_view const& input)
{
  std::vector<std::unique_ptr<column>> nested_struct_validity;
  auto const has_nested_struct = std::any_of(input.begin(), input.end(), [](auto const& col) {
    return col.type().id() == type_id::STRUCT and
           std::any_of(col.child_begin(), col.child_end(), [](auto const& child) {
             return child.type().id() == type_id::STRUCT;
           });
  });
  if (not has_nested_struct) return std::make_pair(input, std::move(nested_struct_validity));

  std::vector<column_view> all_columns;
  for (auto const& col : input) {
    auto is_nested_struct = (col.type().id() == type_id::STRUCT) and
                            std::any_of(col.child_begin(), col.child_end(), [](auto const& col) {
                              return col.type().id() == type_id::STRUCT;
                            });
    if (is_nested_struct) {
      std::vector<column_view> flattened;
      flatten_struct_column(col, flattened, nested_struct_validity);
      auto flat_struct = column_view(col.type(),
                                     col.size(),
                                     col.head(),
                                     col.null_mask(),
                                     col.null_count(),
                                     col.offset(),
                                     std::move(flattened));
      all_columns.push_back(flat_struct);
    } else {
      all_columns.push_back(col);
    }
  }
  return std::make_pair(table_view{all_columns}, std::move(nested_struct_validity));
}

std::unique_ptr<column> sorted_order(table_view input,
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
  template <typename T, typename std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>
  void operator()(mutable_column_view& col, bool ascending, rmm::cuda_stream_view stream) const
  {
    CUDF_EXPECTS(!col.has_nulls(), "Nulls not supported for in-place sort");
    if (ascending) {
      thrust::sort(rmm::exec_policy(stream), col.begin<T>(), col.end<T>(), thrust::less<T>());
    } else {
      thrust::sort(rmm::exec_policy(stream), col.begin<T>(), col.end<T>(), thrust::greater<T>());
    }
  }

  template <typename T, typename std::enable_if_t<!cudf::is_fixed_width<T>()>* = nullptr>
  void operator()(mutable_column_view&, bool, rmm::cuda_stream_view) const
  {
    CUDF_FAIL("Column type must be relationally comparable and fixed-width");
  }
};

std::unique_ptr<table> sort(table_view input,
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
    input, input, column_order, null_precedence, rmm::cuda_stream_default, mr);
}

}  // namespace detail

std::unique_ptr<column> sorted_order(table_view input,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::sorted_order(input, column_order, null_precedence, rmm::cuda_stream_default, mr);
}

std::unique_ptr<table> sort(table_view input,
                            std::vector<order> const& column_order,
                            std::vector<null_order> const& null_precedence,
                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::sort(input, column_order, null_precedence, rmm::cuda_stream_default, mr);
}

std::unique_ptr<table> sort_by_key(table_view const& values,
                                   table_view const& keys,
                                   std::vector<order> const& column_order,
                                   std::vector<null_order> const& null_precedence,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::sort_by_key(
    values, keys, column_order, null_precedence, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
