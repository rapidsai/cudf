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
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>

#include <sort/sort_impl.cuh>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

/**
 * @brief Flattens struct columns to constituent non-struct columns in the input table.
 *
 */
struct flatten_table {
  // reference variables
  table_view const& input;
  std::vector<order> const& column_order;
  std::vector<null_order> const& null_precedence;
  // output
  std::vector<std::unique_ptr<column>> validity_as_column;
  std::vector<column_view> flat_columns;
  std::vector<order> flat_column_order;
  std::vector<null_order> flat_null_precedence;

  flatten_table(table_view const& input,
                std::vector<order> const& column_order,
                std::vector<null_order> const& null_precedence)
    : input(input), column_order(column_order), null_precedence(null_precedence)
  {
  }

  // Convert null_mask to BOOL8 columns and flatten the struct children in order.
  void flatten_struct_column(column_view const& col, order col_order, null_order col_null_order)
  {
    if (col.nullable()) {
      validity_as_column.push_back(cudf::is_valid(col));
      validity_as_column.back()->set_null_mask(copy_bitmask(col));
      flat_columns.push_back(validity_as_column.back()->view());
      if (not column_order.empty()) flat_column_order.push_back(col_order);  // doesn't matter.
      if (not null_precedence.empty()) flat_null_precedence.push_back(col_null_order);
    }
    for (auto it = col.child_begin(); it != col.child_end(); it++) {
      auto const& child = *it;
      if (child.type().id() == type_id::STRUCT) {
        flatten_struct_column(child, col_order, null_order::BEFORE);
        // default spark behaviour is null_order::BEFORE
      } else {
        flat_columns.push_back(child);
        if (not column_order.empty()) flat_column_order.push_back(col_order);
        if (not null_precedence.empty()) flat_null_precedence.push_back(null_order::BEFORE);
        // default spark behaviour is null_order::BEFORE
      }
    }
  }
  // Note: possibly expand for flattening list columns too.

  /**
   * @copydoc flatten_table
   *
   * @return tuple with flattened table, vector of boolean columns (struct validity),
   * flattened column order, flattened null precedence.
   */
  auto operator()()
  {
    for (decltype(input.num_columns()) i = 0; i < input.num_columns(); ++i) {
      auto const& col = input.column(i);
      if (col.type().id() == type_id::STRUCT) {
        flatten_struct_column(col,
                              (column_order.empty() ? order() : column_order[i]),
                              (null_precedence.empty() ? null_order() : null_precedence[i]));
      } else {
        flat_columns.push_back(col);
        if (not column_order.empty()) flat_column_order.push_back(column_order[i]);
        if (not null_precedence.empty()) flat_null_precedence.push_back(null_precedence[i]);
      }
    }

    return std::make_tuple(table_view{flat_columns},
                           std::move(validity_as_column),
                           std::move(flat_column_order),
                           std::move(flat_null_precedence));
  }
};

/**
 * @copydoc cudf::detail::flatten_nested_columns
 */
std::tuple<table_view,
           std::vector<std::unique_ptr<column>>,
           std::vector<order>,
           std::vector<null_order>>
flatten_nested_columns(table_view const& input,
                       std::vector<order> const& column_order,
                       std::vector<null_order> const& null_precedence)
{
  std::vector<std::unique_ptr<column>> validity_as_column;
  auto const has_struct = std::any_of(
    input.begin(), input.end(), [](auto const& col) { return col.type().id() == type_id::STRUCT; });
  if (not has_struct)
    return std::make_tuple(input, std::move(validity_as_column), column_order, null_precedence);

  return flatten_table{input, column_order, null_precedence}();
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
