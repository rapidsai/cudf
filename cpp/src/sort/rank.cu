/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/column/column_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/detail/gather.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/sequence.h>

namespace cudf {
namespace experimental {

namespace detail {
  /*
// Create permuted row indices that would materialize sorted order
rmm::device_vector<size_type> sorted_order1(table_view input,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     bool stable,
                                     cudaStream_t stream) {
  if (input.num_rows() == 0 or input.num_columns() == 0) {
    return rmm::device_vector<size_type>(0);
  }

  if (not column_order.empty()) {
    CUDF_EXPECTS(
        static_cast<std::size_t>(input.num_columns()) == column_order.size(),
        "Mismatch between number of columns and column order.");
  }

  if (not null_precedence.empty()) {
    CUDF_EXPECTS(
        static_cast<std::size_t>(input.num_columns()) == null_precedence.size(),
        "Mismatch between number of columns and null_precedence size.");
  }

  rmm::device_vector<size_type> sorted_indices(input.num_rows());
  thrust::sequence(rmm::exec_policy(stream)->on(stream),
                   sorted_indices.begin(),
                   sorted_indices.end(), 0);

  auto device_table = table_device_view::create(input, stream);
  rmm::device_vector<order> d_column_order(column_order);

  if (has_nulls(input)) {
    rmm::device_vector<null_order> d_null_precedence(null_precedence);
    auto comparator = row_lexicographic_comparator<true>(
        *device_table, *device_table, d_column_order.data().get(),
        d_null_precedence.data().get());
    if (stable)
      thrust::stable_sort(rmm::exec_policy(stream)->on(stream),
                          sorted_indices.begin(), sorted_indices.end(),
                          comparator);
    else
      thrust::sort(rmm::exec_policy(stream)->on(stream), 
                   sorted_indices.begin(), sorted_indices.end(), comparator);

  } else {
    auto comparator = row_lexicographic_comparator<false>(
        *device_table, *device_table, d_column_order.data().get());
    if (stable)
      thrust::stable_sort(rmm::exec_policy(stream)->on(stream),
                          sorted_indices.begin(), sorted_indices.end(),
                          comparator);
    else
      thrust::sort(rmm::exec_policy(stream)->on(stream),
                   sorted_indices.begin(), sorted_indices.end(), comparator);
  }

  return sorted_indices;
}
*/

std::unique_ptr<table> rank(
    table_view const& input,
    rank_method method,
    order column_order,
    include_nulls _include_nulls,
    null_order null_precedence,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream=0) {
  //na_option=keep assign NA to NA values
  if(_include_nulls == include_nulls::NO)
    null_precedence = null_order::AFTER;
  auto const size = input.num_rows();
  
  std::vector<std::unique_ptr<column>> rank_columns;
  for (auto const& input_col : input) {
    std::unique_ptr<column> sorted_order =
        (method == rank_method::FIRST)
            ? detail::stable_sorted_order(
                table_view{{input_col}}, {column_order}, {null_precedence}, mr, stream)
            : detail::sorted_order(
                table_view{{input_col}}, {column_order}, {null_precedence}, mr, stream);
    column_view sorted_order_view = sorted_order->view();

    if(_include_nulls == include_nulls::NO)
      rank_columns.push_back(
          make_numeric_column(data_type(FLOAT64), size,
                              copy_bitmask(input_col, stream, mr),
                              input_col.null_count(), stream, mr));
    else
      rank_columns.push_back(make_numeric_column(
          data_type(FLOAT64), size, mask_state::UNALLOCATED, stream, mr));

    auto rank_mutable_view = rank_columns.back()->mutable_view();
    auto rank_data = rank_mutable_view.data<double>();

    //FIRST
    thrust::scatter(
        rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<double>(1),
        thrust::make_counting_iterator<double>(input_col.size() + 1),
        sorted_order_view.begin<size_type>(), rank_data);
  }
  return std::make_unique<table>(std::move(rank_columns));
}
}  // namespace detail

std::unique_ptr<table> rank(table_view input,
                             rank_method method,
                             order column_order,
                             include_nulls _include_nulls,
                             null_order null_precedence,
                             rmm::mr::device_memory_resource* mr) {
    return detail::rank(input, method, column_order, _include_nulls, null_precedence, mr);
}
}  // namespace experimental
}  // namespace cudf
