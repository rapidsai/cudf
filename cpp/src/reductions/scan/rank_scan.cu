/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/table/experimental/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief generate row ranks or dense ranks using a row comparison then scan the results
 *
 * @tparam value_resolver flag value resolver with boolean first and row number arguments
 * @tparam scan_operator scan function ran on the flag values
 * @param order_by input column to generate ranks for
 * @param resolver flag value resolver
 * @param scan_op scan operation ran on the flag results
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return std::unique_ptr<column> rank values
 */
template <typename value_resolver, typename scan_operator>
std::unique_ptr<column> rank_generator(column_view const& order_by,
                                       value_resolver resolver,
                                       scan_operator scan_op,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto comp = cudf::experimental::row::equality::self_comparator(table_view{{order_by}}, stream);
  auto const device_comparator =
    comp.equal_to(nullate::DYNAMIC{has_nested_nulls(table_view({order_by}))});
  auto ranks = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, order_by.size(), mask_state::UNALLOCATED, stream, mr);
  auto mutable_ranks = ranks->mutable_view();

  thrust::tabulate(rmm::exec_policy(stream),
                   mutable_ranks.begin<size_type>(),
                   mutable_ranks.end<size_type>(),
                   [comparator = device_comparator, resolver] __device__(size_type row_index) {
                     return resolver(row_index == 0 || !comparator(row_index, row_index - 1),
                                     row_index);
                   });

  thrust::inclusive_scan(rmm::exec_policy(stream),
                         mutable_ranks.begin<size_type>(),
                         mutable_ranks.end<size_type>(),
                         mutable_ranks.begin<size_type>(),
                         scan_op);
  return ranks;
}

}  // namespace

std::unique_ptr<column> inclusive_dense_rank_scan(column_view const& order_by,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  return rank_generator(
    order_by,
    [] __device__(bool const unequal, size_type const) { return unequal ? 1 : 0; },
    DeviceSum{},
    stream,
    mr);
}

std::unique_ptr<column> inclusive_rank_scan(column_view const& order_by,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(!cudf::structs::detail::is_or_has_nested_lists(order_by),
               "Unsupported list type in rank scan.");
  return rank_generator(
    order_by,
    [] __device__(bool unequal, auto row_index) { return unequal ? row_index + 1 : 0; },
    DeviceMax{},
    stream,
    mr);
}

std::unique_ptr<column> inclusive_one_normalized_percent_rank_scan(
  column_view const& order_by, rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr)
{
  auto const rank_column =
    inclusive_rank_scan(order_by, stream, rmm::mr::get_current_device_resource());
  auto const rank_view = rank_column->view();

  // Result type for min 0-index percent rank is independent of input type.
  using result_type        = double;
  auto percent_rank_result = cudf::make_fixed_width_column(
    data_type{type_to_id<result_type>()}, rank_view.size(), mask_state::UNALLOCATED, stream, mr);

  thrust::transform(rmm::exec_policy(stream),
                    rank_view.begin<size_type>(),
                    rank_view.end<size_type>(),
                    percent_rank_result->mutable_view().begin<result_type>(),
                    [n_rows = rank_view.size()] __device__(auto const rank) {
                      return n_rows == 1 ? 0.0 : ((rank - 1.0) / (n_rows - 1));
                    });
  return percent_rank_result;
}

}  // namespace detail
}  // namespace cudf
