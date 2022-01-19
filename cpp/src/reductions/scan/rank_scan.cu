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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/table/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>
#include <thrust/tabulate.h>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief generate row ranks or dense ranks using a row comparison then scan the results
 *
 * @tparam value_resolver flag value resolver with boolean first and row number arguments
 * @tparam scan_operator scan function ran on the flag values
 * @param order_by input column to generate ranks for
 * @param has_nulls if the order_by column has nested nulls
 * @param resolver flag value resolver
 * @param scan_op scan operation ran on the flag results
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return std::unique_ptr<column> rank values
 */
template <typename value_resolver, typename scan_operator>
std::unique_ptr<column> rank_generator(column_view const& order_by,
                                       bool has_nulls,
                                       value_resolver resolver,
                                       scan_operator scan_op,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto const flattened = cudf::structs::detail::flatten_nested_columns(
    table_view{{order_by}}, {}, {}, structs::detail::column_nullability::MATCH_INCOMING);
  auto const d_flat_order = table_device_view::create(flattened, stream);
  row_equality_comparator comparator(
    nullate::DYNAMIC{has_nulls}, *d_flat_order, *d_flat_order, null_equality::EQUAL);
  auto ranks         = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                       flattened.flattened_columns().num_rows(),
                                       mask_state::UNALLOCATED,
                                       stream,
                                       mr);
  auto mutable_ranks = ranks->mutable_view();

  thrust::tabulate(rmm::exec_policy(stream),
                   mutable_ranks.begin<size_type>(),
                   mutable_ranks.end<size_type>(),
                   [comparator, resolver] __device__(size_type row_index) {
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
  CUDF_EXPECTS(!cudf::structs::detail::is_or_has_nested_lists(order_by),
               "Unsupported list type in dense_rank scan.");
  return rank_generator(
    order_by,
    has_nested_nulls(table_view{{order_by}}),
    [] __device__(bool equality, auto row_index) { return equality; },
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
    has_nested_nulls(table_view{{order_by}}),
    [] __device__(bool equality, auto row_index) { return equality ? row_index + 1 : 0; },
    DeviceMax{},
    stream,
    mr);
}

}  // namespace detail
}  // namespace cudf
