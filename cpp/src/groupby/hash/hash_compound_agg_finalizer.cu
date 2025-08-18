/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "hash_compound_agg_finalizer.hpp"
#include "helpers.cuh"
#include "m2_var_functor.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf::groupby::detail::hash {

hash_compound_agg_finalizer::hash_compound_agg_finalizer(column_view col,
                                                         cudf::detail::result_cache* cache,
                                                         size_type const* d_output_index_map,
                                                         bitmask_type const* d_row_bitmask,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
  : col{col},
    cache{cache},
    d_output_index_map{d_output_index_map},
    d_row_bitmask{d_row_bitmask},
    stream{stream},
    mr{mr},
    result_type{cudf::is_dictionary(col.type()) ? cudf::dictionary_column_view(col).keys().type()
                                                : col.type()}
{
}

auto hash_compound_agg_finalizer::gather_argminmax(aggregation const& agg)
{
  auto arg_result = cache->get_result(col, agg);
  // We make a view of ARG(MIN/MAX) result without a null mask and gather
  // using this map. The values in data buffer of ARG(MIN/MAX) result
  // corresponding to null values was initialized to ARG(MIN/MAX)_SENTINEL
  // which is an out of bounds index value (-1) and causes the gathered
  // value to be null.
  column_view null_removed_map(data_type(type_to_id<size_type>()),
                               arg_result.size(),
                               static_cast<void const*>(arg_result.data<size_type>()),
                               nullptr,
                               0);
  auto gather_argminmax =
    cudf::detail::gather(table_view({col}),
                         null_removed_map,
                         arg_result.nullable() ? cudf::out_of_bounds_policy::NULLIFY
                                               : cudf::out_of_bounds_policy::DONT_CHECK,
                         cudf::detail::negative_index_policy::NOT_ALLOWED,
                         stream,
                         mr);
  return std::move(gather_argminmax->release()[0]);
}

void hash_compound_agg_finalizer::visit(cudf::detail::min_aggregation const& agg)
{
  if (result_type.id() == type_id::STRING) {
    if (cache->has_result(col, agg)) { return; }
    auto transformed_agg = make_argmin_aggregation();
    cache->add_result(col, agg, gather_argminmax(*transformed_agg));
  }  // else: no-op, since this is only relevant for compound aggregations
  // TODO: support other nested types.
}

void hash_compound_agg_finalizer::visit(cudf::detail::max_aggregation const& agg)
{
  if (result_type.id() == type_id::STRING) {
    if (cache->has_result(col, agg)) { return; }
    auto transformed_agg = make_argmax_aggregation();
    cache->add_result(col, agg, gather_argminmax(*transformed_agg));
  }  // else: no-op, since this is only relevant for compound aggregations
  // TODO: support other nested types.
}

void hash_compound_agg_finalizer::visit(cudf::detail::mean_aggregation const& agg)
{
  if (cache->has_result(col, agg)) { return; }

  auto const sum_agg      = make_sum_aggregation();
  auto const count_agg    = make_count_aggregation();
  auto const sum_result   = cache->get_result(col, *sum_agg);
  auto const count_result = cache->get_result(col, *count_agg);

  auto result =
    cudf::detail::binary_operation(sum_result,
                                   count_result,
                                   binary_operator::DIV,
                                   cudf::detail::target_type(result_type, aggregation::MEAN),
                                   stream,
                                   mr);
  cache->add_result(col, agg, std::move(result));
}

void hash_compound_agg_finalizer::visit(cudf::detail::m2_aggregation const& agg)
{
  if (cache->has_result(col, agg)) { return; }

  auto const sum_agg      = make_sum_aggregation();
  auto const count_agg    = make_count_aggregation();
  auto const sum_result   = cache->get_result(col, *sum_agg);
  auto const count_result = cache->get_result(col, *count_agg);

  auto const d_values_ptr = column_device_view::create(col, stream);
  auto const d_sum_ptr    = column_device_view::create(sum_result, stream).release();
  auto const d_count_ptr  = column_device_view::create(count_result, stream).release();

  auto output = make_fixed_width_column(
    cudf::detail::target_type(result_type, agg.kind), col.size(), mask_state::ALL_NULL, stream);
  auto output_view  = mutable_column_device_view::create(output->mutable_view(), stream);
  auto output_tview = mutable_table_view{{output->mutable_view()}};
  cudf::detail::initialize_with_identity(
    output_tview, host_span<cudf::aggregation::Kind const>(&agg.kind, 1), stream);

  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0),
    col.size(),
    m2_hash_functor{
      d_output_index_map, d_row_bitmask, *output_view, *d_values_ptr, *d_sum_ptr, *d_count_ptr});
  cache->add_result(col, agg, std::move(output));
}

void hash_compound_agg_finalizer::visit(cudf::detail::var_aggregation const& agg)
{
  if (cache->has_result(col, agg)) { return; }

  auto const sum_agg      = make_sum_aggregation();
  auto const count_agg    = make_count_aggregation();
  auto const sum_result   = cache->get_result(col, *sum_agg);
  auto const count_result = cache->get_result(col, *count_agg);

  auto values_view = column_device_view::create(col, stream);
  auto sum_view    = column_device_view::create(sum_result, stream);
  auto count_view  = column_device_view::create(count_result, stream);

  auto var_result = make_fixed_width_column(
    cudf::detail::target_type(result_type, agg.kind), col.size(), mask_state::ALL_NULL, stream);
  auto var_result_view = mutable_column_device_view::create(var_result->mutable_view(), stream);
  mutable_table_view var_table_view{{var_result->mutable_view()}};
  cudf::detail::initialize_with_identity(
    var_table_view, host_span<cudf::aggregation::Kind const>(&agg.kind, 1), stream);

  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     thrust::make_counting_iterator(0),
                     col.size(),
                     var_hash_functor{d_output_index_map,
                                      d_row_bitmask,
                                      *var_result_view,
                                      *values_view,
                                      *sum_view,
                                      *count_view,
                                      agg._ddof});
  cache->add_result(col, agg, std::move(var_result));
}

void hash_compound_agg_finalizer::visit(cudf::detail::std_aggregation const& agg)
{
  if (cache->has_result(col, agg)) { return; }

  auto const var_agg  = make_variance_aggregation(agg._ddof);
  auto const variance = cache->get_result(col, *var_agg);

  auto result = cudf::detail::unary_operation(variance, unary_operator::SQRT, stream, mr);
  cache->add_result(col, agg, std::move(result));
}

}  // namespace cudf::groupby::detail::hash
