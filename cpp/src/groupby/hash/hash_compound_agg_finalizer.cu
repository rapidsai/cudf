/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/common/m2_var_std.hpp"
#include "hash_compound_agg_finalizer.hpp"
#include "helpers.cuh"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf::groupby::detail::hash {

hash_compound_agg_finalizer::hash_compound_agg_finalizer(column_view col,
                                                         cudf::detail::result_cache* cache,
                                                         bitmask_type const* d_row_bitmask,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
  : col{col},
    input_type{is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type()},
    cache{cache},
    d_row_bitmask{d_row_bitmask},
    stream{stream},
    mr{mr}
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
  if (cache->has_result(col, agg)) { return; }
  if (input_type.id() == type_id::STRING) {
    auto transformed_agg = make_argmin_aggregation();
    cache->add_result(col, agg, gather_argminmax(*transformed_agg));
  }  // else: no-op, since this is only relevant for compound aggregations
  // TODO: support other nested types.
}

void hash_compound_agg_finalizer::visit(cudf::detail::max_aggregation const& agg)
{
  if (cache->has_result(col, agg)) { return; }
  if (input_type.id() == type_id::STRING) {
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
                                   cudf::detail::target_type(input_type, aggregation::MEAN),
                                   stream,
                                   mr);
  cache->add_result(col, agg, std::move(result));
}

void hash_compound_agg_finalizer::visit(cudf::detail::m2_aggregation const& agg)
{
  if (cache->has_result(col, agg)) { return; }

  auto const sum_sqr_agg = make_sum_of_squares_aggregation();
  auto const sum_agg     = make_sum_aggregation();
  auto const count_agg   = make_count_aggregation();
  this->visit(*sum_sqr_agg);
  this->visit(*sum_agg);
  this->visit(*count_agg);
  auto const sum_sqr_result = cache->get_result(col, *sum_sqr_agg);
  auto const sum_result     = cache->get_result(col, *sum_agg);
  auto const count_result   = cache->get_result(col, *count_agg);

  auto output = compute_m2(input_type, sum_sqr_result, sum_result, count_result, stream, mr);
  cache->add_result(col, agg, std::move(output));
}

void hash_compound_agg_finalizer::visit(cudf::detail::var_aggregation const& agg)
{
  if (cache->has_result(col, agg)) { return; }

  auto const m2_agg    = make_m2_aggregation();
  auto const count_agg = make_count_aggregation();
  this->visit(*dynamic_cast<cudf::detail::m2_aggregation*>(m2_agg.get()));
  this->visit(*count_agg);
  auto const m2_result    = cache->get_result(col, *m2_agg);
  auto const count_result = cache->get_result(col, *count_agg);

  auto output = compute_variance(m2_result, count_result, agg._ddof, stream, mr);
  cache->add_result(col, agg, std::move(output));
}

void hash_compound_agg_finalizer::visit(cudf::detail::std_aggregation const& agg)
{
  if (cache->has_result(col, agg)) { return; }

  auto const m2_agg    = make_m2_aggregation();
  auto const count_agg = make_count_aggregation();
  this->visit(*dynamic_cast<cudf::detail::m2_aggregation*>(m2_agg.get()));
  this->visit(*count_agg);
  auto const m2_result    = cache->get_result(col, *m2_agg);
  auto const count_result = cache->get_result(col, *count_agg);

  auto output = compute_std(m2_result, count_result, agg._ddof, stream, mr);
  cache->add_result(col, agg, std::move(output));
}

}  // namespace cudf::groupby::detail::hash
