/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include "var_hash_functor.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>

namespace cudf::groupby::detail::hash {
template <typename SetType>
hash_compound_agg_finalizer<SetType>::hash_compound_agg_finalizer(
  column_view col,
  cudf::detail::result_cache* sparse_results,
  cudf::detail::result_cache* dense_results,
  device_span<size_type const> gather_map,
  SetType set,
  bitmask_type const* row_bitmask,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
  : col(col),
    sparse_results(sparse_results),
    dense_results(dense_results),
    gather_map(gather_map),
    set(set),
    row_bitmask(row_bitmask),
    stream(stream),
    mr(mr)
{
  result_type =
    cudf::is_dictionary(col.type()) ? cudf::dictionary_column_view(col).keys().type() : col.type();
}

template <typename SetType>
auto hash_compound_agg_finalizer<SetType>::to_dense_agg_result(cudf::aggregation const& agg)
{
  auto s                  = sparse_results->get_result(col, agg);
  auto dense_result_table = cudf::detail::gather(table_view({std::move(s)}),
                                                 gather_map,
                                                 out_of_bounds_policy::DONT_CHECK,
                                                 cudf::detail::negative_index_policy::NOT_ALLOWED,
                                                 stream,
                                                 mr);
  return std::move(dense_result_table->release()[0]);
}

template <typename SetType>
auto hash_compound_agg_finalizer<SetType>::gather_argminmax(aggregation const& agg)
{
  auto arg_result = to_dense_agg_result(agg);
  // We make a view of ARG(MIN/MAX) result without a null mask and gather
  // using this map. The values in data buffer of ARG(MIN/MAX) result
  // corresponding to null values was initialized to ARG(MIN/MAX)_SENTINEL
  // which is an out of bounds index value (-1) and causes the gathered
  // value to be null.
  column_view null_removed_map(
    data_type(type_to_id<size_type>()),
    arg_result->size(),
    static_cast<void const*>(arg_result->view().template data<size_type>()),
    nullptr,
    0);
  auto gather_argminmax =
    cudf::detail::gather(table_view({col}),
                         null_removed_map,
                         arg_result->nullable() ? cudf::out_of_bounds_policy::NULLIFY
                                                : cudf::out_of_bounds_policy::DONT_CHECK,
                         cudf::detail::negative_index_policy::NOT_ALLOWED,
                         stream,
                         mr);
  return std::move(gather_argminmax->release()[0]);
}

template <typename SetType>
void hash_compound_agg_finalizer<SetType>::visit(cudf::aggregation const& agg)
{
  if (dense_results->has_result(col, agg)) return;
  dense_results->add_result(col, agg, to_dense_agg_result(agg));
}

template <typename SetType>
void hash_compound_agg_finalizer<SetType>::visit(cudf::detail::min_aggregation const& agg)
{
  if (dense_results->has_result(col, agg)) return;
  if (result_type.id() == type_id::STRING) {
    auto transformed_agg = make_argmin_aggregation();
    dense_results->add_result(col, agg, gather_argminmax(*transformed_agg));
  } else {
    dense_results->add_result(col, agg, to_dense_agg_result(agg));
  }
}

template <typename SetType>
void hash_compound_agg_finalizer<SetType>::visit(cudf::detail::max_aggregation const& agg)
{
  if (dense_results->has_result(col, agg)) return;

  if (result_type.id() == type_id::STRING) {
    auto transformed_agg = make_argmax_aggregation();
    dense_results->add_result(col, agg, gather_argminmax(*transformed_agg));
  } else {
    dense_results->add_result(col, agg, to_dense_agg_result(agg));
  }
}

template <typename SetType>
void hash_compound_agg_finalizer<SetType>::visit(cudf::detail::mean_aggregation const& agg)
{
  if (dense_results->has_result(col, agg)) return;

  auto sum_agg   = make_sum_aggregation();
  auto count_agg = make_count_aggregation();
  this->visit(*sum_agg);
  this->visit(*count_agg);
  column_view sum_result   = dense_results->get_result(col, *sum_agg);
  column_view count_result = dense_results->get_result(col, *count_agg);

  auto result =
    cudf::detail::binary_operation(sum_result,
                                   count_result,
                                   binary_operator::DIV,
                                   cudf::detail::target_type(result_type, aggregation::MEAN),
                                   stream,
                                   mr);
  dense_results->add_result(col, agg, std::move(result));
}

template <typename SetType>
void hash_compound_agg_finalizer<SetType>::visit(cudf::detail::var_aggregation const& agg)
{
  if (dense_results->has_result(col, agg)) return;

  auto sum_agg   = make_sum_aggregation();
  auto count_agg = make_count_aggregation();
  this->visit(*sum_agg);
  this->visit(*count_agg);
  column_view sum_result   = sparse_results->get_result(col, *sum_agg);
  column_view count_result = sparse_results->get_result(col, *count_agg);

  auto values_view = column_device_view::create(col, stream);
  auto sum_view    = column_device_view::create(sum_result, stream);
  auto count_view  = column_device_view::create(count_result, stream);

  auto var_result = make_fixed_width_column(
    cudf::detail::target_type(result_type, agg.kind), col.size(), mask_state::ALL_NULL, stream);
  auto var_result_view = mutable_column_device_view::create(var_result->mutable_view(), stream);
  mutable_table_view var_table_view{{var_result->mutable_view()}};
  cudf::detail::initialize_with_identity(
    var_table_view, host_span<cudf::aggregation::Kind const>(&agg.kind, 1), stream);

  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0),
    col.size(),
    var_hash_functor{
      set, row_bitmask, *var_result_view, *values_view, *sum_view, *count_view, agg._ddof});
  sparse_results->add_result(col, agg, std::move(var_result));
  dense_results->add_result(col, agg, to_dense_agg_result(agg));
}

template <typename SetType>
void hash_compound_agg_finalizer<SetType>::visit(cudf::detail::std_aggregation const& agg)
{
  if (dense_results->has_result(col, agg)) return;
  auto var_agg = make_variance_aggregation(agg._ddof);
  this->visit(*dynamic_cast<cudf::detail::var_aggregation*>(var_agg.get()));
  column_view variance = dense_results->get_result(col, *var_agg);

  auto result = cudf::detail::unary_operation(variance, unary_operator::SQRT, stream, mr);
  dense_results->add_result(col, agg, std::move(result));
}

template class hash_compound_agg_finalizer<hash_set_ref_t<cuco::find_tag>>;
template class hash_compound_agg_finalizer<nullable_hash_set_ref_t<cuco::find_tag>>;
}  // namespace cudf::groupby::detail::hash
