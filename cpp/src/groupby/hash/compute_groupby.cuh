/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#pragma once

#include "compute_single_pass_aggs.cuh"
#include "compute_single_pass_aggs.hpp"
#include "multi_pass_functors.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/unary.hpp>

namespace cudf::groupby::detail::hash {

template <typename SetType>
class hash_compound_agg_finalizer final : public cudf::detail::aggregation_finalizer {
  column_view col;
  data_type result_type;
  cudf::detail::result_cache* sparse_results;
  cudf::detail::result_cache* dense_results;
  device_span<size_type const> gather_map;
  SetType set;
  bitmask_type const* __restrict__ row_bitmask;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

 public:
  using cudf::detail::aggregation_finalizer::visit;

  hash_compound_agg_finalizer(column_view col,
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
    result_type = cudf::is_dictionary(col.type()) ? cudf::dictionary_column_view(col).keys().type()
                                                  : col.type();
  }

  auto to_dense_agg_result(cudf::aggregation const& agg)
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

  // Enables conversion of ARGMIN/ARGMAX into MIN/MAX
  auto gather_argminmax(aggregation const& agg)
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

  // Declare overloads for each kind of aggregation to dispatch
  void visit(cudf::aggregation const& agg) override
  {
    if (dense_results->has_result(col, agg)) return;
    dense_results->add_result(col, agg, to_dense_agg_result(agg));
  }

  void visit(cudf::detail::min_aggregation const& agg) override
  {
    if (dense_results->has_result(col, agg)) return;
    if (result_type.id() == type_id::STRING) {
      auto transformed_agg = make_argmin_aggregation();
      dense_results->add_result(col, agg, gather_argminmax(*transformed_agg));
    } else {
      dense_results->add_result(col, agg, to_dense_agg_result(agg));
    }
  }

  void visit(cudf::detail::max_aggregation const& agg) override
  {
    if (dense_results->has_result(col, agg)) return;

    if (result_type.id() == type_id::STRING) {
      auto transformed_agg = make_argmax_aggregation();
      dense_results->add_result(col, agg, gather_argminmax(*transformed_agg));
    } else {
      dense_results->add_result(col, agg, to_dense_agg_result(agg));
    }
  }

  void visit(cudf::detail::mean_aggregation const& agg) override
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

  void visit(cudf::detail::var_aggregation const& agg) override
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
    cudf::detail::initialize_with_identity(var_table_view, {agg.kind}, stream);

    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      col.size(),
      ::cudf::detail::var_hash_functor{
        set, row_bitmask, *var_result_view, *values_view, *sum_view, *count_view, agg._ddof});
    sparse_results->add_result(col, agg, std::move(var_result));
    dense_results->add_result(col, agg, to_dense_agg_result(agg));
  }

  void visit(cudf::detail::std_aggregation const& agg) override
  {
    if (dense_results->has_result(col, agg)) return;
    auto var_agg = make_variance_aggregation(agg._ddof);
    this->visit(*dynamic_cast<cudf::detail::var_aggregation*>(var_agg.get()));
    column_view variance = dense_results->get_result(col, *var_agg);

    auto result = cudf::detail::unary_operation(variance, unary_operator::SQRT, stream, mr);
    dense_results->add_result(col, agg, std::move(result));
  }
};

/**
 * @brief Gather sparse results into dense using `gather_map` and add to
 * `dense_cache`
 *
 * @see groupby_null_templated()
 */
template <typename SetType>
void sparse_to_dense_results(table_view const& keys,
                             host_span<aggregation_request const> requests,
                             cudf::detail::result_cache* sparse_results,
                             cudf::detail::result_cache* dense_results,
                             device_span<size_type const> gather_map,
                             SetType set,
                             bool skip_key_rows_with_nulls,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  auto row_bitmask =
    cudf::detail::bitmask_and(keys, stream, cudf::get_current_device_resource_ref()).first;
  bitmask_type const* row_bitmask_ptr =
    skip_key_rows_with_nulls ? static_cast<bitmask_type*>(row_bitmask.data()) : nullptr;

  for (auto const& request : requests) {
    auto const& agg_v = request.aggregations;
    auto const& col   = request.values;

    // Given an aggregation, this will get the result from sparse_results and
    // convert and return dense, compacted result
    auto finalizer = hash_compound_agg_finalizer(
      col, sparse_results, dense_results, gather_map, set, row_bitmask_ptr, stream, mr);
    for (auto&& agg : agg_v) {
      agg->finalize(finalizer);
    }
  }
}

/**
 * @brief Computes groupby using hash table.
 *
 * First, we create a hash table that stores the indices of unique rows in
 * `keys`. The upper limit on the number of values in this map is the number
 * of rows in `keys`.
 *
 * To store the results of aggregations, we create temporary sparse columns
 * which have the same size as input value columns. Using the hash map, we
 * determine the location within the sparse column to write the result of the
 * aggregation into.
 *
 * The sparse column results of all aggregations are stored into the cache
 * `sparse_results`. This enables the use of previously calculated results in
 * other aggregations.
 *
 * All the aggregations which can be computed in a single pass are computed
 * first, in a combined kernel. Then using these results, aggregations that
 * require multiple passes, will be computed.
 *
 * Finally, using the hash map, we generate a vector of indices of populated
 * values in sparse result columns. Then, for each aggregation originally
 * requested in `requests`, we gather sparse results into a column of dense
 * results using the aforementioned index vector. Dense results are stored into
 * the in/out parameter `cache`.
 */
template <typename Equal>
std::unique_ptr<table> compute_groupby(
  table_view const& keys,
  host_span<aggregation_request const> requests,
  cudf::detail::result_cache* cache,
  bool skip_key_rows_with_nulls,
  Equal const& d_row_equal,
  cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                   cudf::nullate::DYNAMIC> const& d_row_hash,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // convert to int64_t to avoid potential overflow with large `keys`
  auto const num_keys = static_cast<int64_t>(keys.num_rows());

  // Cache of sparse results where the location of aggregate value in each
  // column is indexed by the hash set
  cudf::detail::result_cache sparse_results(requests.size());

  auto const set = cuco::static_set{
    cuco::extent<int64_t>{num_keys},
    cudf::detail::CUCO_DESIRED_LOAD_FACTOR,  // 50% occupancy
    cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
    d_row_equal,
    probing_scheme_t{d_row_hash},
    cuco::thread_scope_device,
    cuco::storage<GROUPBY_WINDOW_SIZE>{},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
    stream.value()};

  // Compute all single pass aggs first
  auto gather_map = compute_single_pass_aggs(
    keys, requests, &sparse_results, set, skip_key_rows_with_nulls, stream);

  // Compact all results from sparse_results and insert into cache
  sparse_to_dense_results(keys,
                          requests,
                          &sparse_results,
                          cache,
                          gather_map,
                          set.ref(cuco::find),
                          skip_key_rows_with_nulls,
                          stream,
                          mr);

  return cudf::detail::gather(keys,
                              gather_map,
                              out_of_bounds_policy::DONT_CHECK,
                              cudf::detail::negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr);
}

}  // namespace cudf::groupby::detail::hash
