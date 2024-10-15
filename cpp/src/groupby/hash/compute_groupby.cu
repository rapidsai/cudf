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

#include "flatten_single_pass_aggs.hpp"
#include "groupby_kernels.cuh"
#include "helpers.cuh"
#include "sparse_to_dense_results.hpp"
#include "var_hash_functor.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuco/static_set.cuh>

#include <memory>

namespace cudf::groupby::detail::hash {
// make table that will hold sparse results
auto create_sparse_results_table(table_view const& flattened_values,
                                 std::vector<aggregation::Kind> aggs,
                                 rmm::cuda_stream_view stream)
{
  // TODO single allocation - room for performance improvement
  std::vector<std::unique_ptr<column>> sparse_columns;
  std::transform(
    flattened_values.begin(),
    flattened_values.end(),
    aggs.begin(),
    std::back_inserter(sparse_columns),
    [stream](auto const& col, auto const& agg) {
      bool nullable =
        (agg == aggregation::COUNT_VALID or agg == aggregation::COUNT_ALL)
          ? false
          : (col.has_nulls() or agg == aggregation::VARIANCE or agg == aggregation::STD);
      auto mask_flag = (nullable) ? mask_state::ALL_NULL : mask_state::UNALLOCATED;

      auto col_type = cudf::is_dictionary(col.type())
                        ? cudf::dictionary_column_view(col).keys().type()
                        : col.type();

      return make_fixed_width_column(
        cudf::detail::target_type(col_type, agg), col.size(), mask_flag, stream);
    });

  table sparse_table(std::move(sparse_columns));
  mutable_table_view table_view = sparse_table.mutable_view();
  cudf::detail::initialize_with_identity(table_view, aggs, stream);
  return sparse_table;
}

/**
 * @brief Computes all aggregations from `requests` that require a single pass
 * over the data and stores the results in `sparse_results`
 */
template <typename SetType>
void compute_single_pass_aggs(table_view const& keys,
                              host_span<aggregation_request const> requests,
                              SetType set,
                              bool skip_rows_with_nulls,
                              bitmask_type const* row_bitmask,
                              cudf::detail::result_cache* sparse_results,
                              rmm::cuda_stream_view stream)
{
  // flatten the aggs to a table that can be operated on by aggregate_row
  auto const [flattened_values, agg_kinds, aggs] = flatten_single_pass_aggs(requests);

  // make table that will hold sparse results
  table sparse_table = create_sparse_results_table(flattened_values, agg_kinds, stream);
  // prepare to launch kernel to do the actual aggregation
  auto d_sparse_table = mutable_table_device_view::create(sparse_table, stream);
  auto d_values       = table_device_view::create(flattened_values, stream);
  auto const d_aggs   = cudf::detail::make_device_uvector_async(
    agg_kinds, stream, cudf::get_current_device_resource_ref());

  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    keys.num_rows(),
    hash::compute_single_pass_aggs_fn{
      set, *d_values, *d_sparse_table, d_aggs.data(), row_bitmask, skip_rows_with_nulls});
  // Add results back to sparse_results cache
  auto sparse_result_cols = sparse_table.release();
  for (size_t i = 0; i < aggs.size(); i++) {
    // Note that the cache will make a copy of this temporary aggregation
    sparse_results->add_result(
      flattened_values.column(i), *aggs[i], std::move(sparse_result_cols[i]));
  }
}

/**
 * @brief Computes and returns a device vector containing all populated keys in
 * `map`.
 */
template <typename SetType>
rmm::device_uvector<size_type> extract_populated_keys(SetType const& key_set,
                                                      size_type num_keys,
                                                      rmm::cuda_stream_view stream)
{
  rmm::device_uvector<size_type> populated_keys(num_keys, stream);
  auto const keys_end = key_set.retrieve_all(populated_keys.begin(), stream.value());

  populated_keys.resize(std::distance(populated_keys.begin(), keys_end), stream);
  return populated_keys;
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
std::unique_ptr<table> compute_groupby(table_view const& keys,
                                       host_span<aggregation_request const> requests,
                                       bool skip_rows_with_nulls,
                                       Equal const& d_row_equal,
                                       row_hash_t const& d_row_hash,
                                       cudf::detail::result_cache* cache,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  // convert to int64_t to avoid potential overflow with large `keys`
  auto const num_keys = static_cast<int64_t>(keys.num_rows());

  // Cache of sparse results where the location of aggregate value in each
  // column is indexed by the hash set
  cudf::detail::result_cache sparse_results(requests.size());

  auto const set = cuco::static_set{
    num_keys,
    0.5,  // desired load factor
    cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
    d_row_equal,
    probing_scheme_t{d_row_hash},
    cuco::thread_scope_device,
    cuco::storage<1>{},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
    stream.value()};

  auto row_bitmask =
    skip_rows_with_nulls
      ? cudf::bitmask_and(keys, stream, cudf::get_current_device_resource_ref()).first
      : rmm::device_buffer{};

  // Compute all single pass aggs first
  compute_single_pass_aggs(keys,
                           requests,
                           set.ref(cuco::insert_and_find),
                           skip_rows_with_nulls,
                           static_cast<bitmask_type*>(row_bitmask.data()),
                           &sparse_results,
                           stream);

  // Extract the populated indices from the hash set and create a gather map.
  // Gathering using this map from sparse results will give dense results.
  auto gather_map = extract_populated_keys(set, keys.num_rows(), stream);

  // Compact all results from sparse_results and insert into cache
  sparse_to_dense_results(requests,
                          &sparse_results,
                          cache,
                          gather_map,
                          set.ref(cuco::find),
                          static_cast<bitmask_type*>(row_bitmask.data()),
                          stream,
                          mr);

  return cudf::detail::gather(keys,
                              gather_map,
                              out_of_bounds_policy::DONT_CHECK,
                              cudf::detail::negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr);
}

template std::unique_ptr<table> compute_groupby<row_comparator_t>(
  table_view const& keys,
  host_span<aggregation_request const> requests,
  bool skip_rows_with_nulls,
  row_comparator_t const& d_row_equal,
  row_hash_t const& d_row_hash,
  cudf::detail::result_cache* cache,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<table> compute_groupby<nullable_row_comparator_t>(
  table_view const& keys,
  host_span<aggregation_request const> requests,
  bool skip_rows_with_nulls,
  nullable_row_comparator_t const& d_row_equal,
  row_hash_t const& d_row_hash,
  cudf::detail::result_cache* cache,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
}  // namespace cudf::groupby::detail::hash
