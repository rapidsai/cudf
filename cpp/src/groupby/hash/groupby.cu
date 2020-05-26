/*
 * Copyright (c) 2019-20, NVIDIA CORPORATION.
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

#include <groupby/common/utils.hpp>
#include <groupby/hash/groupby_kernels.cuh>

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <hash/concurrent_unordered_map.cuh>

#include <memory>
#include <utility>

namespace cudf {
namespace groupby {
namespace detail {
namespace hash {
namespace {
// This is a temporary fix due to compiler bug and we can resort back to
// constexpr once cuda 10.2 becomes RAPIDS's minimum compiler version
#if 0
/**
 * @brief List of aggregation operations that can be computed with a hash-based
 * implementation.
 */
constexpr std::array<aggregation::Kind, 7> hash_aggregations{
    aggregation::SUM, aggregation::MIN, aggregation::MAX,
    aggregation::COUNT_VALID, aggregation::COUNT_ALL,
    aggregation::ARGMIN, aggregation::ARGMAX};

template <class T, size_t N>
constexpr bool array_contains(std::array<T, N> const& haystack, T needle) {
  for (auto i = 0u; i < N; ++i) {
    if (haystack[i] == needle) return true;
  }
  return false;
}
#endif

/**
 * @brief Indicates whether the specified aggregation operation can be computed
 * with a hash-based implementation.
 *
 * @param t The aggregation operation to verify
 * @return true `t` is valid for a hash based groupby
 * @return false `t` is invalid for a hash based groupby
 */
bool constexpr is_hash_aggregation(aggregation::Kind t)
{
  // this is a temporary fix due to compiler bug and we can resort back to
  // constexpr once cuda 10.2 becomes RAPIDS's minimum compiler version
  // return array_contains(hash_aggregations, t);
  return (t == aggregation::SUM) or (t == aggregation::MIN) or (t == aggregation::MAX) or
         (t == aggregation::COUNT_VALID) or (t == aggregation::COUNT_ALL) or
         (t == aggregation::ARGMIN) or (t == aggregation::ARGMAX);
}

// flatten aggs to filter in single pass aggs
std::tuple<table_view, std::vector<aggregation::Kind>, std::vector<size_t>>
flatten_single_pass_aggs(std::vector<aggregation_request> const& requests)
{
  std::vector<column_view> columns;
  std::vector<aggregation::Kind> agg_kinds;
  std::vector<size_t> col_ids;

  for (size_t i = 0; i < requests.size(); i++) {
    auto const& request = requests[i];
    auto const& agg_v   = request.aggregations;

    auto insert_agg = [&agg_kinds, &columns, &col_ids, &request, i](aggregation::Kind k) {
      agg_kinds.push_back(k);
      columns.push_back(request.values);
      col_ids.push_back(i);
    };

    for (auto&& agg : agg_v) {
      if (is_hash_aggregation(agg->kind)) {
        if (is_fixed_width(request.values.type()) or agg->kind == aggregation::COUNT_VALID or
            agg->kind == aggregation::COUNT_ALL) {
          insert_agg(agg->kind);
        } else if (request.values.type().id() == type_id::STRING) {
          // For string type, only ARGMIN, ARGMAX, MIN, and MAX are supported
          if (agg->kind == aggregation::ARGMIN or agg->kind == aggregation::ARGMAX) {
            insert_agg(agg->kind);
          } else if (agg->kind == aggregation::MIN) {
            insert_agg(aggregation::ARGMIN);
          } else if (agg->kind == aggregation::MAX) {
            insert_agg(aggregation::ARGMAX);
          }
        }
      }
    }
  }
  return std::make_tuple(table_view(columns), std::move(agg_kinds), std::move(col_ids));
}

/**
 * @brief Gather sparse results into dense using `gather_map` and add to
 * `dense_cache`
 *
 * @see groupby_null_templated()
 */
void sparse_to_dense_results(std::vector<aggregation_request> const& requests,
                             cudf::detail::result_cache const& sparse_results,
                             cudf::detail::result_cache* dense_results,
                             rmm::device_vector<size_type> const& gather_map,
                             size_type map_size,
                             cudaStream_t stream,
                             rmm::mr::device_memory_resource* mr)
{
  for (size_t i = 0; i < requests.size(); i++) {
    auto const& agg_v = requests[i].aggregations;
    auto const& col   = requests[i].values;

    // Given an aggregation, this will get the result from sparse_results and
    // convert and return dense, compacted result
    auto to_dense_agg_result =
      [&sparse_results, &gather_map, map_size, i, mr, stream](auto const& agg) {
        auto s                  = sparse_results.get_result(i, agg);
        auto dense_result_table = cudf::detail::gather(
          table_view({s}), gather_map.begin(), gather_map.begin() + map_size, false, mr, stream);
        return std::move(dense_result_table->release()[0]);
      };

    // Enables conversion of ARGMIN/ARGMAX into MIN/MAX
    auto transformed_result = [&col, to_dense_agg_result, mr, stream](auto const& agg_kind) {
      auto transformed_agg = std::make_unique<aggregation>(agg_kind);
      auto arg_result      = to_dense_agg_result(*transformed_agg);
      // We make a view of ARG(MIN/MAX) result without a null mask and gather
      // using this map. The values in data buffer of ARG(MIN/MAX) result
      // corresponding to null values was initialized to ARG(MIN/MAX)_SENTINEL
      // which is an out of bounds index value (-1) and causes the gathered
      // value to be null.
      column_view null_removed_map(
        data_type(type_to_id<size_type>()),
        arg_result->size(),
        static_cast<void const*>(arg_result->view().template data<size_type>()));
      auto transformed_result =
        cudf::detail::gather(table_view({col}),
                             null_removed_map,
                             arg_result->nullable() ? cudf::detail::out_of_bounds_policy::IGNORE
                                                    : cudf::detail::out_of_bounds_policy::NULLIFY,
                             cudf::detail::negative_index_policy::NOT_ALLOWED,
                             mr,
                             stream);
      return std::move(transformed_result->release()[0]);
    };

    for (auto&& agg : agg_v) {
      auto const& agg_ref = *agg;
      if (agg->kind == aggregation::COUNT_VALID or agg->kind == aggregation::COUNT_ALL) {
        dense_results->add_result(i, agg_ref, to_dense_agg_result(agg_ref));
      } else if (col.type().id() == type_id::STRING and
                 (agg->kind == aggregation::MAX or agg->kind == aggregation::MIN)) {
        if (agg->kind == aggregation::MAX) {
          dense_results->add_result(i, agg_ref, transformed_result(aggregation::ARGMAX));
        } else if (agg->kind == aggregation::MIN) {
          dense_results->add_result(i, agg_ref, transformed_result(aggregation::ARGMIN));
        }
      } else if (sparse_results.has_result(i, agg_ref)) {
        dense_results->add_result(i, agg_ref, to_dense_agg_result(agg_ref));
      }
    }
  }
}

/**
 * @brief Construct hash map that uses row comparator and row hasher on
 * `d_keys` table and stores indices
 */
template <bool keys_have_nulls>
auto create_hash_map(table_device_view const& d_keys,
                     null_policy include_null_keys,
                     cudaStream_t stream = 0)
{
  size_type constexpr unused_key{std::numeric_limits<size_type>::max()};
  size_type constexpr unused_value{std::numeric_limits<size_type>::max()};

  using map_type = concurrent_unordered_map<size_type,
                                            size_type,
                                            row_hasher<default_hash, keys_have_nulls>,
                                            row_equality_comparator<keys_have_nulls>>;

  using allocator_type = typename map_type::allocator_type;

  bool const null_keys_are_equal{include_null_keys == null_policy::INCLUDE};

  row_hasher<default_hash, keys_have_nulls> hasher{d_keys};
  row_equality_comparator<keys_have_nulls> rows_equal{d_keys, d_keys, null_keys_are_equal};

  return map_type::create(compute_hash_table_size(d_keys.num_rows()),
                          unused_key,
                          unused_value,
                          hasher,
                          rows_equal,
                          allocator_type(),
                          stream);
}

/**
 * @brief Computes all aggregations from `requests` that require a single pass
 * over the data and stores the results in `sparse_results`
 *
 * @see groupby_null_templated()
 */
template <bool keys_have_nulls, typename Map>
void compute_single_pass_aggs(table_view const& keys,
                              std::vector<aggregation_request> const& requests,
                              cudf::detail::result_cache* sparse_results,
                              Map& map,
                              null_policy include_null_keys,
                              cudaStream_t stream)
{
  // flatten the aggs to a table that can be operated on by aggregate_row
  table_view flattened_values;
  std::vector<aggregation::Kind> aggs;
  std::vector<size_t> col_ids;
  std::tie(flattened_values, aggs, col_ids) = flatten_single_pass_aggs(requests);

  // make table that will hold sparse results
  std::vector<std::unique_ptr<column>> sparse_columns;
  std::transform(flattened_values.begin(),
                 flattened_values.end(),
                 aggs.begin(),
                 std::back_inserter(sparse_columns),
                 [stream](auto const& col, auto const& agg) {
                   bool nullable =
                     (agg == aggregation::COUNT_VALID or agg == aggregation::COUNT_ALL)
                       ? false
                       : col.has_nulls();
                   auto mask_flag = (nullable) ? mask_state::ALL_NULL : mask_state::UNALLOCATED;

                   return make_fixed_width_column(
                     cudf::detail::target_type(col.type(), agg), col.size(), mask_flag, stream);
                 });

  table sparse_table(std::move(sparse_columns));
  mutable_table_view table_view = sparse_table.mutable_view();
  cudf::detail::initialize_with_identity(table_view, aggs, stream);

  // prepare to launch kernel to do the actual aggregation
  auto d_sparse_table = mutable_table_device_view::create(sparse_table);
  auto d_values       = table_device_view::create(flattened_values);
  rmm::device_vector<aggregation::Kind> d_aggs(aggs);

  bool skip_key_rows_with_nulls = keys_have_nulls and include_null_keys == null_policy::EXCLUDE;

  if (skip_key_rows_with_nulls) {
    auto row_bitmask{bitmask_and(keys, rmm::mr::get_default_resource(), stream)};
    thrust::for_each_n(
      rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator(0),
      keys.num_rows(),
      hash::compute_single_pass_aggs<true, Map>{map,
                                                keys.num_rows(),
                                                *d_values,
                                                *d_sparse_table,
                                                d_aggs.data().get(),
                                                static_cast<bitmask_type*>(row_bitmask.data())});
  } else {
    thrust::for_each_n(
      rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator(0),
      keys.num_rows(),
      hash::compute_single_pass_aggs<false, Map>{
        map, keys.num_rows(), *d_values, *d_sparse_table, d_aggs.data().get(), nullptr});
  }

  // Add results back to sparse_results cache
  auto sparse_result_cols = sparse_table.release();
  for (size_t i = 0; i < aggs.size(); i++) {
    // Note that the cache will make a copy of this temporary aggregation
    auto agg = std::make_unique<aggregation>(aggs[i]);
    sparse_results->add_result(col_ids[i], *agg, std::move(sparse_result_cols[i]));
  }
}

/**
 * @brief Computes and returns a device vector containing all populated keys in
 * `map`.
 */
template <typename Map>
std::pair<rmm::device_vector<size_type>, size_type> extract_populated_keys(Map map,
                                                                           size_type num_keys,
                                                                           cudaStream_t stream = 0)
{
  rmm::device_vector<size_type> populated_keys(num_keys);

  auto get_key = [] __device__(auto const& element) {
    size_type key, value;
    thrust::tie(key, value) = element;
    return key;
  };

  auto end_it = thrust::copy_if(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_transform_iterator(map.data(), get_key),
    thrust::make_transform_iterator(map.data() + map.capacity(), get_key),
    populated_keys.begin(),
    [unused_key = map.get_unused_key()] __device__(size_type key) { return key != unused_key; });

  size_type map_size = end_it - populated_keys.begin();

  return std::make_pair(std::move(populated_keys), map_size);
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
 *
 */
template <bool keys_have_nulls>
std::unique_ptr<table> groupby_null_templated(table_view const& keys,
                                              std::vector<aggregation_request> const& requests,
                                              cudf::detail::result_cache* cache,
                                              null_policy include_null_keys,
                                              cudaStream_t stream,
                                              rmm::mr::device_memory_resource* mr)
{
  auto d_keys = table_device_view::create(keys);
  auto map    = create_hash_map<keys_have_nulls>(*d_keys, include_null_keys, stream);

  // Cache of sparse results where the location of aggregate value in each
  // column is indexed by the hash map
  cudf::detail::result_cache sparse_results(requests.size());

  // Compute all single pass aggs first
  compute_single_pass_aggs<keys_have_nulls>(
    keys, requests, &sparse_results, *map, include_null_keys, stream);

  // Now continue with remaining multi-pass aggs
  // <placeholder>

  // Extract the populated indices from the hash map and create a gather map.
  // Gathering using this map from sparse results will give dense results.
  rmm::device_vector<size_type> gather_map;
  size_type map_size;
  std::tie(gather_map, map_size) = extract_populated_keys(*map, keys.num_rows(), stream);

  // Compact all results from sparse_results and insert into cache
  sparse_to_dense_results(requests, sparse_results, cache, gather_map, map_size, stream, mr);

  return cudf::detail::gather(
    keys, gather_map.begin(), gather_map.begin() + map_size, false, mr, stream);
}

}  // namespace

/**
 * @brief Indicates if a set of aggregation requests can be satisfied with a
 * hash-based groupby implementation.
 *
 * @param keys The table of keys
 * @param requests The set of columns to aggregate and the aggregations to
 * perform
 * @return true A hash-based groupby should be used
 * @return false A hash-based groupby should not be used
 */
bool can_use_hash_groupby(table_view const& keys, std::vector<aggregation_request> const& requests)
{
  return std::all_of(requests.begin(), requests.end(), [](aggregation_request const& r) {
    return std::all_of(r.aggregations.begin(), r.aggregations.end(), [](auto const& a) {
      return is_hash_aggregation(a->kind);
    });
  });
}

// Hash-based groupby
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby(
  table_view const& keys,
  std::vector<aggregation_request> const& requests,
  null_policy include_null_keys,
  cudaStream_t stream,
  rmm::mr::device_memory_resource* mr)
{
  cudf::detail::result_cache cache(requests.size());

  std::unique_ptr<table> unique_keys;
  if (has_nulls(keys)) {
    unique_keys =
      groupby_null_templated<true>(keys, requests, &cache, include_null_keys, stream, mr);
  } else {
    unique_keys =
      groupby_null_templated<false>(keys, requests, &cache, include_null_keys, stream, mr);
  }

  return std::make_pair(std::move(unique_keys), extract_results(requests, cache));
}
}  // namespace hash
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
