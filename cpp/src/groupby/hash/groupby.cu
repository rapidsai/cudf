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

#include <hash/concurrent_unordered_map.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/types.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/utilities/traits.hpp>

#include <memory>
#include <utility>

namespace cudf {
namespace experimental {
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
constexpr std::array<aggregation::Kind, 5> hash_aggregations{
    aggregation::SUM, aggregation::MIN, aggregation::MAX, aggregation::COUNT,
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
bool constexpr is_hash_aggregation(aggregation::Kind t) {
  // this is a temporary fix due to compiler bug and we can resort back to
  // constexpr once cuda 10.2 becomes RAPIDS's minimum compiler version
  // return array_contains(hash_aggregations, t);
  return (t == aggregation::SUM) or (t == aggregation::MIN) or
         (t == aggregation::MAX) or (t == aggregation::COUNT) or
         (t == aggregation::ARGMIN) or (t == aggregation::ARGMAX);
}

// flatten aggs to filter in single pass aggs 
std::tuple<table_view, std::vector<aggregation::Kind>, std::vector<size_t> >
flatten_single_pass_aggs(std::vector<aggregation_request> const& requests) {
  std::vector<column_view> columns;
  std::vector<aggregation::Kind> agg_kinds;
  std::vector<size_t> col_ids;

  for (size_t i = 0; i < requests.size(); i++) {
    auto const& request = requests[i];
    auto const& agg_v = request.aggregations;

    auto insert_agg = [&agg_kinds, &columns, &col_ids, &request, i]
    (aggregation::Kind k) {
      agg_kinds.push_back(k);
      columns.push_back(request.values);
      col_ids.push_back(i);
    };

    std::for_each(agg_v.begin(), agg_v.end(),
      [&columns, &agg_kinds, &request, &col_ids, insert_agg] 
      (std::unique_ptr<aggregation> const& agg) {

        if (is_hash_aggregation(agg->kind)) {
          if (is_fixed_width(request.values.type())) {
            insert_agg(agg->kind);
          } else if (request.values.type().id() == type_id::STRING) {
            // For string type, only MIN and MAX are supported
            if (agg->kind == aggregation::MIN) {
              insert_agg(aggregation::ARGMIN);
            } else if (agg->kind == aggregation::MAX) {
              insert_agg(aggregation::ARGMAX);
            }
          }
        }
      });
  }
  return std::make_tuple(table_view(columns), 
                         std::move(agg_kinds), std::move(col_ids));
}

/**
 * @brief Gather sparse results into dense using @p gather_map and add to 
 * @p dense_cache
 */
void sparse_to_dense_results(
    std::vector<aggregation_request> const& requests,
    experimental::detail::result_cache const& sparse_results,
    experimental::detail::result_cache& dense_results,
    rmm::device_vector<size_type> const& gather_map,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr)
{
  for (size_t i = 0; i < requests.size(); i++) {
    auto const& agg_v = requests[i].aggregations;
    auto const& col = requests[i].values;

    // Given an aggregation, this will get the result from sparse_results and 
    // convert and return dense, compacted result
    auto to_dense_agg_result =
    [&sparse_results, &gather_map, i, mr, stream]
    (auto const& agg) {
      auto s = sparse_results.get_result(i, agg);
      auto dense_result_table = 
        experimental::detail::gather(
          table_view({s}),
          gather_map.begin(),
          gather_map.end(),
          false, false, false, mr, stream);
      auto dense_result = std::move(dense_result_table->release()[0]);
      return dense_result;
    };

    // Enables conversion of ARGMIN/ARGMAX into MIN/MAX
    auto transformed_result =
    [&col, to_dense_agg_result, mr, stream]
    (auto const& agg_kind) {
      auto tranformed_agg = std::make_unique<aggregation>(agg_kind);
      auto argmax_result = to_dense_agg_result(tranformed_agg);
      auto transformed_result = experimental::detail::gather(
        table_view({col}), *argmax_result, false, false, false, mr, stream);
      return std::move(transformed_result->release()[0]);
    };

    std::for_each(agg_v.begin(), agg_v.end(),
      [&sparse_results, &dense_results, to_dense_agg_result, transformed_result,
       &col, i]
      (auto const& agg) {
        if (col.type().id() == type_id::STRING) {
          if (agg->kind == aggregation::MAX) {
            dense_results.add_result(i, agg,
              transformed_result(aggregation::ARGMAX));
          }
          else if (agg->kind == aggregation::MIN) {
            dense_results.add_result(i, agg,
              transformed_result(aggregation::ARGMIN));
          }
        }
        else if (sparse_results.has_result(i, agg)) {
          dense_results.add_result(i, agg, to_dense_agg_result(agg));
        }
      });
  }
}

/**
 * @brief Construct hash map that uses row comparator and row hasher on 
 * @p d_keys table and stores indices
 */
template <bool keys_have_nulls>
auto create_hash_map(table_device_view const& d_keys, bool ignore_null_keys,
                     cudaStream_t stream = 0)
{
  size_type constexpr unused_key{std::numeric_limits<size_type>::max()};
  size_type constexpr unused_value{std::numeric_limits<size_type>::max()};

  using map_type =
      concurrent_unordered_map<size_type, size_type,
                              row_hasher<default_hash, keys_have_nulls>,
                              row_equality_comparator<keys_have_nulls>>;

  using allocator_type = typename map_type::allocator_type;

  bool const null_keys_are_equal{not ignore_null_keys};

  row_hasher<default_hash, keys_have_nulls> hasher{d_keys};
  row_equality_comparator<keys_have_nulls> rows_equal{
      d_keys, d_keys, null_keys_are_equal};

  return map_type::create(compute_hash_table_size(d_keys.num_rows()),
                            unused_key, unused_value, hasher, rows_equal,
                            allocator_type(), stream);
}

/**
 * @brief Computes all aggregations from @p requests that require a single pass
 * over the data and stores the results in @p sparse_results
 */
template <bool keys_have_nulls, typename Map>
void compute_single_pass_aggs(table_view const& keys,
                              std::vector<aggregation_request> const& requests,
                              experimental::detail::result_cache& sparse_results,
                              Map& map, bool ignore_null_keys,
                              cudaStream_t stream)
{
  // flatten the aggs to a table that can be operated on by aggregate_row
  table_view flattened_values;
  std::vector<aggregation::Kind> aggs;
  std::vector<size_t> col_ids;
  std::tie(flattened_values, aggs, col_ids) = flatten_single_pass_aggs(requests);

  // make table that will hold sparse results
  std::vector<std::unique_ptr<column>> sparse_columns;
  for (size_t i = 0; i < aggs.size(); i++) {
    auto const& col = flattened_values.column(i);
    bool nullable = (aggs[i] == aggregation::COUNT) ? false : col.has_nulls();
    auto mask_state = (nullable) ? ALL_NULL : UNALLOCATED;
    
    sparse_columns.emplace_back(make_fixed_width_column(
      experimental::detail::target_type(col.type(), aggs[i]),
      col.size(), mask_state, stream));
  }
  table sparse_table(std::move(sparse_columns));
  mutable_table_view table_view = sparse_table.mutable_view();
  experimental::detail::initialize_with_identity(table_view, aggs, stream);

  // prepare to launch kernel to do the actual aggregation
  auto d_sparse_table = mutable_table_device_view::create(sparse_table);
  auto d_values = table_device_view::create(flattened_values);
  rmm::device_vector<aggregation::Kind> d_aggs(aggs);

  bool skip_key_rows_with_nulls = keys_have_nulls and ignore_null_keys;

  experimental::detail::grid_1d grid(keys.num_rows(), 256);
  if (skip_key_rows_with_nulls) {
    auto row_bitmask{bitmask_and(keys, rmm::mr::get_default_resource(), stream)};
    hash::compute_single_pass_aggs<true>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(
        map, keys.num_rows(), *d_values, *d_sparse_table, d_aggs.data().get(),
        static_cast<bitmask_type*>(row_bitmask.data()));
  } else {
    hash::compute_single_pass_aggs<false>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(
        map, keys.num_rows(), *d_values, *d_sparse_table, d_aggs.data().get(),
        nullptr);
  }

  // Add results back to sparse_results cache
  auto sparse_result_cols = sparse_table.release();
  for (size_t i = 0; i < aggs.size(); i++) {
    sparse_results.add_result(col_ids[i],
                              std::make_unique<aggregation>(aggs[i]),
                              std::move(sparse_result_cols[i]));
  }
}

template <bool keys_have_nulls>
auto groupby_null_templated(
    table_view const& keys, std::vector<aggregation_request> const& requests,
    experimental::detail::result_cache& cache,
    bool ignore_null_keys, cudaStream_t stream,
    rmm::mr::device_memory_resource* mr)
{
  auto d_keys = table_device_view::create(keys);
  auto map = create_hash_map<keys_have_nulls>(*d_keys, ignore_null_keys, stream);

  // Cache of sparse results where the location of aggregate value in each
  // column is indexed by the hash map
  experimental::detail::result_cache sparse_results(requests.size());

  // Compute all single pass aggs first
  compute_single_pass_aggs<keys_have_nulls>(
    keys, requests, sparse_results, *map, ignore_null_keys, stream);

  // Now continue with remaining multi-pass aggs
  // <placeholder>

  // Extract the populated indices from the hash map and create a gather map.
  // Gathering using this map from sparse results will give dense results.
  rmm::device_vector<size_type> gather_map(keys.num_rows());
  rmm::device_scalar<size_type> num_groups(0);
  experimental::detail::grid_1d grid(keys.num_rows(), 256);
  extract_gather_map<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>
    (*map, gather_map.data().get(), num_groups.data());
  gather_map.resize(num_groups.value());

  // Compact all results from sparse_results and insert into cache
  sparse_to_dense_results(requests, sparse_results, cache, gather_map, stream, mr);

  // Extract unique keys and return
  auto unique_keys = experimental::detail::gather(
    keys, gather_map.begin(), gather_map.end(),
    false, false, false, mr, stream);
  return unique_keys;
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
bool can_use_hash_groupby(table_view const& keys,
                      std::vector<aggregation_request> const& requests) {
  return std::all_of(
      requests.begin(), requests.end(), [](aggregation_request const& r) {
        return std::all_of(
            r.aggregations.begin(), r.aggregations.end(),
            [](auto const& a) { return is_hash_aggregation(a->kind); });
      });
}

// Hash-based groupby
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby(
    table_view const& keys, std::vector<aggregation_request> const& requests,
    bool ignore_null_keys, cudaStream_t stream,
    rmm::mr::device_memory_resource* mr)
{
  experimental::detail::result_cache cache(requests.size());

  std::unique_ptr<table> unique_keys;
  if (has_nulls(keys)) {
    unique_keys = groupby_null_templated<true>(keys, requests, cache, 
                                               ignore_null_keys, stream, mr);
  } else {
    unique_keys = groupby_null_templated<false>(keys, requests, cache,
                                                ignore_null_keys, stream, mr);
  }

  return std::make_pair(std::move(unique_keys), extract_results(requests, cache));  
}
}  // namespace hash
}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
