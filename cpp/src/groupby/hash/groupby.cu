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

#include "compute_groupby.hpp"
#include "groupby/common/utils.hpp"
#include "helpers.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.cuh>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace cudf::groupby::detail::hash {
namespace {
/**
 * @brief List of aggregation operations that can be computed with a hash-based
 * implementation.
 */
constexpr std::array<aggregation::Kind, 12> hash_aggregations{aggregation::SUM,
                                                              aggregation::PRODUCT,
                                                              aggregation::MIN,
                                                              aggregation::MAX,
                                                              aggregation::COUNT_VALID,
                                                              aggregation::COUNT_ALL,
                                                              aggregation::ARGMIN,
                                                              aggregation::ARGMAX,
                                                              aggregation::SUM_OF_SQUARES,
                                                              aggregation::MEAN,
                                                              aggregation::STD,
                                                              aggregation::VARIANCE};

// Could be hash: SUM, PRODUCT, MIN, MAX, COUNT_VALID, COUNT_ALL, ANY, ALL,
// Compound: MEAN(SUM, COUNT_VALID), VARIANCE, STD(MEAN (SUM, COUNT_VALID), COUNT_VALID),
// ARGMAX, ARGMIN

// TODO replace with std::find in C++20 onwards.
template <class T, size_t N>
constexpr bool array_contains(std::array<T, N> const& haystack, T needle)
{
  for (auto const& val : haystack) {
    if (val == needle) return true;
  }
  return false;
}

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
  return array_contains(hash_aggregations, t);
}

std::unique_ptr<table> dispatch_groupby(table_view const& keys,
                                        host_span<aggregation_request const> requests,
                                        cudf::detail::result_cache* cache,
                                        bool const keys_have_nulls,
                                        null_policy const include_null_keys,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  auto const null_keys_are_equal  = null_equality::EQUAL;
  auto const has_null             = nullate::DYNAMIC{cudf::has_nested_nulls(keys)};
  auto const skip_rows_with_nulls = keys_have_nulls and include_null_keys == null_policy::EXCLUDE;

  auto preprocessed_keys = cudf::experimental::row::hash::preprocessed_table::create(keys, stream);
  auto const comparator  = cudf::experimental::row::equality::self_comparator{preprocessed_keys};
  auto const row_hash    = cudf::experimental::row::hash::row_hasher{std::move(preprocessed_keys)};
  auto const d_row_hash  = row_hash.device_hasher(has_null);

  if (cudf::detail::has_nested_columns(keys)) {
    auto const d_row_equal = comparator.equal_to<true>(has_null, null_keys_are_equal);
    return compute_groupby<nullable_row_comparator_t>(
      keys, requests, skip_rows_with_nulls, d_row_equal, d_row_hash, cache, stream, mr);
  } else {
    auto const d_row_equal = comparator.equal_to<false>(has_null, null_keys_are_equal);
    return compute_groupby<row_comparator_t>(
      keys, requests, skip_rows_with_nulls, d_row_equal, d_row_hash, cache, stream, mr);
  }
}
}  // namespace

/**
 * @brief Indicates if a set of aggregation requests can be satisfied with a
 * hash-based groupby implementation.
 *
 * @param requests The set of columns to aggregate and the aggregations to
 * perform
 * @return true A hash-based groupby should be used
 * @return false A hash-based groupby should not be used
 */
bool can_use_hash_groupby(host_span<aggregation_request const> requests)
{
  return std::all_of(requests.begin(), requests.end(), [](aggregation_request const& r) {
    auto const v_type = is_dictionary(r.values.type())
                          ? cudf::dictionary_column_view(r.values).keys().type()
                          : r.values.type();

    // Currently, input values (not keys) of STRUCT and LIST types are not supported in any of
    // hash-based aggregations. For those situations, we fallback to sort-based aggregations.
    if (v_type.id() == type_id::STRUCT or v_type.id() == type_id::LIST) { return false; }

    return std::all_of(r.aggregations.begin(), r.aggregations.end(), [v_type](auto const& a) {
      return cudf::has_atomic_support(cudf::detail::target_type(v_type, a->kind)) and
             is_hash_aggregation(a->kind);
    });
  });
}

// Hash-based groupby
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby(
  table_view const& keys,
  host_span<aggregation_request const> requests,
  null_policy include_null_keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudf::detail::result_cache cache(requests.size());

  std::unique_ptr<table> unique_keys =
    dispatch_groupby(keys, requests, &cache, cudf::has_nulls(keys), include_null_keys, stream, mr);

  return std::pair(std::move(unique_keys), extract_results(requests, cache, stream, mr));
}
}  // namespace cudf::groupby::detail::hash
