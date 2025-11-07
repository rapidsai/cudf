/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_groupby.hpp"
#include "extract_single_pass_aggs.hpp"
#include "groupby/common/utils.hpp"
#include "helpers.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.cuh>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cudf::groupby::detail::hash {
namespace {
/**
 * @brief List of aggregation operations that can be computed with a hash-based implementation.
 *
 * For single pass aggregations, the supported operations are the ones that can be atomically
 * updated: SUM, SUM_WITH_OVERFLOW, SUM_OF_SQUARES, PRODUCT, MIN, MAX, COUNT_VALID, COUNT_ALL.
 * For compound aggregations, the supported operations are the ones that depends on the single pass
 * aggregations above: ARGMIN(MIN), ARGMAX(MAX), MEAN(SUM, COUNT_VALID), M2/STD/VARIANCE(M2,
 * COUNT_VALID).
 */
const auto hash_aggregations = std::unordered_set{// Single pass aggregations:
                                                  aggregation::SUM,
                                                  aggregation::SUM_WITH_OVERFLOW,
                                                  aggregation::SUM_OF_SQUARES,
                                                  aggregation::PRODUCT,
                                                  aggregation::MIN,
                                                  aggregation::MAX,
                                                  aggregation::COUNT_VALID,
                                                  aggregation::COUNT_ALL,
                                                  // Compound aggregations:
                                                  aggregation::ARGMIN,
                                                  aggregation::ARGMAX,
                                                  aggregation::MEAN,
                                                  aggregation::M2,
                                                  aggregation::STD,
                                                  aggregation::VARIANCE};

/**
 * @brief Indicates whether the specified aggregation operation can be computed
 * with a hash-based implementation.
 *
 * @param t The aggregation operation to verify
 * @return true `t` is valid for a hash based groupby
 * @return false `t` is invalid for a hash based groupby
 */
bool is_hash_aggregation(aggregation::Kind t) { return hash_aggregations.contains(t); }

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

  auto preprocessed_keys = cudf::detail::row::hash::preprocessed_table::create(keys, stream);
  auto const comparator  = cudf::detail::row::equality::self_comparator{preprocessed_keys};
  auto const row_hash    = cudf::detail::row::hash::row_hasher{std::move(preprocessed_keys)};
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

// check if the target_type of the aggregation/type pair supports atomic operations
struct can_use_hash_groupby_fn {
  template <typename T, aggregation::Kind K>
    requires(cudf::is_nested<T>())
  bool operator()() const
  {
    // Currently, input values (not keys) of STRUCT and LIST types are not supported in any of
    // hash-based aggregations. For those situations, we fallback to sort-based aggregations.
    return false;
  }

  template <aggregation::Kind k>
  constexpr static bool uses_underlying_type()
  {
    return k == aggregation::MIN or k == aggregation::MAX or k == aggregation::SUM;
  }

  template <typename T, aggregation::Kind K>
    requires(cudf::is_fixed_point<T>())
  bool operator()() const
  {
    using TargetType       = cudf::detail::target_type_t<T, K>;
    using DeviceTargetType = std::
      conditional_t<uses_underlying_type<K>(), cudf::device_storage_type_t<TargetType>, TargetType>;
    if constexpr (not std::is_void_v<DeviceTargetType>) {
      return cudf::has_atomic_support<DeviceTargetType>();
    }
    return false;
  }

  template <typename T, aggregation::Kind K>
    requires(not cudf::is_nested<T>() and not cudf::is_fixed_point<T>())
  bool operator()() const
  {
    using TargetType = cudf::detail::target_type_t<T, K>;
    if constexpr (not std::is_void_v<TargetType>) { return cudf::has_atomic_support<TargetType>(); }
    return false;
  }
};

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

    return std::all_of(r.aggregations.begin(), r.aggregations.end(), [v_type](auto const& a) {
      if (not is_hash_aggregation(a->kind)) { return false; }
      // compound aggregations are made up of simple aggregations
      auto const agg_kinds = get_simple_aggregations(*a, v_type);
      return std::all_of(agg_kinds.begin(), agg_kinds.end(), [v_type = v_type](auto k) {
        return cudf::detail::dispatch_type_and_aggregation(v_type, k, can_use_hash_groupby_fn{});
      });
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
