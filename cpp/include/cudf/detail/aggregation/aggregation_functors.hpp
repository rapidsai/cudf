/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace cudf {
namespace detail {

/**
 * @brief Functor to collect simple aggregations for compound aggregations.
 *
 * This functor has a templated operator() that is specialized for aggregation types
 * that require decomposition into simpler aggregations. The default implementation
 * returns the aggregation itself.
 */
struct simple_aggregation_collector_fn {
  /**
   * @brief Default implementation: returns the aggregation itself (for simple aggregations).
   *
   * @tparam k The aggregation kind
   * @param col_type The type of the column being aggregated
   * @param agg The aggregation to decompose
   * @return Vector containing a clone of the aggregation
   */
  template <aggregation::Kind k>
  std::vector<std::unique_ptr<aggregation>> operator()(data_type col_type,
                                                       aggregation const& agg) const
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(agg.clone());
    return aggs;
  }
};

/**
 * @brief Functor to finalize compound aggregations.
 *
 * This functor has a templated operator() that is specialized for aggregation types
 * that require post-processing after simple aggregations have been computed.
 * The default implementation does nothing.
 */
struct aggregation_finalizer_fn {
  /**
   * @brief Default implementation: no-op (for simple aggregations).
   *
   * @tparam k The aggregation kind
   * @param agg The aggregation to finalize
   */
  template <aggregation::Kind k>
  void operator()(aggregation const& agg) const {}
};

}  // namespace detail
}  // namespace cudf
