/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/types.hpp>

namespace cudf::groupby::detail::hash {

/**
 * @brief Context structure for hash compound aggregation finalization.
 *
 * Contains the state needed to finalize compound aggregations in hash groupby.
 */
struct hash_compound_agg_finalizer_context {
  column_view col;
  data_type input_type;
  cudf::detail::result_cache* cache;
  bitmask_type const* d_row_bitmask;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  hash_compound_agg_finalizer_context(column_view col,
                                      cudf::detail::result_cache* cache,
                                      bitmask_type const* d_row_bitmask,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

  // Enables conversion of ARGMIN/ARGMAX into MIN/MAX
  auto gather_argminmax(cudf::aggregation const& agg);
};

/**
 * @brief Functor to finalize compound aggregations in hash groupby.
 */
struct hash_compound_agg_finalizer_fn {
  hash_compound_agg_finalizer_context& ctx;

  explicit hash_compound_agg_finalizer_fn(hash_compound_agg_finalizer_context& ctx) : ctx(ctx) {}

  // Default case: no-op
  template <aggregation::Kind k>
  void operator()(aggregation const& agg) const {}
};

// Declare specializations
template <>
void hash_compound_agg_finalizer_fn::operator()<aggregation::MIN>(aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer_fn::operator()<aggregation::MAX>(aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer_fn::operator()<aggregation::MEAN>(aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer_fn::operator()<aggregation::M2>(aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer_fn::operator()<aggregation::VARIANCE>(
  aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer_fn::operator()<aggregation::STD>(aggregation const& agg) const;

}  // namespace cudf::groupby::detail::hash
