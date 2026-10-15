/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <optional>

namespace cudf::groupby::detail::hash {

/**
 * @brief Functor to finalize compound aggregations in hash groupby.
 */
struct hash_compound_agg_finalizer {
  column_view const col;
  data_type const input_type;
  bool const input_has_nulls;
  cudf::detail::result_cache* const cache;
  cuda::stream_ref const stream;
  cudf::memory_resources const mr;

  /**
   * @param input_has_nulls Whether the aggregated input had nulls; defaults to
   * `col.has_nulls()`. Callers that pass intermediate results as `col` must set it.
   */
  hash_compound_agg_finalizer(column_view const& col,
                              cudf::detail::result_cache* cache,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr,
                              std::optional<bool> input_has_nulls = std::nullopt);

  // Default case: no-op
  template <aggregation::Kind k>
  void operator()(aggregation const& agg) const
  {
  }
};

// Declare specializations
template <>
void hash_compound_agg_finalizer::operator()<aggregation::MIN>(aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer::operator()<aggregation::MAX>(aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer::operator()<aggregation::MEAN>(aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer::operator()<aggregation::M2>(aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer::operator()<aggregation::VARIANCE>(aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer::operator()<aggregation::STD>(aggregation const& agg) const;

}  // namespace cudf::groupby::detail::hash
