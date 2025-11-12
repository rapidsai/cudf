/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/groupby.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <utility>

namespace CUDF_EXPORT cudf {
namespace groupby::detail::hash {
/**
 * @brief Indicates if a set of aggregation requests can be satisfied with a
 * hash-based groupby implementation.
 *
 * @param requests The set of columns to aggregate and the aggregations to
 * perform
 * @return true A hash-based groupby can be used
 * @return false A hash-based groupby cannot be used
 */
bool can_use_hash_groupby(host_span<aggregation_request const> requests);

// Hash-based groupby
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby(
  table_view const& keys,
  host_span<aggregation_request const> requests,
  null_policy include_null_keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
}  // namespace groupby::detail::hash
}  // namespace CUDF_EXPORT cudf
