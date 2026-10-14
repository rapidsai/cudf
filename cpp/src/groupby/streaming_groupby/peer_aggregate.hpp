/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

namespace cudf::groupby {

/// Combines supported columns by warp peers and preserves scalar aggregation for other runs.
void compute_peer_aggs(table_view const& values,
                       host_span<aggregation::Kind const> kinds,
                       aggregation::Kind const* device_kinds,
                       device_span<size_type const> targets,
                       mutable_table_view const& output,
                       mutable_table_device_view device_output,
                       cuda::stream_ref stream,
                       cudf::memory_resources mr);

}  // namespace cudf::groupby
