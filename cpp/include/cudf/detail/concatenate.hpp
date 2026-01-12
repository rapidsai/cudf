/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <vector>

namespace CUDF_EXPORT cudf {
//! Inner interfaces and implementations
namespace detail {
/**
 * @copydoc cudf::concatenate(host_span<column_view const>,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> concatenate(host_span<column_view const> columns_to_concat,
                                    rmm::cuda_stream_view stream,
                                    cudf::memory_resources resources);

/**
 * @copydoc cudf::concatenate(host_span<table_view const>,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> concatenate(host_span<table_view const> tables_to_concat,
                                   rmm::cuda_stream_view stream,
                                   cudf::memory_resources resources);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
