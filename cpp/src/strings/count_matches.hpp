/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {

class column_device_view;

namespace strings {

class regex_program;

namespace detail {

/**
 * @brief Returns a column of regex match counts for each string in the given column.
 *
 * A null entry will result in a zero count for that output row.
 *
 * @param d_strings Device view of the input strings column.
 * @param prog Regex program to evaluate on each string.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return Integer column of match counts
 */
std::unique_ptr<column> count_matches(column_device_view const& d_strings,
                                      regex_program const& prog,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace strings
}  // namespace cudf
