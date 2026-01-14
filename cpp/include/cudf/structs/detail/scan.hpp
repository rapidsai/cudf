/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace structs::detail {
/**
 * @brief Scan function for struct column type
 *
 * Called by cudf::scan() with only min and max aggregates.
 *
 * @tparam Op Either DeviceMin or DeviceMax operations
 *
 * @param input Input column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New struct column
 */
template <typename Op>
std::unique_ptr<column> scan_inclusive(column_view const& input,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

}  // namespace structs::detail
}  // namespace CUDF_EXPORT cudf
