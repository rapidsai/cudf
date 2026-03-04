/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace io {
namespace detail {
namespace csv {

std::unique_ptr<column> pandas_format_durations(column_view const& durations,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr);

}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace cudf
