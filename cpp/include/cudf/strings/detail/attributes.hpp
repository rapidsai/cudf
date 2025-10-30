/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @copydoc cudf::strings::count_characters
 */
std::unique_ptr<column> count_characters(strings_column_view const& input,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace strings
}  // namespace cudf
