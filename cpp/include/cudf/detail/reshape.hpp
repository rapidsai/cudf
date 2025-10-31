/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace detail {
/**
 * @copydoc cudf::tile
 */
std::unique_ptr<table> tile(table_view const& input,
                            size_type count,
                            rmm::cuda_stream_view,
                            rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::interleave_columns
 */
std::unique_ptr<column> interleave_columns(table_view const& input,
                                           rmm::cuda_stream_view,
                                           rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::table_to_array
 */
void table_to_array(table_view const& input,
                    device_span<cuda::std::byte> output,
                    rmm::cuda_stream_view stream = cudf::get_default_stream());

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
