/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/filling.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {
/**
 * @copydoc cudf::sequence(size_type size, scalar const& init, scalar const& step,
 *                                       rmm::device_async_resource_ref mr =
 *cudf::get_current_device_resource_ref())
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> sequence(size_type size,
                                 scalar const& init,
                                 scalar const& step,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::sequence(size_type size, scalar const& init,
                                         rmm::device_async_resource_ref mr =
 cudf::get_current_device_resource_ref())
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> sequence(size_type size,
                                 scalar const& init,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::calendrical_month_sequence(size_type size,
 *                                           scalar const& init,
 *                                           size_type months,
 *                                           rmm::device_async_resource_ref mr)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<cudf::column> calendrical_month_sequence(size_type size,
                                                         scalar const& init,
                                                         size_type months,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
