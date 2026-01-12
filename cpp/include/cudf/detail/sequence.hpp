/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
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
 *                                       cudf::memory_resources resources =
 *cudf::get_current_device_resource_ref())
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> sequence(size_type size,
                                 scalar const& init,
                                 scalar const& step,
                                 rmm::cuda_stream_view stream,
                                 cudf::memory_resources resources);

/**
 * @copydoc cudf::sequence(size_type size, scalar const& init,
                                         cudf::memory_resources resources =
 cudf::get_current_device_resource_ref())
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> sequence(size_type size,
                                 scalar const& init,
                                 rmm::cuda_stream_view stream,
                                 cudf::memory_resources resources);

/**
 * @copydoc cudf::calendrical_month_sequence(size_type size,
 *                                           scalar const& init,
 *                                           size_type months,
 *                                           cudf::memory_resources resources)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<cudf::column> calendrical_month_sequence(size_type size,
                                                         scalar const& init,
                                                         size_type months,
                                                         rmm::cuda_stream_view stream,
                                                         cudf::memory_resources resources);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
