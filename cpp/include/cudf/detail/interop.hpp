/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/interop.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <numbers>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @copydoc cudf::from_dlpack
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> from_dlpack(DLManagedTensor const* managed_tensor,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::to_dlpack
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
DLManagedTensor* to_dlpack(table_view const& input,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr);

/**
 * @brief Return a maximum precision for a given type.
 *
 * @tparam T the type to get the maximum precision for
 */
template <typename T>
constexpr std::size_t max_precision()
{
  auto constexpr num_bits = sizeof(T) * 8;
  return std::floor(num_bits * std::numbers::ln2 / std::numbers::ln10);
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
