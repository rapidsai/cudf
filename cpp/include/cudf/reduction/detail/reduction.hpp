/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <optional>

namespace CUDF_EXPORT cudf {
namespace reduction::detail {

/**
 * @copydoc cudf::reduce(column_view const&, reduce_aggregation const&, data_type,
 * std::optional<std::reference_wrapper<scalar const>>, rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<scalar> reduce(column_view const& col,
                               reduce_aggregation const& agg,
                               data_type output_dtype,
                               std::optional<std::reference_wrapper<scalar const>> init,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr);

}  // namespace reduction::detail
}  // namespace CUDF_EXPORT cudf
