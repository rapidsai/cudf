/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/binaryop.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
//! Inner interfaces and implementations
namespace detail {

/**
 * @copydoc cudf::binary_operation(column_view const&, column_view const&,
 * std::string const&, data_type, rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         std::string const& ptx,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::binary_operation(scalar const&, column_view const&, binary_operator,
 * data_type, rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> binary_operation(scalar const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::binary_operation(column_view const&, scalar const&, binary_operator,
 * data_type, rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         scalar const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::binary_operation(column_view const&, column_view const&,
 * binary_operator, data_type, rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
