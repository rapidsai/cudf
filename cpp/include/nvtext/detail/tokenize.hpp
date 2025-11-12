/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT nvtext {
namespace detail {
/**
 * @copydoc nvtext::tokenize(strings_column_view const&,string_scalar
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<cudf::column> tokenize(cudf::strings_column_view const& strings,
                                       cudf::string_scalar const& delimiter,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc nvtext::tokenize(strings_column_view const&,strings_column_view
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<cudf::column> tokenize(cudf::strings_column_view const& strings,
                                       cudf::strings_column_view const& delimiters,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc nvtext::count_tokens(strings_column_view const&, string_scalar
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<cudf::column> count_tokens(cudf::strings_column_view const& strings,
                                           cudf::string_scalar const& delimiter,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr);

/**
 * @copydoc nvtext::count_tokens(strings_column_view const&,strings_column_view
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<cudf::column> count_tokens(cudf::strings_column_view const& strings,
                                           cudf::strings_column_view const& delimiters,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace CUDF_EXPORT nvtext
