/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/replace.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace detail {
/**
 * @copydoc cudf::replace_nulls(column_view const&, column_view const&,
 * rmm::device_async_resource_ref)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> replace_nulls(column_view const& input,
                                      cudf::column_view const& replacement,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::replace_nulls(column_view const&, scalar const&,
 * rmm::device_async_resource_ref)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> replace_nulls(column_view const& input,
                                      scalar const& replacement,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::replace_nulls(column_view const&, replace_policy const&,
 * rmm::device_async_resource_ref)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> replace_nulls(column_view const& input,
                                      replace_policy const& replace_policy,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::replace_nans(column_view const&, column_view const&,
 * rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> replace_nans(column_view const& input,
                                     column_view const& replacement,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::replace_nans(column_view const&, scalar const&,
 * rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> replace_nans(column_view const& input,
                                     scalar const& replacement,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::find_and_replace_all
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> find_and_replace_all(column_view const& input_col,
                                             column_view const& values_to_replace,
                                             column_view const& replacement_values,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::normalize_nans_and_zeros
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> normalize_nans_and_zeros(column_view const& input,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
