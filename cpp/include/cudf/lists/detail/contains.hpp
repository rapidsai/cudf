/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/lists/contains.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT cudf {
namespace lists::detail {

/**
 * @copydoc cudf::lists::index_of(cudf::lists_column_view const&,
 *                                cudf::scalar const&,
 *                                duplicate_find_option,
 *                                rmm::device_async_resource_ref)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> index_of(cudf::lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 cudf::lists::duplicate_find_option find_option,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::lists::index_of(cudf::lists_column_view const&,
 *                                cudf::column_view const&,
 *                                duplicate_find_option,
 *                                rmm::device_async_resource_ref)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> index_of(cudf::lists_column_view const& lists,
                                 cudf::column_view const& search_keys,
                                 cudf::lists::duplicate_find_option find_option,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::lists::contains(cudf::lists_column_view const&,
 *                                cudf::scalar const&,
 *                                rmm::device_async_resource_ref)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::lists::contains(cudf::lists_column_view const&,
 *                                cudf::column_view const&,
 *                                rmm::device_async_resource_ref)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::column_view const& search_keys,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);
}  // namespace lists::detail
}  // namespace CUDF_EXPORT cudf
