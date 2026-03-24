/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/lists/contains.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace lists::detail {

/**
 * @copydoc cudf::lists::index_of(cudf::lists_column_view const&,
 *                                cudf::scalar const&,
 *                                duplicate_find_option,
 *                                rmm::cuda_stream_view stream,
 *                                rmm::device_async_resource_ref)
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
 *                                rmm::cuda_stream_view stream,
 *                                rmm::device_async_resource_ref)
 */
std::unique_ptr<column> index_of(cudf::lists_column_view const& lists,
                                 cudf::column_view const& search_keys,
                                 cudf::lists::duplicate_find_option find_option,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::lists::contains(cudf::lists_column_view const&,
 *                                cudf::scalar const&,
 *                                rmm::cuda_stream_view stream,
 *                                rmm::device_async_resource_ref)
 */
std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::lists::contains(cudf::lists_column_view const&,
 *                                cudf::column_view const&,
 *                                rmm::cuda_stream_view stream,
 *                                rmm::device_async_resource_ref)
 */
std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::column_view const& search_keys,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);
}  // namespace lists::detail
}  // namespace cudf
