/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {
/**
 * @copydoc cudf::drop_nulls(table_view const&, std::vector<size_type> const&,
 *                           cudf::size_type, rmm::cuda_stream_view, rmm::device_async_resource_ref)
 */
std::unique_ptr<table> drop_nulls(table_view const& input,
                                  std::vector<size_type> const& keys,
                                  cudf::size_type keep_threshold,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::drop_nans(table_view const&, std::vector<size_type> const&,
 *                          cudf::size_type, rmm::cuda_stream_view, rmm::device_async_resource_ref)
 */
std::unique_ptr<table> drop_nans(table_view const& input,
                                 std::vector<size_type> const& keys,
                                 cudf::size_type keep_threshold,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::apply_boolean_mask
 */
std::unique_ptr<table> apply_boolean_mask(table_view const& input,
                                          column_view const& boolean_mask,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::unique
 */
std::unique_ptr<table> unique(table_view const& input,
                              std::vector<size_type> const& keys,
                              duplicate_keep_option keep,
                              null_equality nulls_equal,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::distinct
 */
std::unique_ptr<table> distinct(table_view const& input,
                                std::vector<size_type> const& keys,
                                duplicate_keep_option keep,
                                null_equality nulls_equal,
                                nan_equality nans_equal,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::stable_distinct
 */
std::unique_ptr<table> stable_distinct(table_view const& input,
                                       std::vector<size_type> const& keys,
                                       duplicate_keep_option keep,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::distinct_indices
 *
 * @return A device_uvector containing the result indices
 */
rmm::device_uvector<size_type> distinct_indices(table_view const& input,
                                                duplicate_keep_option keep,
                                                null_equality nulls_equal,
                                                nan_equality nans_equal,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::unique_count(column_view const&, null_policy, nan_policy, rmm::cuda_stream_view)
 */
cudf::size_type unique_count(column_view const& input,
                             null_policy null_handling,
                             nan_policy nan_handling,
                             rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::unique_count(table_view const&, null_equality, rmm::cuda_stream_view)
 */
cudf::size_type unique_count(table_view const& input,
                             null_equality nulls_equal,
                             rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::distinct_count(column_view const&, null_policy, nan_policy, rmm::cuda_stream_view)
 */
cudf::size_type distinct_count(column_view const& input,
                               null_policy null_handling,
                               nan_policy nan_handling,
                               rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::distinct_count(table_view const&, null_equality, rmm::cuda_stream_view)
 */
cudf::size_type distinct_count(table_view const& input,
                               null_equality nulls_equal,
                               rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
