/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/sorting.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @copydoc cudf::sorted_order
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> sorted_order(table_view const& input,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::stable_sorted_order
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> stable_sorted_order(table_view const& input,
                                            std::vector<order> const& column_order,
                                            std::vector<null_order> const& null_precedence,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::sort_by_key
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> sort_by_key(table_view const& values,
                                   table_view const& keys,
                                   std::vector<order> const& column_order,
                                   std::vector<null_order> const& null_precedence,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::rank
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> rank(column_view const& input,
                             rank_method method,
                             order column_order,
                             null_policy null_handling,
                             null_order null_precedence,
                             bool percentage,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::stable_sort_by_key
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> stable_sort_by_key(table_view const& values,
                                          table_view const& keys,
                                          std::vector<order> const& column_order,
                                          std::vector<null_order> const& null_precedence,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::segmented_sorted_order
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> segmented_sorted_order(table_view const& keys,
                                               column_view const& segment_offsets,
                                               std::vector<order> const& column_order,
                                               std::vector<null_order> const& null_precedence,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::stable_segmented_sorted_order
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> stable_segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::segmented_sort_by_key
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> segmented_sort_by_key(table_view const& values,
                                             table_view const& keys,
                                             column_view const& segment_offsets,
                                             std::vector<order> const& column_order,
                                             std::vector<null_order> const& null_precedence,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::stable_segmented_sort_by_key
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> stable_segmented_sort_by_key(table_view const& values,
                                                    table_view const& keys,
                                                    column_view const& segment_offsets,
                                                    std::vector<order> const& column_order,
                                                    std::vector<null_order> const& null_precedence,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::sort
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> sort(table_view const& values,
                            std::vector<order> const& column_order,
                            std::vector<null_order> const& null_precedence,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::stable_sort
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> stable_sort(table_view const& values,
                                   std::vector<order> const& column_order,
                                   std::vector<null_order> const& null_precedence,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
