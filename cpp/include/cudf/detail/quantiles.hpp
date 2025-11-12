/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/quantiles.hpp>
#include <cudf/tdigest/tdigest_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @copydoc cudf::quantile()
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> quantile(column_view const& input,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 column_view const& ordered_indices,
                                 bool exact,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::quantiles()
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> quantiles(table_view const& input,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 cudf::sorted is_input_sorted,
                                 std::vector<order> const& column_order,
                                 std::vector<null_order> const& null_precedence,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::percentile_approx(tdigest_column_view const&, column_view const&,
 * rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> percentile_approx(tdigest::tdigest_column_view const& input,
                                          column_view const& percentiles,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
