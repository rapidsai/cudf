/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf::groupby::detail {

std::unique_ptr<column> compute_m2(data_type source_type,
                                   column_view const& sum_sqr,
                                   column_view const& sum,
                                   column_view const& count,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

std::unique_ptr<column> compute_variance(column_view const& m2,
                                         column_view const& count,
                                         size_type ddof,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

std::unique_ptr<column> compute_std(column_view const& m2,
                                    column_view const& count,
                                    size_type ddof,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

}  // namespace cudf::groupby::detail
