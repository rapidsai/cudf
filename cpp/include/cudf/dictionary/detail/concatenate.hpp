/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace dictionary::detail {
/**
 * @brief Returns a single column by vertically concatenating the given vector of
 * dictionary columns.
 *
 * @throw cudf::logic_error if `columns.size()==0`
 * @throw cudf::logic_error if dictionary column keys are not all the same type.
 *
 * @param columns Vector of dictionary columns to concatenate.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column with concatenated results.
 */
std::unique_ptr<column> concatenate(host_span<column_view const> columns,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

}  // namespace dictionary::detail
}  // namespace CUDF_EXPORT cudf
