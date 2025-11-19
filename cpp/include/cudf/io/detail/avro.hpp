/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/avro.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace io::detail::avro {

/**
 * @brief Reads the entire dataset.
 *
 * @param source Input `datasource` object to read the dataset from
 * @param options Settings for controlling reading behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to use for device memory allocation
 *
 * @return The set of columns along with table metadata
 */
table_with_metadata read_avro(std::unique_ptr<cudf::io::datasource>&& source,
                              avro_reader_options const& options,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

}  // namespace io::detail::avro
}  // namespace CUDF_EXPORT cudf
