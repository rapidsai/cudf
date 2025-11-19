/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace io::json::detail {

/**
 * @brief Reads and returns the entire data set.
 *
 * @param sources Input `datasource` objects to read the dataset from
 * @param options Settings for controlling reading behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to use for device memory allocation
 *
 * @return cudf::table object that contains the array of cudf::column.
 */
table_with_metadata read_json(host_span<std::unique_ptr<datasource>> sources,
                              json_reader_options const& options,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

/**
 * @brief Write an entire dataset to JSON format.
 *
 * @param sink Output sink
 * @param table The set of columns
 * @param options Settings for controlling behavior
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
void write_json(data_sink* sink,
                table_view const& table,
                json_writer_options const& options,
                rmm::cuda_stream_view stream);

/**
 * @brief Normalize single quotes to double quotes using FST
 *
 * @param indata    Input device buffer
 * @param delimiter Line-separating delimiter character in JSONL inputs
 * @param stream    CUDA stream used for device memory operations and kernel launches
 * @param mr        Device memory resource to use for device memory allocation
 */
void normalize_single_quotes(datasource::owning_buffer<rmm::device_buffer>& indata,
                             char delimiter,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr);

/**
 * @brief Normalize unquoted whitespace (space and tab characters) using FST
 *
 * @param indata Input device buffer
 * @param col_offsets Offsets to column contents in input buffer
 * @param col_lengths Length of contents of each row in column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to use for device memory allocation
 *
 * @returns Tuple of the normalized column, offsets to each row in column, and lengths of contents
 * of each row
 */
std::
  tuple<rmm::device_uvector<char>, rmm::device_uvector<size_type>, rmm::device_uvector<size_type>>
  normalize_whitespace(device_span<char const> d_input,
                       device_span<size_type const> col_offsets,
                       device_span<size_type const> col_lengths,
                       rmm::cuda_stream_view stream,
                       rmm::device_async_resource_ref mr);

}  // namespace io::json::detail
}  // namespace CUDF_EXPORT cudf
