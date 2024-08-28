/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

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
 * @param indata Input device buffer
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to use for device memory allocation
 */
void normalize_single_quotes(datasource::owning_buffer<rmm::device_buffer>& indata,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr);

/**
 * @brief Normalize unquoted whitespace (space and tab characters) using FST
 *
 * @param indata Input device buffer
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to use for device memory allocation
 */
void normalize_whitespace(datasource::owning_buffer<rmm::device_buffer>& indata,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr);
}  // namespace io::json::detail
}  // namespace CUDF_EXPORT cudf
