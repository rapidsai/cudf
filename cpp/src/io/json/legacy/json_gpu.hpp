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

#include "hash/concurrent_unordered_map.cuh"
#include "io/utilities/column_type_histogram.hpp"
#include "io/utilities/parsing_utils.cuh"

#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/optional>

using cudf::device_span;

namespace cudf::io::json::detail::legacy {

using col_map_type = concurrent_unordered_map<uint32_t, cudf::size_type>;
/**
 * @brief Convert a buffer of input data (text) into raw cuDF column data.
 *
 * @param[in] options A set of parsing options
 * @param[in] data The entire data to read
 * @param[in] row_offsets The start of each data record
 * @param[in] dtypes The data type of each column
 * @param[in] col_map Pointer to the (column name hash -> column index) map in device memory.
 * nullptr is passed when the input file does not consist of objects.
 * @param[out] output_columns The output column data
 * @param[out] valid_fields The bitmaps indicating whether column fields are valid
 * @param[out] num_valid_fields The numbers of valid fields in columns
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
void convert_json_to_columns(parse_options_view const& options,
                             device_span<char const> data,
                             device_span<uint64_t const> row_offsets,
                             device_span<data_type const> column_types,
                             col_map_type* col_map,
                             device_span<void* const> output_columns,
                             device_span<bitmask_type* const> valid_fields,
                             device_span<cudf::size_type> num_valid_fields,
                             rmm::cuda_stream_view stream);

/**
 * @brief Process a buffer of data and determine information about the column types within.
 *
 * @param[in] options A set of parsing options
 * @param[in] data Input data buffer
 * @param[in] row_offsets The offset of each row in the input
 * @param[in] num_columns The number of columns of input data
 * @param[in] col_map Pointer to the (column name hash -> column index) map in device memory.
 * nullptr is passed when the input file does not consist of objects.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 *
 * @returns The count for each column data type
 */
std::vector<cudf::io::column_type_histogram> detect_data_types(
  parse_options_view const& options,
  device_span<char const> data,
  device_span<uint64_t const> row_offsets,
  bool do_set_null_count,
  int num_columns,
  col_map_type* col_map,
  rmm::cuda_stream_view stream);

/**
 * @brief Collects information about JSON object keys in the file.
 *
 * @param[in] options A set of parsing options
 * @param[in] data Input data buffer
 * @param[in] row_offsets The offset of each row in the input
 * @param[out] keys_cnt Number of keys found in the file
 * @param[out] keys_info optional, information (offset, length, hash) for each found key
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
void collect_keys_info(parse_options_view const& options,
                       device_span<char const> data,
                       device_span<uint64_t const> row_offsets,
                       unsigned long long int* keys_cnt,
                       cuda::std::optional<mutable_table_device_view> keys_info,
                       rmm::cuda_stream_view stream);

}  // namespace cudf::io::json::detail::legacy
