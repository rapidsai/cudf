/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <io/utilities/parsing_utils.cuh>

namespace cudf {
namespace io {
namespace json {
namespace gpu {
/**
 * @brief Convert a buffer of input data (text) into raw cuDF column data.
 *
 * @param[in] input_data The entire data to read
 * @param[in] dtypes The data type of each column
 * @param[out] output_columns The output column data
 * @param[in] num_records The number of lines/rows
 * @param[in] num_columns The number of columns
 * @param[in] rec_starts The start of each data record
 * @param[out] valid_fields The bitmaps indicating whether column fields are valid
 * @param[out] num_valid_fields The numbers of valid fields in columns
 * @param[in] opts A set of parsing options
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 *
 * @returns void
 **/
void convert_json_to_columns(rmm::device_buffer const &input_data,
                             data_type *const dtypes,
                             void *const *output_columns,
                             cudf::size_type num_records,
                             cudf::size_type num_columns,
                             const uint64_t *rec_starts,
                             bitmask_type *const *valid_fields,
                             cudf::size_type *num_valid_fields,
                             ParseOptions const &opts,
                             cudaStream_t stream = 0);

/**
 * @brief Process a buffer of data and determine information about the column types within.
 *
 * @param[out] column_infos The count for each column data type
 * @param[in] data Input data buffer
 * @param[in] data_size Size of the data buffer, in bytes
 * @param[in] opts A set of parsing options
 * @param[in] num_columns The number of columns of input data
 * @param[in] rec_starts The start the input data of interest
 * @param[in] num_records The number of lines/rows of input data
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 *
 * @returns void
 **/
void detect_data_types(ColumnInfo *column_infos,
                       const char *data,
                       size_t data_size,
                       const ParseOptions &options,
                       int num_columns,
                       const uint64_t *rec_starts,
                       cudf::size_type num_records,
                       cudaStream_t stream = 0);

}  // namespace gpu
}  // namespace json
}  // namespace io
}  // namespace cudf
