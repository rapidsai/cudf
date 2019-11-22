/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

namespace cudf {
namespace io {
namespace csv {
namespace gpu {

/**
 * @brief Launches kernel for detecting possible dtype of each column of data
 *
 * @param[in] data The row-column data
 * @param[in] row_starts List of row data start positions (offsets)
 * @param[in] num_rows Number of rows
 * @param[in] num_columns Number of columns
 * @param[in] options Options that control individual field data conversion
 * @param[in,out] flags Flags that control individual column parsing
 * @param[out] stats Histogram of each dtypes' occurrence for each column
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t DetectColumnTypes(const char *data, const uint64_t *row_starts,
                              size_t num_rows, size_t num_columns,
                              const ParseOptions &options,
                              column_parse::flags *flags,
                              column_parse::stats *stats,
                              cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Launches kernel for decoding row-column data
 *
 * @param[in] data The row-column data
 * @param[in] row_starts List of row data start positions (offsets)
 * @param[in] num_rows Number of rows
 * @param[in] num_columns Number of columns
 * @param[in] options Options that control individual field data conversion
 * @param[in] flags Flags that control individual column parsing
 * @param[in] dtypes List of dtype corresponding to each column
 * @param[out] columns Device memory output of column data
 * @param[out] valids Device memory output of column valids bitmap data
 * @param[out] num_valid Number of valid fields in each column
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t DecodeRowColumnData(const char *data, const uint64_t *row_starts,
                                size_t num_rows, size_t num_columns,
                                const ParseOptions &options,
                                const column_parse::flags *flags,
                                cudf::data_type *dtypes, void **columns,
                                cudf::bitmask_type **valids,
                                cudaStream_t stream = (cudaStream_t)0);

}  // namespace gpu
}  // namespace csv
}  // namespace io
}  // namespace cudf
