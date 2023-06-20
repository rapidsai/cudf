/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

/**
 * @file parquet_column_statistics.cu
 * @brief Template specialization for PARQUET statistics calls
 */

#include "column_statistics.cuh"

namespace cudf {
namespace io {
namespace detail {

template <>
void merge_group_statistics<detail::io_file_format::PARQUET>(statistics_chunk* chunks_out,
                                                             statistics_chunk const* chunks_in,
                                                             statistics_merge_group const* groups,
                                                             uint32_t num_chunks,
                                                             rmm::cuda_stream_view stream);
template <>
void calculate_group_statistics<detail::io_file_format::PARQUET>(statistics_chunk* chunks,
                                                                 statistics_group const* groups,
                                                                 uint32_t num_chunks,
                                                                 rmm::cuda_stream_view stream,
                                                                 bool int96_timestamp);

}  // namespace detail
}  // namespace io
}  // namespace cudf
