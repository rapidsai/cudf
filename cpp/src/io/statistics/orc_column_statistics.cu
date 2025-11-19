/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file orc_column_statistics.cu
 * @brief Template specialization for ORC statistics calls
 */

#include "column_statistics.cuh"

namespace cudf {
namespace io {
namespace detail {

template <>
void merge_group_statistics<detail::io_file_format::ORC>(statistics_chunk* chunks_out,
                                                         statistics_chunk const* chunks_in,
                                                         statistics_merge_group const* groups,
                                                         uint32_t num_chunks,
                                                         rmm::cuda_stream_view stream);
template <>
void calculate_group_statistics<detail::io_file_format::ORC>(statistics_chunk* chunks,
                                                             statistics_group const* groups,
                                                             uint32_t num_chunks,
                                                             rmm::cuda_stream_view stream,
                                                             bool int96_timestamp);

}  // namespace detail
}  // namespace io
}  // namespace cudf
