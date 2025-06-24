/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "nvcomp_adapter.hpp"

#include <cudf/io/types.hpp>

#include <optional>

namespace cudf::io::detail {

/**
 * @brief GZIP header flags
 * See https://tools.ietf.org/html/rfc1952
 */
namespace GZIPHeaderFlag {
constexpr uint8_t ftext    = 0x01;  // ASCII text hint
constexpr uint8_t fhcrc    = 0x02;  // Header CRC present
constexpr uint8_t fextra   = 0x04;  // Extra fields present
constexpr uint8_t fname    = 0x08;  // Original file name present
constexpr uint8_t fcomment = 0x10;  // Comment present
};  // namespace GZIPHeaderFlag

// Threshold for buffer count where device compression is favored over host in AUTO mode
constexpr size_t default_host_compression_auto_threshold = 128;
// Threshold for buffer count where device decompression is favored over host in AUTO mode
constexpr size_t default_host_decompression_auto_threshold = 128;
// Estimated ratio between total CPU decompression throughput and decompression throughput of a
// single GPU block; higher values lead to more host decompression in HYBRID mode
constexpr double default_host_decompression_ratio = 100;
// Estimated ratio between total CPU compression throughput and compression throughput of a
// single GPU block; higher values lead to more host compression in HYBRID mode
constexpr double default_host_compression_ratio = 100;

[[nodiscard]] std::optional<nvcomp::compression_type> to_nvcomp_compression(
  compression_type compression);

struct sorted_codec_parameters {
  rmm::device_uvector<device_span<uint8_t const>> inputs;
  rmm::device_uvector<device_span<uint8_t>> outputs;
  rmm::device_uvector<std::size_t> order;  // mapping from sorted position to original position
};

/**
 * @brief Sorts input and output spans by input size in descending order
 *
 * This function creates a sorted view of the inputs and outputs where they are
 * ordered by the size of each input span in descending order (largest first).
 * This can reduce latency by processing larger chunks first.
 *
 * @param inputs Device spans of input data to be sorted
 * @param outputs Device spans of output buffers corresponding to inputs
 * @param stream CUDA stream for asynchronous execution
 * @return sorted_codec_parameters containing sorted inputs, outputs, and original ordering
 */
sorted_codec_parameters sort_tasks(device_span<device_span<uint8_t const> const> inputs,
                                   device_span<device_span<uint8_t> const> outputs,
                                   rmm::cuda_stream_view stream);

/**
 * @brief Copies results back to their original positions using the ordering map
 *
 * This function restores the results to their original positions in the output
 * array using the ordering information created by sort_tasks().
 *
 * @param src Source results in sorted order
 * @param dst Destination array where results will be placed in original order
 * @param order Mapping from sorted position to original position
 * @param stream CUDA stream for asynchronous execution
 */
void copy_results_to_original_order(device_span<codec_exec_result const> sorted_results,
                                    device_span<codec_exec_result> original_results,
                                    device_span<std::size_t const> order,
                                    rmm::cuda_stream_view stream);

}  // namespace cudf::io::detail
