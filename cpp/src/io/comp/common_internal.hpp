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

enum class host_engine_state : uint8_t { ON, OFF, AUTO, HYBRID };

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
constexpr double default_host_device_decompression_work_ratio = 100;
// Estimated ratio between total CPU compression throughput and compression throughput of a
// single GPU block; higher values lead to more host compression in HYBRID mode
constexpr double default_host_device_compression_work_ratio = 100;

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
 * @param mr Memory resource to use for allocations of results
 * @return sorted_codec_parameters containing sorted inputs, outputs, and original ordering
 */
sorted_codec_parameters sort_tasks(device_span<device_span<uint8_t const> const> inputs,
                                   device_span<device_span<uint8_t> const> outputs,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

/**
 * @brief Finds the split index for input data based on the specified thresholds and target ratio.
 *
 * This function determines the index at which to split the input data for processing, using
 * the provided thresholds and target ratio. Inputs before the split index will be processed
 * using the host engine, while those after will be processed using the device engine.
 *
 * @param inputs A device span of input data to be split
 * @param host_state The state of the host-side engine used for processing
 * @param auto_mode_threshold The threshold value used to determine the split in automatic mode
 * @param hybrid_mode_target_ratio The target ratio used to determine the split in hybrid mode
 * @param stream The CUDA stream to be used for asynchronous execution
 *
 * @return The index at which the input data should be split.
 */
[[nodiscard]] size_t find_split_index(device_span<device_span<uint8_t const> const> inputs,
                                      host_engine_state host_state,
                                      size_t auto_mode_threshold,
                                      size_t hybrid_mode_target_ratio,
                                      rmm::cuda_stream_view stream);

/**
 * @brief Copies results back to their original positions using the ordering map
 *
 * This function restores the results to their original positions in the output
 * array using the ordering information created by sort_tasks().
 *
 * @param sorted_results Source results in sorted order
 * @param original_results Destination array where results will be placed in original order
 * @param order Mapping from sorted position to original position
 * @param stream CUDA stream for asynchronous execution
 */
void copy_results_to_original_order(device_span<codec_exec_result const> sorted_results,
                                    device_span<codec_exec_result> original_results,
                                    device_span<std::size_t const> order,
                                    rmm::cuda_stream_view stream);

}  // namespace cudf::io::detail
