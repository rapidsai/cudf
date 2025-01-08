/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "comp.hpp"

#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <nvcomp.h>

#include <optional>

namespace cudf::io::detail::nvcomp {

struct batched_args {
  rmm::device_uvector<void const*> input_data_ptrs;
  rmm::device_uvector<size_t> input_data_sizes;
  rmm::device_uvector<void*> output_data_ptrs;
  rmm::device_uvector<size_t> output_data_sizes;
};

/**
 * @brief Split lists of src/dst device spans into lists of pointers/sizes.
 *
 * @param[in] inputs List of input buffers
 * @param[in] outputs List of output buffers
 * @param[in] stream CUDA stream to use
 */
batched_args create_batched_nvcomp_args(device_span<device_span<uint8_t const> const> inputs,
                                        device_span<device_span<uint8_t> const> outputs,
                                        rmm::cuda_stream_view stream);

/**
 * @brief Convert nvcomp statuses and output sizes into cuIO compression results.
 */
void update_compression_results(device_span<nvcompStatus_t const> nvcomp_stats,
                                device_span<size_t const> actual_output_sizes,
                                device_span<compression_result> results,
                                rmm::cuda_stream_view stream);

/**
 * @brief Fill the result array based on the actual output sizes.
 */
void update_compression_results(device_span<size_t const> actual_output_sizes,
                                device_span<compression_result> results,
                                rmm::cuda_stream_view stream);

/**
 * @brief Mark unsupported input chunks for skipping.
 */
void skip_unsupported_inputs(device_span<size_t> input_sizes,
                             device_span<compression_result> results,
                             std::optional<size_t> max_valid_input_size,
                             rmm::cuda_stream_view stream);

/**
 * @brief Returns the size of the largest input chunk and the total input size.
 */
std::pair<size_t, size_t> max_chunk_and_total_input_size(device_span<size_t const> input_sizes,
                                                         rmm::cuda_stream_view stream);

}  // namespace cudf::io::detail::nvcomp
