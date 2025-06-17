/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.
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

#include <cudf/io/types.hpp>
#include <cudf/utilities/span.hpp>

namespace CUDF_EXPORT cudf {
namespace io::detail {

/**
 * @brief Without actually decompressing the compressed input buffer passed, return the size of
 * decompressed output. If the decompressed size cannot be extracted apriori, return zero.
 *
 * @param compression Type of compression of the input data
 * @param src         Compressed host buffer
 *
 * @return Size of decompressed output
 */
size_t get_uncompressed_size(compression_type compression, host_span<uint8_t const> src);

/**
 * @brief Struct to hold information about decompression.
 *
 * This struct contains details about the decompression process, including
 * the type of compression, the number of pages, the maximum size
 * of a decompressed page, and the total decompressed size.
 */
struct decompression_info {
  compression_type type;
  size_t num_pages;
  size_t max_page_decompressed_size;
  size_t total_decompressed_size;
};

/**
 * @brief Functor which returns total scratch space required based on computed decompression_info
 * data.
 *
 */
[[nodiscard]] size_t get_decompression_scratch_size(decompression_info const& di);

/**
 * @brief Computes the uncompressed sizes of Snappy-compressed input data.
 *
 * This function takes a collection of Snappy-compressed input data spans and computes
 * their respective uncompressed sizes. The results are stored in the provided output span.
 *
 * @param inputs Compressed device memory buffers
 * @param uncompressed_sizes Output device memory buffers to store the uncompressed sizes
 * @param stream CUDA stream to be used for device operations and synchronization.
 */
void get_snappy_uncompressed_size(device_span<device_span<uint8_t const> const> inputs,
                                  device_span<size_t> uncompressed_sizes,
                                  rmm::cuda_stream_view stream);

}  // namespace io::detail
}  // namespace CUDF_EXPORT cudf
