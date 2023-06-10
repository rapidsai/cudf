/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "gpuinflate.hpp"

#include <io/utilities/config_utils.hpp>

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <optional>

namespace cudf::io::nvcomp {

enum class compression_type { SNAPPY, ZSTD, DEFLATE };

/**
 * @brief Set of parameters that impact whether the use nvCOMP features is enabled.
 */
struct feature_status_parameters {
  int lib_major_version;
  int lib_minor_version;
  int lib_patch_version;
  bool are_all_integrations_enabled;
  bool are_stable_integrations_enabled;
  int compute_capability_major;

  feature_status_parameters();
  feature_status_parameters(
    int major, int minor, int patch, bool all_enabled, bool stable_enabled, int cc_major)
    : lib_major_version{major},
      lib_minor_version{minor},
      lib_patch_version{patch},
      are_all_integrations_enabled{all_enabled},
      are_stable_integrations_enabled{stable_enabled},
      compute_capability_major{cc_major}
  {
  }
};

/**
 * @brief Equality operator overload. Required to use `feature_status_parameters` as a map key.
 */
inline bool operator==(feature_status_parameters const& lhs, feature_status_parameters const& rhs)
{
  return lhs.lib_major_version == rhs.lib_major_version and
         lhs.lib_minor_version == rhs.lib_minor_version and
         lhs.lib_patch_version == rhs.lib_patch_version and
         lhs.are_all_integrations_enabled == rhs.are_all_integrations_enabled and
         lhs.are_stable_integrations_enabled == rhs.are_stable_integrations_enabled and
         lhs.compute_capability_major == rhs.compute_capability_major;
}

/**
 * @brief If a compression type is disabled through nvCOMP, returns the reason as a string.
 *
 * Result cab depend on nvCOMP version and environment variables.
 *
 * @param compression Compression type
 * @param params Optional parameters to query status with different configurations
 * @returns Reason for the feature disablement, `std::nullopt` if the feature is enabled
 */
[[nodiscard]] std::optional<std::string> is_compression_disabled(
  compression_type compression, feature_status_parameters params = feature_status_parameters());

/**
 * @brief If a decompression type is disabled through nvCOMP, returns the reason as a string.
 *
 * Result can depend on nvCOMP version and environment variables.
 *
 * @param compression Compression type
 * @param params Optional parameters to query status with different configurations
 * @returns Reason for the feature disablement, `std::nullopt` if the feature is enabled
 */
[[nodiscard]] std::optional<std::string> is_decompression_disabled(
  compression_type compression, feature_status_parameters params = feature_status_parameters());

/**
 * @brief Device batch decompression of given type.
 *
 * @param[in] compression Compression type
 * @param[in] inputs List of input buffers
 * @param[out] outputs List of output buffers
 * @param[out] results List of output status structures
 * @param[in] max_uncomp_chunk_size maximum size of uncompressed chunk
 * @param[in] max_total_uncomp_size maximum total size of uncompressed data
 * @param[in] stream CUDA stream to use
 */
void batched_decompress(compression_type compression,
                        device_span<device_span<uint8_t const> const> inputs,
                        device_span<device_span<uint8_t> const> outputs,
                        device_span<compression_result> results,
                        size_t max_uncomp_chunk_size,
                        size_t max_total_uncomp_size,
                        rmm::cuda_stream_view stream);

/**
 * @brief Gets the maximum size any chunk could compress to in the batch.
 *
 * @param compression Compression type
 * @param max_uncomp_chunk_size Size of the largest uncompressed chunk in the batch
 */
[[nodiscard]] size_t compress_max_output_chunk_size(compression_type compression,
                                                    uint32_t max_uncomp_chunk_size);

/**
 * @brief Gets input alignment requirements for the given compression type.
 *
 * @param compression Compression type
 * @returns required alignment, in bits
 */
[[nodiscard]] size_t compress_input_alignment_bits(compression_type compression);

/**
 * @brief Gets output alignment requirements for the given compression type.
 *
 * @param compression Compression type
 * @returns required alignment, in bits
 */
[[nodiscard]] size_t compress_output_alignment_bits(compression_type compression);

/**
 * @brief Maximum size of uncompressed chunks that can be compressed with nvCOMP.
 *
 * @param compression Compression type
 * @returns maximum chunk size
 */
[[nodiscard]] std::optional<size_t> compress_max_allowed_chunk_size(compression_type compression);

/**
 * @brief Device batch compression of given type.
 *
 * @param[in] compression Compression type
 * @param[in] inputs List of input buffers
 * @param[out] outputs List of output buffers
 * @param[out] results List of output status structures
 * @param[in] stream CUDA stream to use
 */
void batched_compress(compression_type compression,
                      device_span<device_span<uint8_t const> const> inputs,
                      device_span<device_span<uint8_t> const> outputs,
                      device_span<compression_result> results,
                      rmm::cuda_stream_view stream);

}  // namespace cudf::io::nvcomp
