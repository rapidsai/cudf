/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/utilities/export.hpp>

#include <optional>
#include <string>

namespace CUDF_EXPORT cudf {
namespace io::nvcomp {

enum class compression_type { SNAPPY, ZSTD, DEFLATE, LZ4 };

/**
 * @brief Set of parameters that impact whether nvCOMP features are enabled.
 *
 */
struct feature_status_parameters {
  int lib_major_version;                 ///< major version
  int lib_minor_version;                 ///< minor version
  int lib_patch_version;                 ///< patch version
  bool are_all_integrations_enabled;     ///< all integrations
  bool are_stable_integrations_enabled;  ///< stable integrations
  int compute_capability_major;          ///< cuda compute major version

  /**
   * @brief Default Constructor
   */
  feature_status_parameters();

  /**
   * @brief feature_status_parameters Constructor
   *
   * @param major positive integer representing major value of nvcomp
   * @param minor positive integer representing minor value of nvcomp
   * @param patch positive integer representing patch value of nvcomp
   * @param all_enabled if all integrations are enabled
   * @param stable_enabled if stable integrations are enabled
   * @param cc_major CUDA compute capability
   */
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
 * Result depends on nvCOMP version and environment variables.
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
 * Result depends on nvCOMP version and environment variables.
 *
 * @param compression Compression type
 * @param params Optional parameters to query status with different configurations
 * @returns Reason for the feature disablement, `std::nullopt` if the feature is enabled
 */
[[nodiscard]] std::optional<std::string> is_decompression_disabled(
  compression_type compression, feature_status_parameters params = feature_status_parameters());

}  // namespace io::nvcomp
}  // namespace CUDF_EXPORT cudf
