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

#include "cudf/utilities/error.hpp"

#include <cudf/utilities/export.hpp>

#include <nvml.h>

#include <functional>
#include <sstream>

#define CHECK_NVML(err_code) cudf::check_nvml(err_code, __FILE__, __LINE__)

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup utility_nvml_shim
 * @{
 * @file
 */

/**
 * @brief Singleton class to manage dynamic loading of the NVML library.
 *
 * NVML initialization is costly. Using this singleton ensures it is performed only once per
 * process. NVML shutdown is not performed in the destructor, but the API is exposed to be called on
 * users' discretion.
 */
class nvml_shim {
 public:
  nvml_shim(nvml_shim const&)            = delete;
  nvml_shim(nvml_shim&&)                 = delete;
  nvml_shim& operator=(nvml_shim const&) = delete;
  nvml_shim& operator=(nvml_shim&&)      = delete;

  /**
   * @brief Get the nvml_shim singleton instance.
   *
   * @return The nvml_shim singleton instance.
   *
   * @throws cudf::logic_error if the NVML shared library exists but the symbols fail to load.
   */
  static nvml_shim& instance();

  /**
   * @brief Check whether the NVML shared library exists (the file can be found and the symbols can
   * be loaded).
   *
   * This function uses dlopen to dynamically load the shared library and check if the loading is
   * successful or not. The order of search conducted by dlopen is documented at
   * https://man7.org/linux/man-pages/man3/dlopen.3.html . Most commonly, the library is located at
   * /usr/lib/<arch>/libnvidia-ml.so.1 .
   *
   * @return Boolean answer.
   */
  bool exists();

  /**
   * @brief Wrapper for nvmlInit_v2.
   */
  std::function<decltype(nvmlInit_v2)> init;

  /**
   * @brief Wrapper for nvmlShutdown.
   */
  std::function<decltype(nvmlShutdown)> shutdown;

  /**
   * @brief Wrapper for nvmlErrorString.
   */
  std::function<decltype(nvmlErrorString)> error_string;

  /**
   * @brief Wrapper for nvmlDeviceGetHandleByIndex_v2.
   */
  std::function<decltype(nvmlDeviceGetHandleByIndex_v2)> device_get_handle_by_index;

  /**
   * @brief Wrapper for nvmlDeviceGetFieldValues.
   */
  std::function<decltype(nvmlDeviceGetFieldValues)> device_get_field_values;

 private:
  nvml_shim();
  void* _lib_handle{};
};

/** @} */

/**
 * @brief Helper function to check the error code of the NVML API call.
 *
 * @param err_code The error code of the NVML API call.
 * @param file The source file name where the NVML API call fails.
 * @param line The line number where the NVML API call fails.
 *
 * @throws std::runtime_error is the NVML API call fails.
 */
inline void check_nvml(nvmlReturn_t err_code, const char* file, int line)
{
  if (err_code == nvmlReturn_enum::NVML_SUCCESS) { return; }
  std::stringstream ss;
  ss << "NVML error: " << err_code << " " << nvml_shim::instance().error_string(err_code) << " in "
     << file << " at line " << line << std::endl;
  throw std::runtime_error(ss.str());
}

}  // namespace CUDF_EXPORT cudf
