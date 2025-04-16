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
 * @brief Perform the initialization of NVML library.
 *
 * NVML initialization can be costly. Using this function ensures it is performed only once per
 * process.
 *
 * @note NVML shutdown is also costly. We may intentionally not perform shutdown at the expense of a
 * slight NVML resource leak.
 */
class nvml_shim {
 public:
  nvml_shim(nvml_shim const&)            = delete;
  nvml_shim(nvml_shim&&)                 = delete;
  nvml_shim& operator=(nvml_shim const&) = delete;
  nvml_shim& operator=(nvml_shim&&)      = delete;

  static nvml_shim& instance();

  bool exists();

  std::function<decltype(nvmlInit_v2)> init;
  std::function<decltype(nvmlShutdown)> shutdown;
  std::function<decltype(nvmlErrorString)> error_string;
  std::function<decltype(nvmlDeviceGetHandleByIndex_v2)> device_get_handle_by_index;
  std::function<decltype(nvmlDeviceGetFieldValues)> device_get_field_values;

 private:
  nvml_shim();
  void* _lib_handle{};
};

inline void check_nvml(nvmlReturn_t err_code, const char* file, int line)
{
  if (err_code == nvmlReturn_enum::NVML_SUCCESS) { return; }
  std::stringstream ss;
  ss << "NVML error: " << err_code << " " << nvml_shim::instance().error_string(err_code) << " in "
     << file << " at line " << line << std::endl;
  throw std::runtime_error(ss.str());
}

}  // namespace CUDF_EXPORT cudf
