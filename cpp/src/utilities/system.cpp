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

#include <cudf/utilities/system.hpp>

#include <rmm/cuda_device.hpp>

#include <nvml.h>

#include <iostream>
#include <mutex>
#include <sstream>

#define CHECK_NVML(err_code) check_nvml(err_code, __FILE__, __LINE__)
inline void check_nvml(nvmlReturn_t err_code, const char* file, int line)
{
  if (err_code != nvmlReturn_enum::NVML_SUCCESS) {
    std::stringstream ss;
    ss << "NVML error: " << err_code << " " << nvmlErrorString(err_code) << " in " << file
       << " at line " << line << std::endl;
    throw std::runtime_error(ss.str());
  }
}

namespace cudf {
namespace detail {
/**
 * @brief Perform the initialization of NVML library.
 *
 * NVML initialization can be costly. Using this function ensures it is performed only once per
 * process.
 *
 * @note NVML shutdown is also costly. We may intentionally not perform shutdown at the expense of a
 * slight NVML resource leak.
 */
void initialize_nvml()
{
  static std::once_flag flag{};
  std::call_once(flag, [] { CHECK_NVML(nvmlInit_v2()); });
}
}  // namespace detail

bool is_c2c_available()
{
  nvmlDevice_t device_handle{};

  detail::initialize_nvml();
  CHECK_NVML(nvmlDeviceGetHandleByIndex_v2(rmm::get_current_cuda_device().value(), &device_handle));

  nvmlFieldValue_t field{};
  field.fieldId = NVML_FI_DEV_C2C_LINK_COUNT;
  CHECK_NVML(nvmlDeviceGetFieldValues(device_handle, 1, &field));

  return (field.nvmlReturn == nvmlReturn_t::NVML_SUCCESS) && (field.value.uiVal > 0);
}
}  // namespace cudf
