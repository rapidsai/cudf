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

#include <cudf/utilities/nvml_shim.hpp>
#include <cudf/utilities/system.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda.h>

#include <nvml.h>

namespace cudf {

bool is_nvml_available() { return cudf::nvml_shim::instance().exists(); }

bool is_c2c_available()
{
  // todo: remove this once CUDA 11 support is dropped
#if CUDA_VERSION < 12000
  return false;
#else
  nvmlDevice_t device_handle{};

  CHECK_NVML(nvml_shim::instance().device_get_handle_by_index(
    rmm::get_current_cuda_device().value(), &device_handle));

  nvmlFieldValue_t field{};
  field.fieldId = NVML_FI_DEV_C2C_LINK_COUNT;
  CHECK_NVML(nvml_shim::instance().device_get_field_values(device_handle, 1, &field));

  return (field.nvmlReturn == nvmlReturn_t::NVML_SUCCESS) && (field.value.uiVal > 0);
#endif
}
}  // namespace cudf
