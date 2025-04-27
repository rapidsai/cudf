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

#include "nvml_shim.hpp"

#include "cudf/logger_macros.hpp"

#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>

#include <dlfcn.h>

#include <string>

namespace cudf {

namespace {
template <typename Ret, typename... Args>
void initialize_shim_function(std::function<Ret(Args...)>& fp, void* lib_handle, char const* symbol)
{
  if (lib_handle == nullptr) {
    fp = [](Args...) -> Ret {
      CUDF_FAIL("NVML shim failed as the shared library does not exist.");
    };
  } else {
    dlerror();  // Clear any previous error
    fp           = reinterpret_cast<std::decay_t<Ret(Args...)>>(dlsym(lib_handle, symbol));
    auto err_msg = dlerror();  // Check if any error arises
    CUDF_EXPECTS(err_msg == nullptr, "Failed to find symbol " + std::string(symbol));
  }
}
}  // namespace

nvml_shim::nvml_shim()
{
  auto lib_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE);

  if (lib_handle == nullptr) {
    CUDF_LOG_INFO("NVIDIA Management Library (NVML) libnvidia-ml.so.1 cannot be opened; reason: %s",
                  dlerror());
  }

  initialize_shim_function(init, lib_handle, "nvmlInit_v2");
  initialize_shim_function(shutdown, lib_handle, "nvmlShutdown");
  initialize_shim_function(error_string, lib_handle, "nvmlErrorString");
  initialize_shim_function(device_get_handle_by_index, lib_handle, "nvmlDeviceGetHandleByIndex_v2");
  initialize_shim_function(device_get_field_values, lib_handle, "nvmlDeviceGetFieldValues");

  if (lib_handle == nullptr) { return; }

  _shared_library_exists = true;
  CHECK_NVML(init());
}

nvml_shim& nvml_shim::instance()
{
  static nvml_shim instance;
  return instance;
}

bool nvml_shim::shared_library_exists() { return _shared_library_exists; }

}  // namespace cudf
