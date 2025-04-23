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
template <typename F>
void get_symbol(std::function<F>& fp, void* lib_handle, char const* symbol)
{
  dlerror();  // Clear any previous error
  fp           = reinterpret_cast<std::decay_t<F>>(dlsym(lib_handle, symbol));
  auto err_msg = dlerror();  // Check if any error arises
  CUDF_EXPECTS(err_msg == nullptr, "Failed to find symbol " + std::string(symbol));
}

template <typename Ret, typename... Args>
void let_shim_throw(std::function<Ret(Args...)>& fp)
{
  fp = [](Args...) -> Ret { CUDF_FAIL("NVML shim failed as the shared library does not exist."); };
}
}  // namespace

nvml_shim::nvml_shim()
{
  auto lib_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE);

  if (lib_handle == nullptr) {
    CUDF_LOG_INFO("NVIDIA Management Library (NVML) libnvidia-ml.so.1 cannot be opened; reason: %s",
                  dlerror());
    let_shim_throw(init);
    let_shim_throw(shutdown);
    let_shim_throw(error_string);
    let_shim_throw(device_get_handle_by_index);
    let_shim_throw(device_get_field_values);
    return;
  } else {
    _shared_library_exists = true;
  }

  get_symbol(init, lib_handle, "nvmlInit_v2");
  get_symbol(shutdown, lib_handle, "nvmlShutdown");
  get_symbol(error_string, lib_handle, "nvmlErrorString");
  get_symbol(device_get_handle_by_index, lib_handle, "nvmlDeviceGetHandleByIndex_v2");
  get_symbol(device_get_field_values, lib_handle, "nvmlDeviceGetFieldValues");

  CHECK_NVML(init());
}

nvml_shim& nvml_shim::instance()
{
  static nvml_shim instance;
  return instance;
}

bool nvml_shim::shared_library_exists() { return _shared_library_exists; }

}  // namespace cudf
