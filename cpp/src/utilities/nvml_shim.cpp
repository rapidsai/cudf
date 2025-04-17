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

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/nvml_shim.hpp>

#include <dlfcn.h>

#include <mutex>
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
}  // namespace

nvml_shim::nvml_shim()
{
  if (!exists()) { return; }

  get_symbol(init, _lib_handle, "nvmlInit_v2");
  get_symbol(shutdown, _lib_handle, "nvmlShutdown");
  get_symbol(error_string, _lib_handle, "nvmlErrorString");
  get_symbol(device_get_handle_by_index, _lib_handle, "nvmlDeviceGetHandleByIndex_v2");
  get_symbol(device_get_field_values, _lib_handle, "nvmlDeviceGetFieldValues");

  CHECK_NVML(init());
}

nvml_shim& nvml_shim::instance()
{
  static nvml_shim instance;
  return instance;
}

bool nvml_shim::exists()
{
  static std::once_flag flag{};
  std::call_once(flag, [&] {
    _lib_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE);
  });

  return _lib_handle != nullptr;
}

}  // namespace cudf
