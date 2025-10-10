/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "getenv_or.hpp"

#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/utilities/error.hpp>

#include <kvikio/defaults.hpp>

#include <string>

namespace cudf::io {

namespace kvikio_integration {

void set_up_kvikio()
{
  static std::once_flag flag{};
  std::call_once(flag, [] {
    // Workaround for https://github.com/rapidsai/cudf/issues/14140, where cuFileDriverOpen errors
    // out if no CUDA calls have been made before it. This is a no-op if the CUDA context is already
    // initialized.
    cudaFree(nullptr);

    auto const compat_mode = kvikio::getenv_or("KVIKIO_COMPAT_MODE", kvikio::CompatMode::ON);
    kvikio::defaults::set_compat_mode(compat_mode);

    auto const nthreads = getenv_or<unsigned int>("KVIKIO_NTHREADS", 4u);
    kvikio::defaults::set_thread_pool_nthreads(nthreads);
  });
}

}  // namespace kvikio_integration

namespace nvcomp_integration {

namespace {
/**
 * @brief Defines which nvCOMP usage to enable.
 */
enum class usage_policy : uint8_t { OFF, STABLE, ALWAYS };

/**
 * @brief Get the current usage policy.
 */
[[nodiscard]] usage_policy get_env_policy()
{
  auto const env_val = getenv_or<std::string>("LIBCUDF_NVCOMP_POLICY", "STABLE");
  if (env_val == "OFF") return usage_policy::OFF;
  if (env_val == "STABLE") return usage_policy::STABLE;
  if (env_val == "ALWAYS") return usage_policy::ALWAYS;
  CUDF_FAIL("Invalid LIBCUDF_NVCOMP_POLICY value: " + env_val);
}
}  // namespace

[[nodiscard]] bool is_all_enabled() { return get_env_policy() == usage_policy::ALWAYS; }

[[nodiscard]] bool is_stable_enabled() { return is_all_enabled() or get_env_policy() == usage_policy::STABLE; }

}  // namespace nvcomp_integration

namespace integrated_memory_optimization {

namespace {
/**
 * @brief Defines which integrated memory optimization policy to use.
 */
enum class optimization_policy : uint8_t { OFF, AUTO, ON };

/**
 * @brief Get the current optimization policy.
 */
 [[nodiscard]]  optimization_policy get_env_policy()
{
  auto const env_val =
    getenv_or<std::string>("LIBCUDF_INTEGRATED_MEMORY_OPTIMIZATION", "AUTO");
  if (env_val == "OFF") return optimization_policy::OFF;
  if (env_val == "AUTO") return optimization_policy::AUTO;
  if (env_val == "ON") return optimization_policy::ON;
  CUDF_FAIL("Invalid LIBCUDF_INTEGRATED_MEMORY_OPTIMIZATION value: " + env_val);
}
}  // namespace

/**
 * @brief Check if integrated memory optimizations are enabled.
 *
 * Controlled by the LIBCUDF_INTEGRATED_MEMORY_OPTIMIZATION environment variable.
 * Valid values: "AUTO" (default), "ON", "OFF"
 * - AUTO: Use hardware detection (cudaDevAttrIntegrated)
 * - ON: Always enable optimization
 * - OFF: Always disable optimization
 */
[[nodiscard]] bool is_enabled()
{
  auto const policy = get_env_policy();
  switch (policy) {
    case optimization_policy::OFF: return false;
    case optimization_policy::ON: return true;
    case optimization_policy::AUTO: return cudf::detail::has_integrated_memory();
  }
  CUDF_FAIL("Invalid LIBCUDF_INTEGRATED_MEMORY_OPTIMIZATION value");
}

}  // namespace integrated_memory_optimization
}  // namespace cudf::io
