/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cudf/utilities/error.hpp>

#include <kvikio/defaults.hpp>

#include <string>

namespace cudf::io {

namespace cufile_integration {

namespace {
/**
 * @brief Defines which cuFile usage to enable.
 */
enum class usage_policy : uint8_t { OFF, GDS, ALWAYS, KVIKIO };

/**
 * @brief Get the current usage policy.
 */
usage_policy get_env_policy()
{
  static auto const env_val = getenv_or<std::string>("LIBCUDF_CUFILE_POLICY", "KVIKIO");
  if (env_val == "OFF") return usage_policy::OFF;
  if (env_val == "GDS") return usage_policy::GDS;
  if (env_val == "ALWAYS") return usage_policy::ALWAYS;
  if (env_val == "KVIKIO") return usage_policy::KVIKIO;
  CUDF_FAIL("Invalid LIBCUDF_CUFILE_POLICY value: " + env_val);
}
}  // namespace

bool is_always_enabled() { return get_env_policy() == usage_policy::ALWAYS; }

bool is_gds_enabled() { return is_always_enabled() or get_env_policy() == usage_policy::GDS; }

bool is_kvikio_enabled() { return get_env_policy() == usage_policy::KVIKIO; }

void set_up_kvikio()
{
  static std::once_flag flag{};
  std::call_once(flag, [] {
    auto const compat_mode =
      kvikio::detail::getenv_or("KVIKIO_COMPAT_MODE", kvikio::CompatMode::ON);
    kvikio::defaults::compat_mode_reset(compat_mode);

    auto const nthreads = getenv_or<unsigned int>("KVIKIO_NTHREADS", 4u);
    kvikio::defaults::thread_pool_nthreads_reset(nthreads);
  });
}
}  // namespace cufile_integration

namespace nvcomp_integration {

namespace {
/**
 * @brief Defines which nvCOMP usage to enable.
 */
enum class usage_policy : uint8_t { OFF, STABLE, ALWAYS };

/**
 * @brief Get the current usage policy.
 */
usage_policy get_env_policy()
{
  static auto const env_val = getenv_or<std::string>("LIBCUDF_NVCOMP_POLICY", "STABLE");
  if (env_val == "OFF") return usage_policy::OFF;
  if (env_val == "STABLE") return usage_policy::STABLE;
  if (env_val == "ALWAYS") return usage_policy::ALWAYS;
  CUDF_FAIL("Invalid LIBCUDF_NVCOMP_POLICY value: " + env_val);
}
}  // namespace

bool is_all_enabled() { return get_env_policy() == usage_policy::ALWAYS; }

bool is_stable_enabled() { return is_all_enabled() or get_env_policy() == usage_policy::STABLE; }

}  // namespace nvcomp_integration
}  // namespace cudf::io
