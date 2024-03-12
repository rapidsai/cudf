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

#include "config_utils.hpp"

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdlib>
#include <string>

namespace cudf::io {

namespace detail {

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

inline std::mutex& host_mr_lock()
{
  static std::mutex map_lock;
  return map_lock;
}

inline rmm::host_async_resource_ref default_pinned_mr()
{
  static rmm::mr::pinned_host_memory_resource default_mr{};
  return default_mr;
}

CUDF_EXPORT inline auto& host_mr()
{
  static rmm::host_async_resource_ref host_mr = default_pinned_mr();
  return host_mr;
}

}  // namespace detail

rmm::host_async_resource_ref set_host_memory_resource(rmm::host_async_resource_ref mr)
{
  std::lock_guard lock{detail::host_mr_lock()};
  auto last_mr      = detail::host_mr();
  detail::host_mr() = mr;
  return last_mr;
}

rmm::host_async_resource_ref get_host_memory_resource()
{
  std::lock_guard lock{detail::host_mr_lock()};
  return detail::host_mr();
}

}  // namespace cudf::io
