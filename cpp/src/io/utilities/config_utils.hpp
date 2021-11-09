/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cstdlib>
#include <string>

namespace cudf::io::detail {

/**
 * @brief Returns the value of the environment variable, or a default value if the variable is not
 * present.
 */
inline std::string getenv_or(std::string const& env_var_name, std::string_view default_val)
{
  auto const env_val = std::getenv(env_var_name.c_str());
  return std::string{(env_val == nullptr) ? default_val : env_val};
}

namespace nvcomp_integration {

namespace {
/**
 * @brief Defines which nvCOMP usage to enable.
 */
enum class usage_policy : uint8_t { OFF, STABLE, ALWAYS };

/**
 * @brief Get the current usage policy.
 */
inline usage_policy get_env_policy()
{
  static auto const env_val = getenv_or("LIBCUDF_NVCOMP_POLICY", "STABLE");
  if (env_val == "OFF") return usage_policy::OFF;
  if (env_val == "ALWAYS") return usage_policy::ALWAYS;
  return usage_policy::STABLE;
}
}  // namespace

/**
 * @brief Returns true if all nvCOMP uses are enabled.
 */
inline bool is_all_enabled() { return get_env_policy() == usage_policy::ALWAYS; }

/**
 * @brief Returns true if stable nvCOMP use is enabled.
 */
inline bool is_stable_enabled()
{
  return is_all_enabled() or get_env_policy() == usage_policy::STABLE;
}

}  // namespace nvcomp_integration
}  // namespace cudf::io::detail
