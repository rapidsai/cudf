/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cudf/logger.hpp>

#include <cstdlib>
#include <sstream>
#include <string>

namespace {
/**
 * @brief Returns the value of the environment variable, or a default value if the variable is not
 * present.
 */
template <typename T>
T getenv_or(std::string_view env_var_name, T default_val)
{
  auto const env_val = std::getenv(env_var_name.data());
  if (env_val != nullptr) {
    CUDF_LOG_INFO("Environment variable %.*s read as %s",
                  static_cast<int>(env_var_name.length()),
                  env_var_name.data(),
                  env_val);
  } else {
    std::stringstream ss;
    ss << default_val;
    CUDF_LOG_INFO("Environment variable %.*s is not set, using default value %s",
                  static_cast<int>(env_var_name.length()),
                  env_var_name.data(),
                  ss.str());
  }

  if (env_val == nullptr) { return std::move(default_val); }

  std::stringstream sstream(env_val);
  T converted_val;
  sstream >> converted_val;
  return converted_val;
}

}  // namespace
