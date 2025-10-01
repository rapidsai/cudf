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

#pragma once

#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <type_traits>

namespace CUDF_EXPORT cudf {

/// @brief Flags for controlling initialization steps
enum class init_flags : std::uint32_t {
  /// @brief Load the nvCOMP library during initialization
  LOAD_NVCOMP = 1 << 0,
  /// @brief Initialize the JIT program cache during initialization
  INIT_JIT_CACHE = 1 << 1,
  /// @brief All initialization steps (default behavior)
  ALL = LOAD_NVCOMP | INIT_JIT_CACHE
};

/// @brief Bitwise OR operator for init_flags
/// @param lhs The left-hand side of the operator
/// @param rhs The right-hand side of the operator
/// @return The result of the bitwise OR operation
constexpr init_flags operator|(init_flags lhs, init_flags rhs) noexcept
{
  using underlying_t = std::underlying_type_t<init_flags>;
  return static_cast<init_flags>(static_cast<underlying_t>(lhs) | static_cast<underlying_t>(rhs));
}

/// @brief Check if a flag is set
/// @param flags The flags to check against
/// @param flag The specific flag to check for
/// @return true if all bits in `flag` are set in `flags`, false otherwise
constexpr bool has_flag(init_flags flags, init_flags flag) noexcept
{
  return (flags | flag) == flags;
}

/// @brief Initialize the cudf global context
/// @param flags Optional flags to control which initialization steps to perform.
/// @throws std::runtime_error if the context is already initialized
void initialize(init_flags flags = init_flags::INIT_JIT_CACHE);

/// @brief de-initialize the cudf global context
/// @throws std::runtime_error if the context is already de-initialized
void deinitialize();

}  // namespace CUDF_EXPORT cudf
