/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <type_traits>

namespace CUDF_EXPORT cudf {

/// @brief Flags for controlling initialization steps
enum class init_flags : std::uint32_t {
  /// @brief No initialization steps
  NONE = 0,
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

/// @brief Bitwise AND operator for init_flags
/// @param lhs The left-hand side of the operator
/// @param rhs The right-hand side of the operator
/// @return The result of the bitwise AND operation
constexpr init_flags operator&(init_flags lhs, init_flags rhs) noexcept
{
  using underlying_t = std::underlying_type_t<init_flags>;
  return static_cast<init_flags>(static_cast<underlying_t>(lhs) & static_cast<underlying_t>(rhs));
}

/// @brief Bitwise NOT operator for init_flags
/// @param flags The flags to negate
/// @return The result of the bitwise NOT operation, only flipping bits that are part of
/// init_flags::ALL
constexpr init_flags operator~(init_flags flags) noexcept
{
  using underlying_t = std::underlying_type_t<init_flags>;
  return static_cast<init_flags>(static_cast<underlying_t>(init_flags::ALL) &
                                 ~static_cast<underlying_t>(flags));
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
/// Can be called multiple times to initialize additional components. If all selected
/// steps are already performed, the call has no effect.
void initialize(init_flags flags = init_flags::INIT_JIT_CACHE);

/// @brief Destroy the cudf global context, resetting it to an uninitialized state. This is
/// primarily intended for testing purposes, allowing for re-initialization of the context after
/// teardown.
/// @warning This is not intended for general use and may lead to undefined behavior if used
/// improperly. The caller must ensure that no threads are concurrently accessing the context during
/// teardown and that only one thread calls teardown at a time.
void teardown();

}  // namespace CUDF_EXPORT cudf
