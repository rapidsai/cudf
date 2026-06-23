/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "regcomp.h"

#include <cudf/types.hpp>

#include <cuda/std/optional>
#include <cuda/std/utility>
#include <cuda_runtime.h>

namespace cudf::strings::detail {

/// Bitmask type: bit i is set when Glushkov position i is active.
using glushkov_state_t = uint64_t;

/// Maximum number of character-consuming positions (states) in the Glushkov NFA.
/// Patterns with more positions fall back to Thompson NFA automatically.
constexpr int32_t GLUSHKOV_MAX_STATES = sizeof(glushkov_state_t) * 8;

/// Maximum shift amounts for the Hyperscan-style shift-and optimization.
constexpr int32_t GLUSHKOV_MAX_SHIFTS = 8;

/// Size of the precomputed ASCII reach table (characters 0–127).
constexpr int32_t GLUSHKOV_ASCII_TABLE_SIZE = 128;

/**
 * @brief Regex class stored on the device and executed by reprog_device.
 *
 * This class holds the unique data for any regex CCLASS instruction.
 */
struct alignas(16) reclass_device {
  int32_t builtins{};
  int32_t count{};
  reclass_range const* literals{};

  __device__ inline bool is_match(char32_t const ch, uint8_t const* flags) const;
};

/**
 * @brief Check for supported new-line characters
 *
 * '\n, \r, \u0085, \u2028, or \u2029'
 */
__host__ __device__ __forceinline__ constexpr bool is_newline(char32_t const ch)
{
  return (ch == '\n' || ch == '\r' || ch == 0x00c285 || ch == 0x00e280a8 || ch == 0x00e280a9);
}

/**
 * @brief Template type used on `find` to specify desired position values in returned match_result
 */
enum class positional : int8_t {
  BEGIN_END = 0,  /// both begin and end positions are returned
  END_ONLY  = 1,  /// only the end position is returned
};

using match_pair   = cuda::std::pair<cudf::size_type, cudf::size_type>;
using match_result = cuda::std::optional<match_pair>;

}  // namespace cudf::strings::detail
