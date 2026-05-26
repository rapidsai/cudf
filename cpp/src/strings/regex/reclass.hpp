/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "regcomp.h"

#include <cuda_runtime.h>

namespace cudf::strings::detail {

/// Bitmask type: bit i is set when Glushkov position i is active.
using g_state_t = uint64_t;

/// Maximum number of character-consuming positions (states) in the Glushkov NFA.
/// Patterns with more positions fall back to Thompson NFA automatically.
constexpr int32_t GLUSHKOV_MAX_STATES = sizeof(g_state_t) * 8;

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

}  // namespace cudf::strings::detail
