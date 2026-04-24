/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

/**
 * @file glushkov.cuh
 * @brief Device-side Glushkov NFA program data structure.
 *
 * This header is included INSIDE the cudf::strings::detail namespace block
 * in regex.cuh, so it assumes that reclass_device is already defined.
 *
 * The device program is a compact representation of the Glushkov NFA:
 *  - State is a 64-bit bitmask (one bit per position, no working memory needed).
 *  - Transition = shift-and + exception lookup (Hyperscan technique).
 *  - Character matching reuses the existing reclass_device::is_match path.
 */

#include <cstddef>
#include <cstdint>

// ---- constants (must match glushkov_regcomp.h) ----------------------------
static constexpr int32_t GLUSHKOV_MAX_STATES_DEV = 64;
static constexpr int32_t GLUSHKOV_MAX_SHIFTS_DEV = 8;
#ifndef GLUSHKOV_ASCII_TABLE_SIZE_DEFINED
#define GLUSHKOV_ASCII_TABLE_SIZE_DEFINED
static constexpr int32_t GLUSHKOV_ASCII_TABLE_SIZE = 128;
#endif
using g_state_t = uint64_t;

// ---------------------------------------------------------------------------
// Per-position character-matching descriptor (device-side)
// ---------------------------------------------------------------------------

/// Stores enough data to check whether a GPU thread's current character
/// matches a given Glushkov position.
struct glushkov_position {
  int32_t inst_type;  ///< CHAR / ANY / ANYNL / CCLASS / NCCLASS
  char32_t ch{};      ///< Literal character (CHAR only)
  int32_t cls_idx{};  ///< Class index into _classes array (CCLASS/NCCLASS only)
};

// ---------------------------------------------------------------------------
// Device program
// ---------------------------------------------------------------------------

/**
 * @brief GPU-resident Glushkov NFA program.
 *
 * All array pointers address device-global memory packed into a single flat
 * buffer (see regexec.cpp :: create_glushkov_device).  The struct itself is
 * also stored at the head of that buffer so that the reprog_device pointer
 * _glushkov points directly into device memory.
 *
 * Unlike reprog_device, this struct does NOT use a global working-memory
 * buffer: the NFA state (g_state_t) fits in a single 64-bit register per
 * thread.
 */
struct glushkov_program_device {
  // ---- scalar fields -------------------------------------------------------
  uint32_t num_states{};       ///< Number of Glushkov positions (≤ 64)
  uint32_t shift_count{};      ///< Number of shift slots (≤ GLUSHKOV_MAX_SHIFTS_DEV)
  g_state_t first_set{};       ///< Initial active positions (before first character)
  g_state_t accept_mask{};     ///< Positions whose match completes the pattern
  g_state_t exception_mask{};  ///< Positions with non-shift follow transitions
  bool nullable{};  ///< Always false when reached (nullable patterns fall back to Thompson)
  /// When true every first_set position is CHAR(startchar): enables a tight
  /// first-character byte-scan skip (mirrors Thompson NFA's find_char).
  bool has_startchar{};
  char32_t startchar{};  ///< The common literal (valid only when has_startchar)

  // ---- device-memory array pointers (set during device program creation) ---
  uint8_t const* _codepoint_flags{};      ///< Character-type lookup table (shared with Thompson)
  reclass_device const* _classes{};       ///< Character class data
  glushkov_position const* _positions{};  ///< [num_states] per-position descriptors
  g_state_t const* _shift_masks{};        ///< [shift_count] shift-and source masks
  uint8_t const* _shift_amounts{};        ///< [shift_count] shift amounts
  g_state_t const*
    _reach_ascii{};  ///< [GLUSHKOV_ASCII_TABLE_SIZE] precomputed reach bitmasks for ASCII chars
  g_state_t const* _exception_succs{};  ///< [num_states] exception successor masks

  std::size_t _prog_size{};  ///< Total buffer size (for potential shmem loading)
};

// ---------------------------------------------------------------------------
// Shared-memory cache for Glushkov program arrays
// ---------------------------------------------------------------------------

/**
 * @brief Block-shared cache of read-only Glushkov program data.
 *
 * Loaded cooperatively by all threads in a block at kernel entry via
 * @ref glushkov_load_shmem.  Subsequent calls to glushkov_find (with cache)
 * read from shared memory (~20-cycle latency) instead of global/L2 (~100 cycles).
 *
 * Total size: ~1608 bytes — negligible impact on occupancy.
 */
struct glushkov_shmem_cache {
  g_state_t reach_ascii[GLUSHKOV_ASCII_TABLE_SIZE];    ///< precomputed ASCII reach masks
  g_state_t shift_masks[GLUSHKOV_MAX_SHIFTS_DEV];      ///< [shift_count] shift-and source masks
  uint8_t shift_amounts[GLUSHKOV_MAX_SHIFTS_DEV];      ///< [shift_count] shift amounts
  g_state_t exception_succs[GLUSHKOV_MAX_STATES_DEV];  ///< [num_states] exception successor masks
};
