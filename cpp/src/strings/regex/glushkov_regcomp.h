/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

/**
 * @file glushkov_regcomp.h
 * @brief Host-side Glushkov NFA construction via ε-elimination from Thompson NFA.
 *
 * Glushkov's NFA (position automaton) has two key advantages over Thompson's NFA:
 *   1. No ε-transitions → no runtime ε-closure overhead per character.
 *   2. States are naturally represented as bit-positions → bit-parallel simulation
 *      on GPU (Hyperscan "shift-and" technique, USENIX ATC'19).
 *
 * The compiled program uses a 64-bit bitmask to represent the set of active
 * positions.  Each step costs only a few bitwise operations, compared to the
 * list-management overhead of the Thompson NFA.
 *
 * Limitations (patterns that fall back to Thompson NFA):
 *   - Zero-width assertions: ^  $  \\b  \\B
 *   - More than GLUSHKOV_MAX_STATES character-consuming positions
 *   - capture-group extraction (uses Thompson NFA always)
 */

#include "strings/regex/regcomp.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace cudf {
namespace strings {
namespace detail {

/// Maximum number of character-consuming positions (states) in the Glushkov NFA.
/// Patterns with more positions fall back to Thompson NFA automatically.
constexpr int32_t GLUSHKOV_MAX_STATES = 64;

/// Maximum shift amounts for the Hyperscan-style shift-and optimisation.
constexpr int32_t GLUSHKOV_MAX_SHIFTS = 8;

/// Size of the precomputed ASCII reach table (characters 0–127).
#ifndef GLUSHKOV_ASCII_TABLE_SIZE_DEFINED
#define GLUSHKOV_ASCII_TABLE_SIZE_DEFINED
constexpr int32_t GLUSHKOV_ASCII_TABLE_SIZE = 128;
#endif

/// Bitmask type: bit i is set when Glushkov position i is active.
using g_state_t = uint64_t;

/**
 * @brief Host-side compiled Glushkov NFA program.
 *
 * Produced by @ref build_glushkov_program from an already-compiled
 * Thompson NFA (@ref reprog).  All arrays are indexed by position ID
 * (0 .. num_states-1), which corresponds to the left-to-right order of
 * character-consuming instructions in the Thompson NFA.
 */
struct glushkov_host_program {
  uint32_t num_states{};    ///< Number of character-consuming positions
  g_state_t first_set{};    ///< Bitmask: positions reachable before first character
  g_state_t accept_mask{};  ///< Bitmask: positions whose match completes the pattern
  bool nullable{};          ///< True if the empty string satisfies the pattern

  /// follow_table[p] = bitmask of positions that can immediately follow position p.
  std::array<g_state_t, GLUSHKOV_MAX_STATES> follow_table{};

  // ---- Per-position character-matching descriptors ----
  // Using struct-of-arrays to avoid a separate shared struct that would need
  // to be visible in the CUDA device header.
  std::array<int32_t, GLUSHKOV_MAX_STATES> pos_inst_type{};  ///< CHAR/ANY/ANYNL/CCLASS/NCCLASS
  std::array<char32_t, GLUSHKOV_MAX_STATES> pos_ch{};        ///< Literal char (CHAR only)
  std::array<int32_t, GLUSHKOV_MAX_STATES> pos_cls_idx{};    ///< Class index (CCLASS/NCCLASS)

  // ---- Shift-and transition optimisation (Hyperscan-style) ----
  //
  // For each pair (p, q) where q ∈ follow_table[p] and 1 ≤ (q-p) ≤ 63:
  //   shift span s = q - p
  //   If s is one of the chosen shift_amounts[k]: set bit p in shift_masks[k]
  //
  // GPU execution:
  //   follow(state) ≈ OR_k( (state & shift_masks[k]) << shift_amounts[k] )
  //                   + exception transitions for states in exception_mask

  /// shift_masks[k]: source positions whose follow includes a span-k transition.
  std::array<g_state_t, GLUSHKOV_MAX_SHIFTS> shift_masks{};
  /// The corresponding shift amounts (ascending, 1-based).
  std::array<uint8_t, GLUSHKOV_MAX_SHIFTS> shift_amounts{};
  /// Number of valid entries in shift_masks / shift_amounts.
  uint32_t shift_count{};

  /// When true, every position in first_set is a CHAR instruction for the
  /// same literal character.  Enables a tight first-character skip in
  /// glushkov_find (mirrors Thompson NFA's find_char optimisation).
  bool has_startchar{};
  char32_t startchar{};

  /// Bitmask of positions with at least one "exception" (non-shift) transition.
  g_state_t exception_mask{};
  /// exception_succs[p] = union of all non-shift successor positions for position p.
  /// Non-zero only when bit p is set in exception_mask.
  std::array<g_state_t, GLUSHKOV_MAX_STATES> exception_succs{};

  /// Character class definitions (copied from the Thompson NFA reprog).
  std::vector<reclass> classes;

  /// Precomputed reach bitmask for each ASCII character (0–127).
  /// reach_ascii[c] = bitmask of positions that match ASCII character c.
  /// Built by build_glushkov_program after the follow table is complete.
  /// Non-ASCII characters are handled on-the-fly at match time.
  std::array<g_state_t, GLUSHKOV_ASCII_TABLE_SIZE> reach_ascii{};
};

/**
 * @brief Attempt to compile a Glushkov NFA from a fully compiled Thompson NFA.
 *
 * Returns nullptr when the pattern cannot be handled by the Glushkov path:
 *   - Pattern contains zero-width assertions (BOL, EOL, BOW, NBOW).
 *   - Pattern has more than GLUSHKOV_MAX_STATES character-consuming positions.
 *   - Pattern is nullable (matches the empty string): priority semantics cannot
 *     be faithfully represented without an ε-position for the empty match.
 *
 * @param prog  Compiled Thompson NFA (after reprog::finalize()).
 * @return      Host-side Glushkov program, or nullptr on failure.
 */
std::unique_ptr<glushkov_host_program> build_glushkov_program(reprog const& prog);

}  // namespace detail
}  // namespace strings
}  // namespace cudf
