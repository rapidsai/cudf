/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file glushkov.inl
 * @brief GPU device functions for Glushkov NFA execution.
 *
 * Included by regex.cuh after both reclass_device and glushkov_program_device
 * are fully defined.
 *
 * Matching algorithm
 * ------------------
 * One thread processes one input string.  The NFA state is a single g_state_t
 * (uint64_t) bitmask held in a register – no global working memory required.
 *
 * For each starting position `start` in the string (unanchored search):
 *   1. Initialise: state = first_set                          (virtual start)
 *   2. For each character c at position pos ≥ start:
 *        follow_state = compute_follow(state)                  (shift + exceptions)
 *        state        = (pos == start ? first_set : follow_state) & reach(c)
 *      Equivalently:
 *        state = first_set & reach(c_start)       (first char)
 *        state = compute_follow(state) & reach(c)  (subsequent chars)
 *   3. If state & accept_mask: record greedy match end.
 *   4. Return leftmost non-empty (or nullable zero-length) match.
 *
 * The shift-and follow computation (Hyperscan-style):
 *   follow(state) ≈ OR_k( (state & shift_masks[k]) << shift_amounts[k] )
 *                   | OR over active exception states (exception_succs lookup)
 */

namespace cudf {
namespace strings {
namespace detail {

// ---------------------------------------------------------------------------
// Character matching
// ---------------------------------------------------------------------------

/**
 * @brief Returns true if character @p c is matched by Glushkov position @p pos.
 *
 * Mirrors the character-dispatch logic in the Thompson NFA regexec loop.
 */
__device__ __forceinline__ bool glushkov_position_matches(
  glushkov_position const& pos,
  char32_t const c,
  reclass_device const* classes,
  uint8_t const* codepoint_flags)
{
  switch (pos.inst_type) {
    case CHAR: return c == pos.ch;
    case ANY: {
      // pos.ch == 'N' → EXT_NEWLINE: reject all is_newline() characters
      // otherwise     → default: reject only '\n' (matches Thompson NFA)
      return (pos.ch == 'N') ? !is_newline(c) : (c != '\n');
    }
    case ANYNL: return true;
    case CCLASS: return classes[pos.cls_idx].is_match(c, codepoint_flags);
    case NCCLASS: return !classes[pos.cls_idx].is_match(c, codepoint_flags);
    default: return false;
  }
}

// ---------------------------------------------------------------------------
// Follow computation (shift-and + exceptions)
// ---------------------------------------------------------------------------

/**
 * @brief Compute the Glushkov follow set for the current state bitmask.
 *
 * Two-phase:
 *   1. Shift masks  – fast, covers most forward transitions.
 *   2. Exception table – backward and large-span transitions.
 */
__device__ __forceinline__ g_state_t glushkov_compute_follow(g_state_t const state,
                                                             glushkov_program_device const& prog)
{
  g_state_t follow = 0;

  // Phase 1: shift-and transitions
  for (uint32_t k = 0; k < prog.shift_count; ++k) {
    follow |= (state & prog._shift_masks[k]) << prog._shift_amounts[k];
  }

  // Phase 2: exception transitions (backward / large-span)
  g_state_t exc = state & prog.exception_mask;
  while (exc) {
    // Index of lowest set bit: __ffsll (device) or __builtin_ctzll (host)
#ifdef __CUDA_ARCH__
    uint32_t const p = static_cast<uint32_t>(__ffsll(static_cast<long long>(exc))) - 1u;
#else
    uint32_t const p = static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(exc)));
#endif
    exc &= exc - 1;  // clear lowest set bit
    follow |= prog._exception_succs[p];
  }

  return follow;
}

// ---------------------------------------------------------------------------
// Reach computation: bitmask of positions that match character c
// ---------------------------------------------------------------------------

/**
 * @brief Compute the reach bitmask for character @p c.
 *
 * reach(c) = { p | position p matches character c }
 *
 * Fast path (ASCII, c < 128): single lookup into the precomputed _reach_ascii
 * table — O(1), one global-memory load.
 * Slow path (non-ASCII): O(num_states) loop over position descriptors.
 */
__device__ __forceinline__ g_state_t glushkov_compute_reach(
  char32_t const c,
  glushkov_program_device const& prog)
{
  if (c < 128u) { return prog._reach_ascii[c]; }

  g_state_t reach = 0;
  for (uint32_t p = 0; p < prog.num_states; ++p) {
    if (glushkov_position_matches(prog._positions[p], c, prog._classes,
                                  prog._codepoint_flags)) {
      reach |= g_state_t(1) << p;
    }
  }
  return reach;
}

// ---------------------------------------------------------------------------
// Main find function
// ---------------------------------------------------------------------------

/**
 * @brief Glushkov NFA find: locate the leftmost (greedy) match in @p d_str.
 *
 * Semantics match reprog_device::find (Thompson NFA) as closely as possible:
 *   - Leftmost start position.
 *   - Greedy (longest match from that start).
 *   - Nullable patterns produce zero-length matches.
 *
 * @tparam P  Positional mode (BEGIN_END or END_ONLY).
 * @param prog   Glushkov device program.
 * @param d_str  Input string.
 * @param begin  Iterator to the first character to consider.
 * @param end    Max start-position to try (−1 = try all positions). Mirrors the
 *               Thompson NFA semantics: `end=1` means only try starting at 0
 *               (used by matches_re / beginning_only); the NFA always runs to
 *               the end of the string once a start position is chosen.
 * @return       Match pair or nullopt.
 */
template <positional P = positional::BEGIN_END>
__device__ __forceinline__ match_result glushkov_find(
  glushkov_program_device const& prog,
  string_view const d_str,
  string_view::const_iterator begin,
  cudf::size_type end)
{
  // size_bytes() is O(1) — a stored field.  We avoid length() (O(n)) entirely
  // by using byte-offset checks for loop termination.  For nullable patterns,
  // the only case needing special handling is when begin is already at/past
  // end-of-string, which we handle with a pre-loop check.
  auto const size_bytes = static_cast<int32_t>(d_str.size_bytes());  // O(1)

  int32_t const start_pos = begin.position();

  // Nullable patterns can match zero-length at end-of-string.  If begin is
  // already at or past the last byte, return immediately without scanning.
  if (prog.nullable && begin.byte_offset() >= size_bytes) {
    if constexpr (P == positional::BEGIN_END) {
      return match_pair{start_pos, start_pos};
    } else {
      return match_pair{-1, start_pos};
    }
  }

  // Outer loop: try each starting position (leftmost-first search).
  // Termination always uses a byte-offset check (O(1)).  When end >= 0,
  // we also check the character-position bound supplied by the caller.
  auto outer_itr = begin;
  for (int32_t start = start_pos; ; ++start, ++outer_itr) {
    // Outer loop termination (end is an exclusive upper bound, matching Thompson)
    if (end >= 0 && start >= end) { break; }
    if (outer_itr.byte_offset() >= size_bytes) { break; }

    // Optimization: skip positions whose first character cannot begin any match.
    if (!prog.nullable) {
      if (prog.has_startchar) {
        // Fast path: tight byte scan to next occurrence of the literal start
        // character (mirrors Thompson NFA's find_char).  Much cheaper than a
        // reach-table lookup per byte since it is just a register compare.
        while (outer_itr.byte_offset() < size_bytes && *outer_itr != prog.startchar) {
          ++outer_itr;
          ++start;
        }
        if (outer_itr.byte_offset() >= size_bytes) { break; }
        // Re-check end bound: the skip may have advanced past it
        if (end >= 0 && start >= end) { break; }
      } else {
        // General path: reach(c) & first_set == 0 → no NFA position in
        // first_set accepts this character; skip without running the NFA.
        char32_t const first_c = *outer_itr;
        if ((glushkov_compute_reach(first_c, prog) & prog.first_set) == 0) { continue; }
      }
    }

    match_result cur_match{};

    // Nullable: a zero-length match is possible at every position
    if (prog.nullable) {
      if constexpr (P == positional::BEGIN_END) {
        cur_match = match_pair{start, start};
      } else {
        cur_match = match_pair{-1, start};
      }
    }

    // Inner loop: run anchored Glushkov NFA from 'start' to end of string.
    // Use byte-offset bound (O(1)) rather than character-count bound so that
    // this loop never requires a prior d_str.length() call.
    g_state_t state = prog.first_set;
    auto inner_itr  = outer_itr;

    for (int32_t pos = start; inner_itr.byte_offset() < size_bytes; ++pos, ++inner_itr) {
      char32_t const c = *inner_itr;

      if (pos == start) {
        // First character: intersect first_set with reach(c)
        state = state & glushkov_compute_reach(c, prog);
      } else {
        // Subsequent characters: follow → reach
        state = glushkov_compute_follow(state, prog) & glushkov_compute_reach(c, prog);
      }

      if (state == 0) { break; }  // Dead state – no match possible from 'start'

      if (state & prog.accept_mask) {
        // Greedy: always update to capture the longest match from 'start'
        if constexpr (P == positional::BEGIN_END) {
          cur_match = match_pair{start, pos + 1};
        } else {
          cur_match = match_pair{-1, pos + 1};
        }
      }
    }

    if (cur_match) { return cur_match; }  // Leftmost match found
  }

  return {};  // No match
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
