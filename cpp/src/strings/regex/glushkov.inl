/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
 * Unanchored search: two-phase O(n) per match.
 *   Phase 1 (forward scan): inject first_set at every character; track the
 *     earliest injection position (inject_start_pos) since state was last 0.
 *     When state & accept_mask fires, record the match end; continue extending
 *     greedily until state dies.
 *   Phase 2 (BEGIN_END only): rescan from inject_start_pos, trying each valid
 *     start position with a single-seed NFA until one reaches accept.  The
 *     earliest start that accepts gives the leftmost match.
 *   END_ONLY (contains_re, matches_re): phase 1 result is returned directly;
 *     the start position is not needed so phase 2 is skipped.
 *
 * Anchored (end >= 0): inner-loop from start_pos, already O(n) per string.
 *
 * Note: nullable patterns are rejected at compile time by build_glushkov_program
 * (they fall back to Thompson NFA), so prog.nullable is always false here.
 *
 * The shift-and follow computation (Hyperscan-style):
 *   follow(state) ≈ OR_k( (state & shift_masks[k]) << shift_amounts[k] )
 *                   | OR over active exception states (exception_succs lookup)
 *
 * DataSource pattern
 * ------------------
 * compute_follow_impl, compute_reach_impl, and glushkov_find_impl are
 * templated on a DataSource type that abstracts where the program arrays are
 * read from (global memory vs. shared-memory cache).  Public overloads with
 * the original signatures act as thin wrappers that construct the appropriate
 * DataSource and delegate to the impl.
 */

namespace cudf {
namespace strings {
namespace detail {

/// Count trailing zeros in a 64-bit value (portable across host and device).
__host__ __device__ __forceinline__ uint32_t glushkov_ctz64(uint64_t x)
{
#ifdef __CUDA_ARCH__
  assert(x != 0ull);
  return static_cast<uint32_t>(__ffsll(static_cast<long long>(x)) - 1);
#else
  return static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(x)));
#endif
}

/**
 * @brief Emulate Thompson's first-alternative-wins priority by killing lower-priority states.
 *
 * When multiple alternatives can accept at the same position, the lowest-indexed accepting
 * position corresponds to the highest-priority alternative (Thompson's relist orders first
 * alternatives first).  All bits above the lowest accepting bit are cleared so that
 * lower-priority paths are killed while self-loops and continuations of the winning path
 * survive.
 *
 * @param state       Current NFA state bitmask.
 * @param accept_mask Bitmask of accepting positions.
 * @return            State with lower-priority alternatives killed.
 */
__host__ __device__ __forceinline__ g_state_t glushkov_priority_kill(g_state_t const state,
                                                                     g_state_t const accept_mask)
{
  auto const lsb_a = glushkov_ctz64(state & accept_mask);
  if (lsb_a < 63) { return state & ((g_state_t(1) << (lsb_a + 1)) - 1); }
  return state;
}

// ---------------------------------------------------------------------------
// Character matching
// ---------------------------------------------------------------------------

/**
 * @brief Returns true if character @p c is matched by Glushkov position @p pos.
 *
 * Mirrors the character-dispatch logic in the Thompson NFA regexec loop.
 */
__device__ __forceinline__ bool glushkov_position_matches(glushkov_position const& pos,
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
// DataSource: abstracts where program arrays are read from
// ---------------------------------------------------------------------------

/// Reads all program arrays from global device memory (pointers in prog).
struct glushkov_global_source {
  glushkov_program_device const& prog;
  __device__ __forceinline__ g_state_t reach_ascii(uint32_t c) const
  { return prog._reach_ascii[c]; }
  __device__ __forceinline__ g_state_t shift_mask(uint32_t k) const { return prog._shift_masks[k]; }
  __device__ __forceinline__ uint8_t shift_amount(uint32_t k) const
  { return prog._shift_amounts[k]; }
  __device__ __forceinline__ g_state_t exception_succ(uint32_t p) const
  { return prog._exception_succs[p]; }
};

// ---------------------------------------------------------------------------
// Follow computation (shift-and + exceptions) — templated impl
// ---------------------------------------------------------------------------

/**
 * @brief Compute the Glushkov follow set for the current state bitmask.
 *
 * Two-phase:
 *   1. Shift masks  – fast, covers most forward transitions.
 *   2. Exception table – backward and large-span transitions.
 *
 * @tparam DataSource  glushkov_global_source or glushkov_shmem_source.
 */
template <typename DataSource>
__device__ __forceinline__ g_state_t glushkov_compute_follow_impl(
  g_state_t const state, glushkov_program_device const& prog, DataSource const& src)
{
  g_state_t follow = 0;

  // Phase 1: shift-and transitions
  for (uint32_t k = 0; k < prog.shift_count; ++k) {
    follow |= (state & src.shift_mask(k)) << src.shift_amount(k);
  }

  // Phase 2: exception transitions (backward / large-span)
  g_state_t exc = state & prog.exception_mask;
  while (exc) {
    uint32_t const p = glushkov_ctz64(exc);
    exc &= exc - 1;  // clear lowest set bit
    follow |= src.exception_succ(p);
  }

  return follow;
}

/// Public overload: reads from global device memory.
__device__ __forceinline__ g_state_t glushkov_compute_follow(g_state_t const state,
                                                             glushkov_program_device const& prog)
{ return glushkov_compute_follow_impl(state, prog, glushkov_global_source{prog}); }

// ---------------------------------------------------------------------------
// Reach computation: bitmask of positions that match character c — templated impl
// ---------------------------------------------------------------------------

/**
 * @brief Compute the reach bitmask for character @p c.
 *
 * reach(c) = { p | position p matches character c }
 *
 * Fast path (ASCII, c < GLUSHKOV_ASCII_TABLE_SIZE): single lookup into the precomputed reach_ascii
 * table — O(1).
 * Slow path (non-ASCII): O(num_states) loop over position descriptors.
 *
 * @tparam DataSource  glushkov_global_source or glushkov_shmem_source.
 */
template <typename DataSource>
__device__ __forceinline__ g_state_t glushkov_compute_reach_impl(
  char32_t const c, glushkov_program_device const& prog, DataSource const& src)
{
  if (c < static_cast<uint32_t>(GLUSHKOV_ASCII_TABLE_SIZE)) { return src.reach_ascii(c); }

  // Non-ASCII fallback: must use prog._positions (global memory)
  g_state_t reach = 0;
  for (uint32_t p = 0; p < prog.num_states; ++p) {
    if (glushkov_position_matches(prog._positions[p], c, prog._classes, prog._codepoint_flags)) {
      reach |= g_state_t(1) << p;
    }
  }
  return reach;
}

/// Public overload: reads from global device memory.
__device__ __forceinline__ g_state_t glushkov_compute_reach(char32_t const c,
                                                            glushkov_program_device const& prog)
{ return glushkov_compute_reach_impl(c, prog, glushkov_global_source{prog}); }

// ---------------------------------------------------------------------------
// Main find function — templated impl
// ---------------------------------------------------------------------------

/**
 * @brief Glushkov NFA find: locate the leftmost (greedy) match in @p d_str.
 *
 * Semantics match reprog_device::find (Thompson NFA) as closely as possible:
 *   - Leftmost start position.
 *   - Greedy (longest match from that start).
 *
 * For unanchored patterns, uses a two-phase O(n) algorithm:
 * Phase 1 injects first_set at every character and extends greedily; when
 * state & accept_mask fires, the match end is recorded.  Phase 2 (BEGIN_END
 * only) rescans from inject_start_pos (the earliest injection position since
 * state was last 0) to find the leftmost valid start for that match end.
 *
 * Nullable patterns are rejected at compile time by build_glushkov_program,
 * so prog.nullable is always false when this function is called.
 *
 * @tparam P          Positional mode (BEGIN_END or END_ONLY).
 * @tparam DataSource glushkov_global_source or glushkov_shmem_source.
 * @param prog   Glushkov device program.
 * @param d_str  Input string.
 * @param begin  Iterator to the first character to consider.
 * @param end    Max start-position to try (−1 = try all positions). Mirrors the
 *               Thompson NFA semantics: `end=1` means only try starting at 0
 *               (used by matches_re / beginning_only); the NFA always runs to
 *               the end of the string once a start position is chosen.
 * @param src    DataSource for reading program arrays.
 * @return       Match pair or nullopt.
 */
template <positional P = positional::BEGIN_END, typename DataSource>
__device__ __forceinline__ match_result glushkov_find_impl(glushkov_program_device const& prog,
                                                           string_view const d_str,
                                                           string_view::const_iterator begin,
                                                           cudf::size_type end,
                                                           DataSource const& src)
{
  auto const size_bytes = static_cast<int32_t>(d_str.size_bytes());  // O(1)

  int32_t const start_pos = begin.position();

  // --- Anchored (end >= 0): inner-loop approach, O(n) ---
  // Outer loop runs at most (end - start_pos) iterations (typically 1 for matches_re).
  if (end >= 0) {
    auto outer_itr = begin;
    for (int32_t start = start_pos;; ++start, ++outer_itr) {
      if (start >= end) { break; }
      if (outer_itr.byte_offset() >= size_bytes) { break; }

      if (prog.has_startchar) {
        while (outer_itr.byte_offset() < size_bytes && *outer_itr != prog.startchar) {
          ++outer_itr;
          ++start;
        }
        if (outer_itr.byte_offset() >= size_bytes) { break; }
        if (start >= end) { break; }
      } else {
        char32_t const first_c = *outer_itr;
        if ((glushkov_compute_reach_impl(first_c, prog, src) & prog.first_set) == 0) { continue; }
      }

      match_result cur_match{};

      g_state_t state = prog.first_set;
      auto inner_itr  = outer_itr;
      for (int32_t pos = start; inner_itr.byte_offset() < size_bytes; ++pos, ++inner_itr) {
        char32_t const c = *inner_itr;
        if (pos == start) {
          state = state & glushkov_compute_reach_impl(c, prog, src);
        } else {
          state = glushkov_compute_follow_impl(state, prog, src) &
                  glushkov_compute_reach_impl(c, prog, src);
        }
        if (state == 0) { break; }
        if (state & prog.accept_mask) {
          if constexpr (P == positional::BEGIN_END) {
            cur_match = match_pair{start, pos + 1};
          } else {
            cur_match = match_pair{-1, pos + 1};
          }
          state = glushkov_priority_kill(state, prog.accept_mask);
        }
      }
      if (cur_match) { return cur_match; }
    }
    return {};
  }

  // --- Unanchored: two-phase O(n) search ---
  //
  // Phase 1 (forward scan): inject first_set at every step; detect match END
  // via state & accept_mask (any active seed reaching accept); greedy extension
  // until state dies.
  //
  // inject_start_pos records the earliest position where seeds were injected
  // since state was last 0.  Phase 2 rescans from there to pinpoint the start.
  //
  // Phase 2 (BEGIN_END only): rescan from inject_start_pos, trying each valid
  // start position with a single-seed NFA (mirrors the anchored inner loop).
  // The first start whose NFA reaches accept gives the leftmost match.
  //
  // Cost: O(n) for phase 1; O(gap + match_length) per match for phase 2.
  // Each character is processed at most twice overall.

  g_state_t state = 0;
  match_result cur_match{};

  // Earliest position where seeds were injected since state was last 0.
  int32_t inject_start_pos                     = start_pos;
  string_view::const_iterator inject_start_itr = begin;

  auto itr = begin;
  for (int32_t pos = start_pos; itr.byte_offset() < size_bytes; ++pos, ++itr) {
    // Fast skip when no active states and no match found yet.
    if (state == 0 && !cur_match) {
      if (prog.has_startchar) {
        // Tight byte scan for the literal start character.
        while (itr.byte_offset() < size_bytes && *itr != prog.startchar) {
          ++itr;
          ++pos;
        }
        if (itr.byte_offset() >= size_bytes) { break; }
      }
      // Record injection start for phase 2 rescan.
      inject_start_pos = pos;
      inject_start_itr = itr;
    }

    char32_t const c = *itr;
    auto const reach = glushkov_compute_reach_impl(c, prog, src);

    g_state_t state_next =
      (state != 0) ? glushkov_compute_follow_impl(state, prog, src) : g_state_t{0};

    // Inject fresh starts unless in greedy-extension mode.
    if (!cur_match) { state_next |= prog.first_set; }

    state = state_next & reach;

    if (state == 0) {
      if (cur_match) { break; }  // greedy extension died
      continue;
    }

    // Detect match: any active state reaching accept.
    if (state & prog.accept_mask) {
      if constexpr (P == positional::BEGIN_END) {
        cur_match = match_pair{inject_start_pos, pos + 1};  // start is provisional
      } else {
        cur_match = match_pair{-1, pos + 1};
      }
      // Greedy: continue extending.  Fresh start injection stops via
      // the !cur_match gate above.
      state = glushkov_priority_kill(state, prog.accept_mask);
    }
  }

  if (!cur_match) { return {}; }

  // For END_ONLY (contains_re, matches_re), start position is unused.
  if constexpr (P != positional::BEGIN_END) { return cur_match; }

  // --- Phase 2: find the earliest match start (BEGIN_END only) ---
  //
  // Rescan from inject_start_pos.  For each position where first_set
  // matches, run a single-seed NFA forward.  The first start whose NFA
  // reaches accept gives the leftmost match with greedy extension.

  auto rescan_itr = inject_start_itr;
  for (int32_t s = inject_start_pos; s < cur_match->second; ++s, ++rescan_itr) {
    char32_t const c = *rescan_itr;
    auto const reach = glushkov_compute_reach_impl(c, prog, src);
    g_state_t seed   = prog.first_set & reach;
    if (seed == 0) { continue; }

    // Check immediate accept (single-position pattern, e.g. `\d`).
    int32_t greedy_end = -1;
    if (seed & prog.accept_mask) {
      greedy_end = s + 1;
      seed       = glushkov_priority_kill(seed, prog.accept_mask);
    }

    // Advance the single-seed NFA forward.
    auto scan_itr = rescan_itr;
    ++scan_itr;
    for (int32_t p = s + 1; scan_itr.byte_offset() < size_bytes; ++p, ++scan_itr) {
      seed = glushkov_compute_follow_impl(seed, prog, src) &
             glushkov_compute_reach_impl(*scan_itr, prog, src);
      if (seed == 0) { break; }
      if (seed & prog.accept_mask) {
        greedy_end = p + 1;
        seed       = glushkov_priority_kill(seed, prog.accept_mask);
      }
    }

    if (greedy_end >= 0) { return match_pair{s, greedy_end}; }
  }

  // Fallback: should not reach here if phase 1 detected a match correctly.
  return cur_match;
}

/**
 * @brief Glushkov NFA find: locate the leftmost (greedy) match in @p d_str.
 *
 * Reads all program arrays from global device memory.
 */
template <positional P = positional::BEGIN_END>
__device__ __forceinline__ match_result glushkov_find(glushkov_program_device const& prog,
                                                      string_view const d_str,
                                                      string_view::const_iterator begin,
                                                      cudf::size_type end)
{ return glushkov_find_impl<P>(prog, d_str, begin, end, glushkov_global_source{prog}); }

// ===========================================================================
// Shared-memory-cached variants (CUDA device code only)
// ===========================================================================
#ifdef __CUDACC__

/// Reads program arrays from the block-shared cache populated by
/// @ref glushkov_load_shmem.
struct glushkov_shmem_source {
  glushkov_program_device const& prog;
  glushkov_shmem_cache const* cache;
  __device__ __forceinline__ g_state_t reach_ascii(uint32_t c) const
  { return cache->reach_ascii[c]; }
  __device__ __forceinline__ g_state_t shift_mask(uint32_t k) const
  { return cache->shift_masks[k]; }
  __device__ __forceinline__ uint8_t shift_amount(uint32_t k) const
  { return cache->shift_amounts[k]; }
  __device__ __forceinline__ g_state_t exception_succ(uint32_t p) const
  { return cache->exception_succs[p]; }
};

/**
 * @brief Cooperatively load Glushkov program arrays into shared memory.
 *
 * Must be called by ALL threads in the block.  After return (includes a
 * __syncthreads), the @p cache is ready for use.
 */
__device__ __forceinline__ void glushkov_load_shmem(glushkov_program_device const& prog,
                                                    glushkov_shmem_cache* cache)
{
  // Cooperative load: each thread handles a strided portion of each array.
  for (uint32_t i = threadIdx.x; i < static_cast<uint32_t>(GLUSHKOV_ASCII_TABLE_SIZE);
       i += blockDim.x) {
    cache->reach_ascii[i] = prog._reach_ascii[i];
  }
  for (uint32_t i = threadIdx.x; i < prog.shift_count; i += blockDim.x) {
    cache->shift_masks[i]   = prog._shift_masks[i];
    cache->shift_amounts[i] = prog._shift_amounts[i];
  }
  for (uint32_t i = threadIdx.x; i < prog.num_states; i += blockDim.x) {
    cache->exception_succs[i] = prog._exception_succs[i];
  }
  __syncthreads();
}

/// Overload of glushkov_compute_follow that reads from the shared-memory cache.
__device__ __forceinline__ g_state_t glushkov_compute_follow(g_state_t const state,
                                                             glushkov_program_device const& prog,
                                                             glushkov_shmem_cache const* cache)
{ return glushkov_compute_follow_impl(state, prog, glushkov_shmem_source{prog, cache}); }

/// Overload of glushkov_compute_reach that reads from the shared-memory cache.
__device__ __forceinline__ g_state_t glushkov_compute_reach(char32_t const c,
                                                            glushkov_program_device const& prog,
                                                            glushkov_shmem_cache const* cache)
{ return glushkov_compute_reach_impl(c, prog, glushkov_shmem_source{prog, cache}); }

/**
 * @brief Glushkov NFA find with shared-memory-cached program data.
 *
 * Identical semantics to the non-cached overload.  All read-only program
 * arrays (reach_ascii, shift_masks, exception_succs) are served from
 * shared memory instead of global/L2 cache.
 */
template <positional P = positional::BEGIN_END>
__device__ __forceinline__ match_result glushkov_find(glushkov_program_device const& prog,
                                                      string_view const d_str,
                                                      string_view::const_iterator begin,
                                                      cudf::size_type end,
                                                      glushkov_shmem_cache const* cache)
{ return glushkov_find_impl<P>(prog, d_str, begin, end, glushkov_shmem_source{prog, cache}); }

#endif  // __CUDACC__

}  // namespace detail
}  // namespace strings
}  // namespace cudf
