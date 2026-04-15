/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file glushkov_regcomp.cpp
 * @brief Glushkov NFA compiler: ε-elimination from a Thompson NFA.
 *
 * Algorithm overview
 * ------------------
 * The Thompson NFA instructions form a directed graph where:
 *   - Character-consuming nodes (CHAR, ANY, ANYNL, CCLASS, NCCLASS) are the
 *     "positions" of Glushkov's construction.
 *   - ε-transition nodes (LBRA, RBRA, OR) are transparent: they are traversed
 *     during ε-closure without consuming input.
 *   - END is the accept node.
 *
 * For each position p we compute:
 *   follow(p) = ε_closure( p.next_id ) ∩ {char-consuming nodes}
 *   is_accept(p) = ε_closure( p.next_id ) reaches END
 *
 * The initial set (first_set) is built from ε_closure(startinst_id).
 *
 * Shift-and optimisation (Hyperscan technique)
 * --------------------------------------------
 * Positions are numbered by their instruction index order (low → high).
 * Most follow transitions go from a smaller to a slightly larger index, i.e.
 * they have a small positive "span" s = q - p.
 * We store up to GLUSHKOV_MAX_SHIFTS such spans as shift masks so that the
 * GPU can compute most of follow(state) with a handful of shift+OR operations.
 * Remaining transitions (backward / large-span) are stored as per-position
 * exception successor masks.
 */

#include "strings/regex/glushkov_regcomp.h"

#include <algorithm>
#include <map>
#include <memory>
#include <unordered_set>
#include <vector>

namespace cudf {
namespace strings {
namespace detail {

namespace {

/// True for instructions that consume a character (Glushkov positions).
bool is_char_consuming(int32_t const type)
{ return type == CHAR || type == ANY || type == ANYNL || type == CCLASS || type == NCCLASS; }

/**
 * @brief Returns true if ASCII character @p c matches the host-side character class @p cls.
 *
 * Mirrors reclass_device::is_match for ASCII input (codepoint == char value),
 * so the same builtin flags and literal ranges produce identical results on host
 * and device for characters in [0, 127].
 */
bool host_reclass_match_ascii(reclass const& cls, uint8_t const c)
{
  for (auto const& lit : cls.literals) {
    if (static_cast<char32_t>(c) >= lit.first && static_cast<char32_t>(c) <= lit.last) {
      return true;
    }
  }
  if (!cls.builtins) { return false; }

  bool const is_alnum = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
  bool const is_space = (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v');
  bool const is_digit = (c >= '0' && c <= '9');

  if ((cls.builtins & CCLASS_W) && (c == '_' || is_alnum)) return true;
  if ((cls.builtins & CCLASS_S) && is_space) return true;
  if ((cls.builtins & CCLASS_D) && is_digit) return true;
  if ((cls.builtins & NCCLASS_W) && (c != '\n' && c != '_' && !is_alnum)) return true;
  if ((cls.builtins & NCCLASS_S) && !is_space) return true;
  if ((cls.builtins & NCCLASS_D) && (c != '\n' && !is_digit)) return true;
  return false;
}

/// True for zero-width assertion instructions (not handled by Glushkov path).
bool is_assertion(int32_t const type)
{ return type == BOL || type == EOL || type == BOW || type == NBOW; }

// ---------------------------------------------------------------------------
// ε-closure traversal
// ---------------------------------------------------------------------------

/**
 * @brief Depth-first ε-closure from @p inst_id.
 *
 * Fills:
 *   @p char_positions  – IDs of reachable character-consuming instructions.
 *   @p is_accept       – set to true if END is reachable.
 *   @p has_assert      – set to true if an assertion instruction is encountered.
 */
void eps_closure(int32_t const inst_id,
                 reprog const& prog,
                 std::unordered_set<int32_t>& seen,
                 std::vector<int32_t>& char_positions,
                 bool& is_accept,
                 bool& has_assert)
{
  if (inst_id < 0 || inst_id >= prog.insts_count()) return;
  if (!seen.insert(inst_id).second) return;  // already visited

  auto const& inst = prog.insts_data()[inst_id];
  switch (inst.type) {
    // Character-consuming: record position, do NOT follow further
    case CHAR:
    case ANY:
    case ANYNL:
    case CCLASS:
    case NCCLASS: char_positions.push_back(inst_id); break;

    // Accept
    case END: is_accept = true; break;

    // ε-transitions: follow without consuming
    case LBRA:
    case RBRA:
      eps_closure(inst.u2.next_id, prog, seen, char_positions, is_accept, has_assert);
      break;

    case OR:
      eps_closure(inst.u2.left_id, prog, seen, char_positions, is_accept, has_assert);
      eps_closure(inst.u1.right_id, prog, seen, char_positions, is_accept, has_assert);
      break;

    // Zero-width assertions: Glushkov path does not support these
    case BOL:
    case EOL:
    case BOW:
    case NBOW: has_assert = true; break;

    default: break;
  }
}

/// Wrapper that creates fresh local bookkeeping objects.
std::vector<int32_t> eps_closure_from(int32_t const start,
                                      reprog const& prog,
                                      bool& is_accept,
                                      bool& has_assert)
{
  std::unordered_set<int32_t> seen;
  std::vector<int32_t> positions;
  eps_closure(start, prog, seen, positions, is_accept, has_assert);
  return positions;
}

// ---------------------------------------------------------------------------
// Thompson-priority frontier and conflict detection
// ---------------------------------------------------------------------------

/// One item in an ordered ε-frontier: either a char-consuming position or END.
struct frontier_item {
  enum kind_t : int32_t { CHAR_POS, ACCEPT };
  kind_t kind;
  int32_t gpos;  ///< Glushkov position index; valid only when kind == CHAR_POS
};

/**
 * @brief ε-closure in Thompson priority order (right_id before left_id for OR).
 *
 * Visits OR branches in the order Thompson's NFA executor would: right_id
 * (first alternative, higher priority) before left_id (second alternative).
 * Records each distinct char-consuming position or END in first-visit order.
 */
void ordered_eps_frontier(int32_t const inst_id,
                          reprog const& prog,
                          std::vector<int32_t> const& inst_to_pos,
                          std::unordered_set<int32_t>& seen,
                          std::vector<frontier_item>& items)
{
  if (inst_id < 0 || inst_id >= prog.insts_count()) return;
  if (!seen.insert(inst_id).second) return;

  auto const& inst = prog.insts_data()[inst_id];
  switch (inst.type) {
    case CHAR:
    case ANY:
    case ANYNL:
    case CCLASS:
    case NCCLASS: items.push_back({frontier_item::CHAR_POS, inst_to_pos[inst_id]}); break;

    case END: items.push_back({frontier_item::ACCEPT, -1}); break;

    case LBRA:
    case RBRA: ordered_eps_frontier(inst.u2.next_id, prog, inst_to_pos, seen, items); break;

    case OR:
      // Thompson priority: right_id (first alternative) before left_id (second)
      ordered_eps_frontier(inst.u1.right_id, prog, inst_to_pos, seen, items);
      ordered_eps_frontier(inst.u2.left_id, prog, inst_to_pos, seen, items);
      break;

    default: break;
  }
}

/**
 * @brief Returns true if Glushkov position @p p matches ASCII character @p c.
 */
bool position_matches_char(glushkov_host_program const& gp, uint32_t const p, uint8_t const c)
{
  switch (gp.pos_inst_type[p]) {
    case CHAR: return gp.pos_ch[p] == static_cast<char32_t>(c);
    case ANY: return (gp.pos_ch[p] == 'N') ? (c != '\n' && c != '\r') : (c != '\n');
    case ANYNL: return true;
    case CCLASS:
    case NCCLASS: {
      bool const m = host_reclass_match_ascii(gp.classes[gp.pos_cls_idx[p]], c);
      return (gp.pos_inst_type[p] == CCLASS) ? m : !m;
    }
    default: return false;
  }
}

/**
 * @brief Returns true if positions @p p and @p q can match a common character.
 *
 * Iterates all ASCII characters for an exact answer.  For non-ASCII input the
 * function is conservative: if either side is a wildcard type (ANY/ANYNL/
 * CCLASS/NCCLASS) it may match non-ASCII code points, so overlap is assumed.
 */
bool positions_chars_overlap(glushkov_host_program const& gp, uint32_t const p, uint32_t const q)
{
  for (int c = 0; c < GLUSHKOV_ASCII_TABLE_SIZE; ++c) {
    if (position_matches_char(gp, p, static_cast<uint8_t>(c)) &&
        position_matches_char(gp, q, static_cast<uint8_t>(c))) {
      return true;
    }
  }
  // Non-ASCII: conservative overlap if both positions are non-CHAR types
  bool const p_nonascii = (gp.pos_inst_type[p] != CHAR) || (gp.pos_ch[p] >= 128);
  bool const q_nonascii = (gp.pos_inst_type[q] != CHAR) || (gp.pos_ch[q] >= 128);
  if (p_nonascii && q_nonascii) { return true; }
  // One side is a non-ASCII CHAR literal; if the other is a wildcard it might match
  if (gp.pos_inst_type[p] == CHAR && gp.pos_ch[p] >= 128) { return (gp.pos_inst_type[q] != CHAR); }
  if (gp.pos_inst_type[q] == CHAR && gp.pos_ch[q] >= 128) { return (gp.pos_inst_type[p] != CHAR); }
  return false;
}

/**
 * @brief Returns true if the frontier contains a priority conflict that
 *        Glushkov's bit-order cannot represent.
 *
 * Two rules:
 *   Rule 1 – END before char: an ACCEPT item appears before the first CHAR_POS
 *             item in Thompson priority order → the pattern is nullable in a way
 *             that priority_kill cannot handle correctly.
 *   Rule 2 – non-monotone gpos + char overlap: two CHAR_POS items appear with
 *             the higher-priority one at a larger gpos (inverted bit order), AND
 *             they can match a common character → priority_kill picks the wrong
 *             alternative when both are active.
 */
bool frontier_has_priority_conflict(std::vector<frontier_item> const& items,
                                    glushkov_host_program const& gp)
{
  // Rule 1: ACCEPT before any CHAR_POS, but only when the frontier also
  // contains at least one CHAR_POS.  An ACCEPT-only frontier (the normal
  // "end of pattern" case) is not a priority conflict.
  bool seen_char          = false;
  bool accept_before_char = false;
  for (auto const& item : items) {
    if (item.kind == frontier_item::CHAR_POS) {
      seen_char = true;
    } else if (item.kind == frontier_item::ACCEPT && !seen_char) {
      accept_before_char = true;
    }
  }
  if (accept_before_char && seen_char) { return true; }

  // Rule 2: non-monotone gpos pair with character overlap
  for (size_t i = 0; i < items.size(); ++i) {
    if (items[i].kind != frontier_item::CHAR_POS) continue;
    for (size_t j = i + 1; j < items.size(); ++j) {
      if (items[j].kind != frontier_item::CHAR_POS) continue;
      // items[i] has higher Thompson priority than items[j].
      // If items[i].gpos > items[j].gpos, bit order is inverted.
      if (items[i].gpos > items[j].gpos) {
        if (positions_chars_overlap(
              gp, static_cast<uint32_t>(items[i].gpos), static_cast<uint32_t>(items[j].gpos))) {
          return true;
        }
      }
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Shift-and mask construction
// ---------------------------------------------------------------------------

/**
 * @brief Build shift masks and exception tables from the follow_table.
 *
 * For each pair (p, q) where q ∈ follow_table[p]:
 *   span = q - p
 *   • 1 ≤ span ≤ 63 (forward): candidate for a shift-mask slot.
 *   • otherwise          : unconditional exception (backward / 0-span / >63).
 *
 * We select the GLUSHKOV_MAX_SHIFTS most-populated span values as shift slots.
 * Any span that does not get a slot is demoted to the exception table.
 */
void build_shift_masks(glushkov_host_program& gp)
{
  // Collect per-span source-position bitmasks
  std::map<int32_t, g_state_t> span_to_mask;

  for (uint32_t p = 0; p < gp.num_states; ++p) {
    g_state_t follow = gp.follow_table[p];
    while (follow) {
      uint32_t const q =
        static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(follow)));
      follow &= follow - 1;

      int32_t const span = static_cast<int32_t>(q) - static_cast<int32_t>(p);
      if (span >= 1 && span <= 63) {
        span_to_mask[span] |= g_state_t(1) << p;
      } else {
        // Unconditional exception
        gp.exception_mask |= g_state_t(1) << p;
        gp.exception_succs[p] |= g_state_t(1) << q;
      }
    }
  }

  // Sort candidate spans by population (descending) to pick the best slots
  std::vector<std::pair<int32_t, g_state_t>> span_list(span_to_mask.begin(), span_to_mask.end());
  std::sort(span_list.begin(), span_list.end(), [](auto const& a, auto const& b) {
    return __builtin_popcountll(static_cast<long long>(a.second)) >
           __builtin_popcountll(static_cast<long long>(b.second));
  });

  // Use a local const for the loop bound so GCC's static analyzer can prove
  // k < GLUSHKOV_MAX_SHIFTS at each array access (avoids -Wstringop-overflow
  // false positive when the function is inlined in release builds).
  uint32_t const shift_count =
    static_cast<uint32_t>(std::min<size_t>(span_list.size(), GLUSHKOV_MAX_SHIFTS));
  gp.shift_count = shift_count;

  for (uint32_t k = 0; k < shift_count; ++k) {
    gp.shift_amounts[k] = static_cast<uint8_t>(span_list[k].first);
    gp.shift_masks[k]   = span_list[k].second;
  }

  // Demote overflow spans to exceptions
  for (size_t i = gp.shift_count; i < span_list.size(); ++i) {
    g_state_t src   = span_list[i].second;
    int32_t const s = span_list[i].first;
    while (src) {
      uint32_t const p =
        static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(src)));
      src &= src - 1;
      uint32_t const q = static_cast<uint32_t>(static_cast<int32_t>(p) + s);
      gp.exception_mask |= g_state_t(1) << p;
      gp.exception_succs[p] |= g_state_t(1) << q;
    }
  }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

std::unique_ptr<glushkov_host_program> build_glushkov_program(reprog const& prog)
{
  int32_t const num_insts = prog.insts_count();

  // ---- Step 1: Reject ineligible patterns ----------------------------------
  // Glushkov cannot handle zero-width assertions or lazy quantifiers (the
  // bit-parallel NFA always returns the greedy/longest match).
  if (prog.has_lazy()) { return nullptr; }
  for (int32_t i = 0; i < num_insts; ++i) {
    if (is_assertion(prog.insts_data()[i].type)) { return nullptr; }
  }

  // ---- Step 2: Enumerate all character-consuming instruction IDs ----------
  std::vector<int32_t> char_insts;
  char_insts.reserve(static_cast<size_t>(num_insts));
  for (int32_t i = 0; i < num_insts; ++i) {
    if (is_char_consuming(prog.insts_data()[i].type)) { char_insts.push_back(i); }
  }

  if (static_cast<int32_t>(char_insts.size()) > GLUSHKOV_MAX_STATES) { return nullptr; }
  if (char_insts.empty()) { return nullptr; }  // degenerate (e.g. empty pattern)

  // Map instruction ID → Glushkov position index (0-based, left-to-right order)
  std::vector<int32_t> inst_to_pos(static_cast<size_t>(num_insts), -1);
  for (int32_t idx = 0; idx < static_cast<int32_t>(char_insts.size()); ++idx) {
    inst_to_pos[char_insts[idx]] = idx;
  }

  auto gp        = std::make_unique<glushkov_host_program>();
  gp->num_states = static_cast<uint32_t>(char_insts.size());

  // ---- Step 3: Per-position character-matching descriptors ----------------
  for (uint32_t idx = 0; idx < gp->num_states; ++idx) {
    int32_t const inst_id  = char_insts[idx];
    auto const& inst       = prog.insts_data()[inst_id];
    gp->pos_inst_type[idx] = inst.type;
    if (inst.type == CHAR) {
      gp->pos_ch[idx] = inst.u1.c;
    } else if (inst.type == ANY) {
      // Store the newline-mode flag from the Thompson instruction:
      //   'N' = EXT_NEWLINE (reject all is_newline chars)
      //   anything else = default (reject only '\n')
      gp->pos_ch[idx] = inst.u1.c;
    } else if (inst.type == CCLASS || inst.type == NCCLASS) {
      gp->pos_cls_idx[idx] = inst.u1.cls_id;
    }
  }

  // Copy character class definitions (referenced by CCLASS/NCCLASS positions)
  gp->classes.resize(static_cast<size_t>(prog.classes_count()));
  for (int32_t i = 0; i < prog.classes_count(); ++i) {
    gp->classes[i] = prog.class_at(i);
  }

  // ---- Step 4: first_set = ε_closure(startinst) ∩ {char-consuming} ------
  {
    bool is_accept   = false;
    bool has_assert  = false;
    auto const plist = eps_closure_from(prog.get_start_inst(), prog, is_accept, has_assert);
    if (has_assert) { return nullptr; }
    for (int32_t iid : plist) {
      int32_t const pos = inst_to_pos[iid];
      if (pos >= 0) { gp->first_set |= g_state_t(1) << pos; }
    }
    if (is_accept) { gp->nullable = true; }
  }

  // Nullable patterns have ε-paths not represented as Glushkov positions.
  // When the ε-path is the first alternative (e.g. `(|a)`), Thompson gives it
  // highest priority (its END fires first), but Glushkov has no position to
  // represent that priority.  Fall back to Thompson for correctness.
  if (gp->nullable) { return nullptr; }

  // ---- Step 5: follow_table + accept_mask for each position --------------
  for (uint32_t idx = 0; idx < gp->num_states; ++idx) {
    int32_t const inst_id = char_insts[idx];
    int32_t const next_id = prog.insts_data()[inst_id].u2.next_id;

    bool is_accept          = false;
    bool has_assert         = false;
    auto const follow_insts = eps_closure_from(next_id, prog, is_accept, has_assert);
    if (has_assert) { return nullptr; }

    for (int32_t fiid : follow_insts) {
      int32_t const fpos = inst_to_pos[fiid];
      if (fpos >= 0) { gp->follow_table[idx] |= g_state_t(1) << fpos; }
    }
    if (is_accept) { gp->accept_mask |= g_state_t(1) << idx; }
  }

  // ---- Step 5b: Detect Thompson priority conflicts ------------------------
  // Verify that Glushkov's bit-order priority agrees with Thompson's branch-
  // priority order for every ε-frontier in the NFA.  If a conflict is found
  // (Rule 1: accept before char; Rule 2: inverted gpos pair with char overlap),
  // the pattern cannot be correctly simulated by Glushkov → fall back.
  {
    std::unordered_set<int32_t> seen;
    std::vector<frontier_item> items;

    // Check start frontier
    ordered_eps_frontier(prog.get_start_inst(), prog, inst_to_pos, seen, items);
    if (frontier_has_priority_conflict(items, *gp)) { return nullptr; }

    // Check follow frontier for each char-consuming position
    for (uint32_t idx = 0; idx < gp->num_states; ++idx) {
      int32_t const inst_id = char_insts[idx];
      int32_t const next_id = prog.insts_data()[inst_id].u2.next_id;
      seen.clear();
      items.clear();
      ordered_eps_frontier(next_id, prog, inst_to_pos, seen, items);
      if (frontier_has_priority_conflict(items, *gp)) { return nullptr; }
    }
  }

  // ---- Step 6: Shift-and masks ------------------------------------------
  build_shift_masks(*gp);

  // ---- Step 7: Precompute ASCII reach table -------------------------------
  // For each ASCII character c (0-127), compute the bitmask of positions that
  // match c.  This converts the O(num_states) on-the-fly loop in
  // glushkov_compute_reach into a single array lookup for ASCII input.
  for (int32_t c = 0; c < GLUSHKOV_ASCII_TABLE_SIZE; ++c) {
    for (uint32_t p = 0; p < gp->num_states; ++p) {
      bool matches = false;
      switch (gp->pos_inst_type[p]) {
        case CHAR: matches = (gp->pos_ch[p] == static_cast<char32_t>(c)); break;
        case ANY: {
          // 'N' = EXT_NEWLINE: reject all ASCII newlines (\n, \r)
          // default: reject only \n (matches Thompson NFA behaviour)
          matches = (gp->pos_ch[p] == 'N') ? (c != '\n' && c != '\r') : (c != '\n');
          break;
        }
        case ANYNL: matches = true; break;
        case CCLASS:
        case NCCLASS: {
          bool const m =
            host_reclass_match_ascii(gp->classes[gp->pos_cls_idx[p]], static_cast<uint8_t>(c));
          matches = (gp->pos_inst_type[p] == CCLASS) ? m : !m;
          break;
        }
        default: break;
      }
      if (matches) { gp->reach_ascii[c] |= g_state_t(1) << p; }
    }
  }

  // ---- Step 8: First-character skip optimisation -------------------------
  // When every position in first_set is a CHAR instruction for the same
  // literal, record that character so glushkov_find can use a tight byte-scan
  // (like Thompson NFA's find_char) instead of a reach-table lookup per byte.
  if (gp->first_set != 0) {
    g_state_t fs    = gp->first_set;
    char32_t common = 0;
    bool all_char   = true;
    bool seen_first = false;
    while (fs) {
      uint32_t const p =
        static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(fs)));
      fs &= fs - 1;
      if (gp->pos_inst_type[p] != CHAR) {
        all_char = false;
        break;
      }
      if (!seen_first) {
        common     = gp->pos_ch[p];
        seen_first = true;
      } else if (gp->pos_ch[p] != common) {
        all_char = false;
        break;
      }
    }
    if (all_char) {
      gp->has_startchar = true;
      gp->startchar     = common;
    }
  }

  return gp;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
