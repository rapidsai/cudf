/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
// TEST FST SPECIFICATIONS
//------------------------------------------------------------------------------
// FST to check for brackets and braces outside of pairs of quotes
enum class dfa_states : char {
  // The state being active while being outside of a string. When encountering an opening bracket or
  // curly brace, we push it onto the stack. When encountering a closing bracket or brace, we pop it
  // from the stack.
  TT_OOS = 0U,
  // The state being active while being within a string (e.g., field name or a string value). We do
  // not push or pop from the stack while being in this state.
  TT_STR,
  // The state being active after encountering an escape symbol (e.g., '\') while being in the
  // TT_STR state.
  TT_ESC,
  // Total number of states
  TT_NUM_STATES
};

/**
 * @brief Definition of the symbol groups
 */
enum class dfa_symbol_group_id : uint32_t {
  OPENING_BRACE,     ///< Opening brace SG: {
  OPENING_BRACKET,   ///< Opening bracket SG: [
  CLOSING_BRACE,     ///< Closing brace SG: }
  CLOSING_BRACKET,   ///< Closing bracket SG: ]
  QUOTE_CHAR,        ///< Quote character SG: "
  ESCAPE_CHAR,       ///< Escape character SG: '\'
  OTHER_SYMBOLS,     ///< SG implicitly matching all other characters
  NUM_SYMBOL_GROUPS  ///< Total number of symbol groups
};

// Aliases for readability of the transition table
constexpr auto TT_OOS = dfa_states::TT_OOS;
constexpr auto TT_STR = dfa_states::TT_STR;
constexpr auto TT_ESC = dfa_states::TT_ESC;

constexpr auto TT_NUM_STATES     = static_cast<char>(dfa_states::TT_NUM_STATES);
constexpr auto NUM_SYMBOL_GROUPS = static_cast<uint32_t>(dfa_symbol_group_id::NUM_SYMBOL_GROUPS);

// Transition table
std::array<std::array<dfa_states, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const pda_state_tt{
  {/* IN_STATE          {       [       }       ]       "       \    OTHER */
   /* TT_OOS    */ {{TT_OOS, TT_OOS, TT_OOS, TT_OOS, TT_STR, TT_OOS, TT_OOS}},
   /* TT_STR    */ {{TT_STR, TT_STR, TT_STR, TT_STR, TT_OOS, TT_ESC, TT_STR}},
   /* TT_ESC    */ {{TT_STR, TT_STR, TT_STR, TT_STR, TT_STR, TT_STR, TT_STR}}}};

// Translation table (i.e., for each transition, what are the symbols that we output)
static constexpr auto min_translated_out = 1;
static constexpr auto max_translated_out = 1;
std::array<std::array<std::vector<char>, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const pda_out_tt{
  {/* IN_STATE         {      [      }      ]      "      \    OTHER */
   /* TT_OOS    */ {{{'{'}, {'['}, {'}'}, {']'}, {'x'}, {'x'}, {'x'}}},
   /* TT_STR    */ {{{'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}}},
   /* TT_ESC    */ {{{'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}}}}};

// The i-th string representing all the characters of a symbol group
std::array<std::string, NUM_SYMBOL_GROUPS - 1> const pda_sgs{"{", "[", "}", "]", "\"", "\\"};

// The DFA's starting state
constexpr char start_state = static_cast<char>(dfa_states::TT_OOS);
