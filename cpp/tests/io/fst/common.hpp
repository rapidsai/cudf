/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <string>
#include <vector>

namespace cudf::test::io::json {
//------------------------------------------------------------------------------
// TEST FST SPECIFICATIONS
//------------------------------------------------------------------------------
// FST to check for brackets and braces outside of pairs of quotes
enum DFA_STATES : char {
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

// Definition of the symbol groups
enum PDA_SG_ID {
  OBC = 0U,          ///< Opening brace SG: {
  OBT,               ///< Opening bracket SG: [
  CBC,               ///< Closing brace SG: }
  CBT,               ///< Closing bracket SG: ]
  QTE,               ///< Quote character SG: "
  ESC,               ///< Escape character SG: '\'
  OTR,               ///< SG implicitly matching all other characters
  NUM_SYMBOL_GROUPS  ///< Total number of symbol groups
};

// Transition table
const std::vector<std::vector<DFA_STATES>> pda_state_tt = {
  /* IN_STATE         {       [       }       ]       "       \    OTHER */
  /* TT_OOS    */ {TT_OOS, TT_OOS, TT_OOS, TT_OOS, TT_STR, TT_OOS, TT_OOS},
  /* TT_STR    */ {TT_STR, TT_STR, TT_STR, TT_STR, TT_OOS, TT_ESC, TT_STR},
  /* TT_ESC    */ {TT_STR, TT_STR, TT_STR, TT_STR, TT_STR, TT_STR, TT_STR}};

// Translation table (i.e., for each transition, what are the symbols that we output)
const std::vector<std::vector<std::vector<char>>> pda_out_tt = {
  /* IN_STATE        {      [      }      ]     "  \   OTHER */
  /* TT_OOS    */ {{'{'}, {'['}, {'}'}, {']'}, {'x'}, {'x'}, {'x'}},
  /* TT_STR    */ {{'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}},
  /* TT_ESC    */ {{'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}}};

// The i-th string representing all the characters of a symbol group
const std::vector<std::string> pda_sgs = {"{", "[", "}", "]", "\"", "\\"};

// The DFA's starting state
constexpr DFA_STATES start_state = TT_OOS;

}  // namespace cudf::test::io::json
