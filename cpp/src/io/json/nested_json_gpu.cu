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

#include "nested_json.hpp"

#include <io/fst/logical_stack.cuh>
#include <io/fst/lookup_tables.cuh>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

namespace cudf::io::json {

// JSON to stack operator DFA (Deterministic Finite Automata)
namespace to_stack_op {

// Type used to represent the target state in the transition table
using StateT = char;

/**
 * @brief Definition of the DFA's states
 */
enum class dfa_states : StateT {
  // The active state while outside of a string. When encountering an opening bracket or curly
  // brace, we push it onto the stack. When encountering a closing bracket or brace, we pop from the
  // stack.
  TT_OOS = 0U,

  // The active state while within a string (e.g., field name or a string value). We do not push or
  // pop from the stack while in this state.
  TT_STR,

  // The active state after encountering an escape symbol (e.g., '\'), while in the TT_STR state.
  TT_ESC,

  // Total number of states
  TT_NUM_STATES
};

// Aliases for readability of the transition table
constexpr auto TT_OOS = dfa_states::TT_OOS;
constexpr auto TT_STR = dfa_states::TT_STR;
constexpr auto TT_ESC = dfa_states::TT_ESC;

/**
 * @brief Definition of the symbol groups
 */
enum class dfa_symbol_group_id : uint8_t {
  OPENING_BRACE,     ///< Opening brace SG: {
  OPENING_BRACKET,   ///< Opening bracket SG: [
  CLOSING_BRACE,     ///< Closing brace SG: }
  CLOSING_BRACKET,   ///< Closing bracket SG: ]
  QUOTE_CHAR,        ///< Quote character SG: "
  ESCAPE_CHAR,       ///< Escape character SG: '\'
  OTHER_SYMBOLS,     ///< SG implicitly matching all other characters
  NUM_SYMBOL_GROUPS  ///< Total number of symbol groups
};

constexpr auto TT_NUM_STATES     = static_cast<StateT>(dfa_states::TT_NUM_STATES);
constexpr auto NUM_SYMBOL_GROUPS = static_cast<uint32_t>(dfa_symbol_group_id::NUM_SYMBOL_GROUPS);

// The i-th string representing all the characters of a symbol group
std::array<std::string, NUM_SYMBOL_GROUPS - 1> const symbol_groups{
  {{"{"}, {"["}, {"}"}, {"]"}, {"\""}, {"\\"}}};

// Transition table
std::array<std::array<dfa_states, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const transition_table{
  {/* IN_STATE          {       [       }       ]       "       \    OTHER */
   /* TT_OOS    */ {{TT_OOS, TT_OOS, TT_OOS, TT_OOS, TT_STR, TT_OOS, TT_OOS}},
   /* TT_STR    */ {{TT_STR, TT_STR, TT_STR, TT_STR, TT_OOS, TT_ESC, TT_STR}},
   /* TT_ESC    */ {{TT_STR, TT_STR, TT_STR, TT_STR, TT_STR, TT_STR, TT_STR}}}};

// Translation table (i.e., for each transition, what are the symbols that we output)
std::array<std::array<std::vector<char>, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const translation_table{
  {/* IN_STATE         {      [      }      ]      "      \    OTHER */
   /* TT_OOS    */ {{{'{'}, {'['}, {'}'}, {']'}, {'x'}, {'x'}, {'x'}}},
   /* TT_STR    */ {{{'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}}},
   /* TT_ESC    */ {{{'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}, {'x'}}}}};

// The DFA's starting state
constexpr auto start_state = static_cast<StateT>(TT_OOS);
}  // namespace to_stack_op

// JSON tokenizer pushdown automaton
namespace tokenizer_pda {

// Type used to represent the target state in the transition table
using StateT = char;

/**
 * @brief Symbol groups for the input alphabet for the pushdown automaton
 */
enum class symbol_group_id : PdaSymbolGroupIdT {
  /// Opening brace
  OPENING_BRACE,
  /// Opening bracket
  OPENING_BRACKET,
  /// Closing brace
  CLOSING_BRACE,
  /// Closing bracket
  CLOSING_BRACKET,
  /// Quote
  QUOTE,
  /// Escape
  ESCAPE,
  /// Comma
  COMMA,
  /// Colon
  COLON,
  /// Whitespace
  WHITE_SPACE,
  /// Other (any input symbol not assigned to one of the above symbol groups)
  OTHER,
  /// Total number of symbol groups amongst which to differentiate
  NUM_PDA_INPUT_SGS
};

/**
 * @brief Symbols in the stack alphabet
 */
enum class stack_symbol_group_id : PdaStackSymbolGroupIdT {
  /// Symbol representing that we're at the JSON root (nesting level 0)
  STACK_ROOT,

  /// Symbol representing that we're currently within a list object
  STACK_LIST,

  /// Symbol representing that we're currently within a struct object
  STACK_STRUCT,

  /// Total number of symbols in the stack alphabet
  NUM_STACK_SGS
};
constexpr auto NUM_PDA_INPUT_SGS =
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::NUM_PDA_INPUT_SGS);
constexpr auto NUM_STACK_SGS =
  static_cast<PdaStackSymbolGroupIdT>(stack_symbol_group_id::NUM_STACK_SGS);

/// Total number of symbol groups to differentiate amongst (stack alphabet * input alphabet)
constexpr PdaSymbolGroupIdT NUM_PDA_SGIDS = NUM_PDA_INPUT_SGS * NUM_STACK_SGS;

/// Mapping a input symbol to the symbol group id
static __constant__ PdaSymbolGroupIdT tos_sg_to_pda_sgid[] = {
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::WHITE_SPACE),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::WHITE_SPACE),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::WHITE_SPACE),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::WHITE_SPACE),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::QUOTE),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::COMMA),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::COLON),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OPENING_BRACKET),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::ESCAPE),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::CLOSING_BRACKET),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OPENING_BRACE),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::CLOSING_BRACE),
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::OTHER)};

/**
 * @brief Maps a (top-of-stack symbol, input symbol)-pair to a symbol group id of the deterministic
 * visibly pushdown automaton (DVPA)
 */
struct PdaSymbolToSymbolGroupId {
  template <typename SymbolT, typename StackSymbolT>
  __device__ __forceinline__ PdaSymbolGroupIdT
  operator()(thrust::tuple<SymbolT, StackSymbolT> symbol_pair)
  {
    // The symbol read from the input
    auto symbol = thrust::get<0>(symbol_pair);

    // The stack symbol (i.e., what is on top of the stack at the time the input symbol was read)
    // I.e., whether we're reading in something within a struct, a list, or the JSON root
    auto stack_symbol = thrust::get<1>(symbol_pair);

    // The stack symbol offset: '_' is the root group (0), '[' is the list group (1), '{' is the
    // struct group (2)
    int32_t stack_idx = static_cast<PdaStackSymbolGroupIdT>(
      (stack_symbol == '_') ? stack_symbol_group_id::STACK_ROOT
                            : ((stack_symbol == '[') ? stack_symbol_group_id::STACK_LIST
                                                     : stack_symbol_group_id::STACK_STRUCT));

    // The relative symbol group id of the current input symbol
    constexpr auto pda_sgid_lookup_size =
      static_cast<int32_t>(sizeof(tos_sg_to_pda_sgid) / sizeof(tos_sg_to_pda_sgid[0]));
    PdaSymbolGroupIdT symbol_gid =
      tos_sg_to_pda_sgid[min(static_cast<int32_t>(symbol), pda_sgid_lookup_size - 1)];
    return stack_idx * static_cast<PdaSymbolGroupIdT>(symbol_group_id::NUM_PDA_INPUT_SGS) +
           symbol_gid;
  }
};

// The states defined by the pushdown automaton
enum class pda_state_t : StateT {
  // Beginning of value
  PD_BOV,
  // Beginning of array
  PD_BOA,
  // Literal or number
  PD_LON,
  // String
  PD_STR,
  // After escape char when within string
  PD_SCE,
  // After having parsed a value
  PD_PVL,
  // Before the next field name
  PD_BFN,
  // Field name
  PD_FLN,
  // After escape char when within field name
  PD_FNE,
  // After a field name inside a struct
  PD_PFN,
  // Error state (trap state)
  PD_ERR,
  // Total number of PDA states
  PD_NUM_STATES
};

// Aliases for readability of the transition table
constexpr auto PD_BOV = pda_state_t::PD_BOV;
constexpr auto PD_BOA = pda_state_t::PD_BOA;
constexpr auto PD_LON = pda_state_t::PD_LON;
constexpr auto PD_STR = pda_state_t::PD_STR;
constexpr auto PD_SCE = pda_state_t::PD_SCE;
constexpr auto PD_PVL = pda_state_t::PD_PVL;
constexpr auto PD_BFN = pda_state_t::PD_BFN;
constexpr auto PD_FLN = pda_state_t::PD_FLN;
constexpr auto PD_FNE = pda_state_t::PD_FNE;
constexpr auto PD_PFN = pda_state_t::PD_PFN;
constexpr auto PD_ERR = pda_state_t::PD_ERR;

constexpr auto PD_NUM_STATES = static_cast<StateT>(pda_state_t::PD_NUM_STATES);

// The starting state of the pushdown automaton
constexpr auto start_state = static_cast<StateT>(pda_state_t::PD_BOV);

// Identity symbol to symbol group lookup table
std::vector<std::vector<char>> const pda_sgids{
  {0},  {1},  {2},  {3},  {4},  {5},  {6},  {7},  {8},  {9},  {10}, {11}, {12}, {13}, {14},
  {15}, {16}, {17}, {18}, {19}, {20}, {21}, {22}, {23}, {24}, {25}, {26}, {27}, {28}, {29}};

/**
 * @brief Getting the transition table
 */
auto get_transition_table()
{
  std::array<std::array<pda_state_t, NUM_PDA_SGIDS>, PD_NUM_STATES> pda_tt;
  //  {       [       }       ]       "       \       ,       :     space   other
  pda_tt[static_cast<StateT>(pda_state_t::PD_BOV)] = {
    PD_BOA, PD_BOA, PD_ERR, PD_ERR, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_LON,
    PD_BOA, PD_BOA, PD_ERR, PD_ERR, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_LON,
    PD_BOA, PD_BOA, PD_ERR, PD_ERR, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_LON};
  pda_tt[static_cast<StateT>(pda_state_t::PD_BOA)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_BOA, PD_BOA, PD_ERR, PD_PVL, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOA, PD_LON,
    PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_FLN, PD_ERR, PD_ERR, PD_ERR, PD_BOA, PD_ERR};
  pda_tt[static_cast<StateT>(pda_state_t::PD_LON)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_LON,
    PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_BOV, PD_ERR, PD_PVL, PD_LON,
    PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_ERR, PD_BFN, PD_ERR, PD_PVL, PD_LON};
  pda_tt[static_cast<StateT>(pda_state_t::PD_STR)] = {
    PD_STR, PD_STR, PD_STR, PD_STR, PD_PVL, PD_SCE, PD_STR, PD_STR, PD_STR, PD_STR,
    PD_STR, PD_STR, PD_STR, PD_STR, PD_PVL, PD_SCE, PD_STR, PD_STR, PD_STR, PD_STR,
    PD_STR, PD_STR, PD_STR, PD_STR, PD_PVL, PD_SCE, PD_STR, PD_STR, PD_STR, PD_STR};
  pda_tt[static_cast<StateT>(pda_state_t::PD_SCE)] = {
    PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR,
    PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR,
    PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR};
  pda_tt[static_cast<StateT>(pda_state_t::PD_PVL)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_BOV, PD_ERR, PD_PVL, PD_ERR,
    PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_ERR, PD_BFN, PD_ERR, PD_PVL, PD_ERR};
  pda_tt[static_cast<StateT>(pda_state_t::PD_BFN)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_FLN, PD_ERR, PD_ERR, PD_ERR, PD_BFN, PD_ERR};
  pda_tt[static_cast<StateT>(pda_state_t::PD_FLN)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_PFN, PD_FNE, PD_FLN, PD_FLN, PD_FLN, PD_FLN};
  pda_tt[static_cast<StateT>(pda_state_t::PD_FNE)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN};
  pda_tt[static_cast<StateT>(pda_state_t::PD_PFN)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_PFN, PD_ERR};
  pda_tt[static_cast<StateT>(pda_state_t::PD_ERR)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR};
  return pda_tt;
}

/**
 * @brief Getting the translation table
 */
auto get_translation_table()
{
  std::array<std::array<std::vector<char>, NUM_PDA_SGIDS>, PD_NUM_STATES> pda_tlt;
  pda_tlt[static_cast<StateT>(pda_state_t::PD_BOV)] = {{{token_t::StructBegin},
                                                        {token_t::ListBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::StringBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {},
                                                        {token_t::ValueBegin},
                                                        {token_t::StructBegin},
                                                        {token_t::ListBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::StringBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {},
                                                        {token_t::ValueBegin},
                                                        {token_t::StructBegin},
                                                        {token_t::ListBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::StringBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {},
                                                        {token_t::ValueBegin}}};
  pda_tlt[static_cast<StateT>(pda_state_t::PD_BOA)] = {{{token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::StructBegin},
                                                        {token_t::ListBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ListEnd},
                                                        {token_t::StringBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {},
                                                        {token_t::ValueBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::StructEnd},
                                                        {token_t::ErrorBegin},
                                                        {token_t::FieldNameBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {},
                                                        {token_t::ErrorBegin}}};
  pda_tlt[static_cast<StateT>(pda_state_t::PD_LON)] = {{{token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ValueEnd},
                                                        {},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ValueEnd, token_t::ListEnd},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ValueEnd},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ValueEnd},
                                                        {},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ValueEnd, token_t::StructEnd},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ValueEnd},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ValueEnd},
                                                        {}}};
  pda_tlt[static_cast<StateT>(pda_state_t::PD_STR)] = {
    {{}, {}, {}, {}, {token_t::StringEnd}, {}, {}, {}, {}, {},
     {}, {}, {}, {}, {token_t::StringEnd}, {}, {}, {}, {}, {},
     {}, {}, {}, {}, {token_t::StringEnd}, {}, {}, {}, {}, {}}};
  pda_tlt[static_cast<StateT>(pda_state_t::PD_SCE)] = {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {},
                                                        {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
                                                        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}}};
  pda_tlt[static_cast<StateT>(pda_state_t::PD_PVL)] = {{{token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ListEnd},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {},
                                                        {token_t::ErrorBegin},
                                                        {},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::StructEnd},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {},
                                                        {token_t::ErrorBegin},
                                                        {},
                                                        {token_t::ErrorBegin}}};
  pda_tlt[static_cast<StateT>(pda_state_t::PD_BFN)] = {{{token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::FieldNameBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {},
                                                        {token_t::ErrorBegin}}};
  pda_tlt[static_cast<StateT>(pda_state_t::PD_FLN)] = {{{token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {},
                                                        {},
                                                        {},
                                                        {},
                                                        {token_t::FieldNameEnd},
                                                        {},
                                                        {},
                                                        {},
                                                        {},
                                                        {}}};
  pda_tlt[static_cast<StateT>(pda_state_t::PD_FNE)] = {{{token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {},
                                                        {},
                                                        {},
                                                        {},
                                                        {},
                                                        {},
                                                        {},
                                                        {},
                                                        {},
                                                        {}}};
  pda_tlt[static_cast<StateT>(pda_state_t::PD_PFN)] = {{{token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {token_t::ErrorBegin},
                                                        {},
                                                        {},
                                                        {token_t::ErrorBegin}}};
  pda_tlt[static_cast<StateT>(pda_state_t::PD_ERR)] = {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {},
                                                        {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
                                                        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}}};
  return pda_tlt;
}

}  // namespace tokenizer_pda

/**
 * @brief Function object used to filter for brackets and braces that represent push and pop
 * operations
 */
struct JSONToStackOp {
  template <typename StackSymbolT>
  constexpr CUDF_HOST_DEVICE fst::stack_op_type operator()(StackSymbolT const& stack_symbol) const
  {
    return (stack_symbol == '{' || stack_symbol == '[')   ? fst::stack_op_type::PUSH
           : (stack_symbol == '}' || stack_symbol == ']') ? fst::stack_op_type::POP
                                                          : fst::stack_op_type::READ;
  }
};

namespace detail {

void get_stack_context(device_span<SymbolT const> json_in,
                       SymbolT* d_top_of_stack,
                       rmm::cuda_stream_view stream)
{
  constexpr std::size_t single_item = 1;

  // Symbol representing the JSON-root (i.e., we're at nesting level '0')
  constexpr StackSymbolT root_symbol = '_';
  // This can be any stack symbol from the stack alphabet that does not push onto stack
  constexpr StackSymbolT read_symbol = 'x';

  // Number of stack operations in the input (i.e., number of '{', '}', '[', ']' outside of quotes)
  hostdevice_vector<SymbolOffsetT> num_stack_ops(single_item, stream);

  // Sequence of stack symbols and their position in the original input (sparse representation)
  rmm::device_uvector<StackSymbolT> stack_ops{json_in.size(), stream};
  rmm::device_uvector<SymbolOffsetT> stack_op_indices{json_in.size(), stream};

  // Prepare finite-state transducer that only selects '{', '}', '[', ']' outside of quotes
  using ToStackOpFstT =
    cudf::io::fst::detail::Dfa<StackSymbolT,
                               static_cast<int32_t>(
                                 to_stack_op::dfa_symbol_group_id::NUM_SYMBOL_GROUPS),
                               static_cast<int32_t>(to_stack_op::dfa_states::TT_NUM_STATES)>;
  ToStackOpFstT json_to_stack_ops_fst{to_stack_op::symbol_groups,
                                      to_stack_op::transition_table,
                                      to_stack_op::translation_table,
                                      stream};

  // "Search" for relevant occurrence of brackets and braces that indicate the beginning/end
  // of structs/lists
  json_to_stack_ops_fst.Transduce(json_in.begin(),
                                  static_cast<SymbolOffsetT>(json_in.size()),
                                  stack_ops.data(),
                                  stack_op_indices.data(),
                                  num_stack_ops.device_ptr(),
                                  to_stack_op::start_state,
                                  stream);

  // stack operations with indices are converted to top of the stack for each character in the input
  fst::sparse_stack_op_to_top_of_stack<StackLevelT>(
    stack_ops.data(),
    device_span<SymbolOffsetT>{stack_op_indices.data(), stack_op_indices.size()},
    JSONToStackOp{},
    d_top_of_stack,
    root_symbol,
    read_symbol,
    json_in.size(),
    stream);
}

// TODO: return pair of device_uvector instead of passing pre-allocated pointers.
void get_token_stream(device_span<SymbolT const> json_in,
                      PdaTokenT* d_tokens,
                      SymbolOffsetT* d_tokens_indices,
                      SymbolOffsetT* d_num_written_tokens,
                      rmm::cuda_stream_view stream)
{
  // Memory holding the top-of-stack stack context for the input
  rmm::device_uvector<StackSymbolT> stack_op_indices{json_in.size(), stream};

  // Identify what is the stack context for each input character (is it: JSON-root, struct, or list)
  get_stack_context(json_in, stack_op_indices.data(), stream);

  // Prepare for PDA transducer pass, merging input symbols with stack symbols
  rmm::device_uvector<PdaSymbolGroupIdT> pda_sgids{json_in.size(), stream};
  auto zip_in = thrust::make_zip_iterator(json_in.data(), stack_op_indices.data());
  thrust::transform(rmm::exec_policy(stream),
                    zip_in,
                    zip_in + json_in.size(),
                    pda_sgids.data(),
                    tokenizer_pda::PdaSymbolToSymbolGroupId{});

  // PDA transducer alias
  using ToTokenStreamFstT =
    cudf::io::fst::detail::Dfa<StackSymbolT,
                               tokenizer_pda::NUM_PDA_SGIDS,
                               static_cast<tokenizer_pda::StateT>(
                                 tokenizer_pda::pda_state_t::PD_NUM_STATES)>;

  // Instantiating PDA transducer
  ToTokenStreamFstT json_to_tokens_fst{tokenizer_pda::pda_sgids,
                                       tokenizer_pda::get_transition_table(),
                                       tokenizer_pda::get_translation_table(),
                                       stream};

  // Perform a PDA-transducer pass
  json_to_tokens_fst.Transduce(pda_sgids.begin(),
                               static_cast<SymbolOffsetT>(json_in.size()),
                               d_tokens,
                               d_tokens_indices,
                               d_num_written_tokens,
                               tokenizer_pda::start_state,
                               stream);
}

}  // namespace detail

}  // namespace cudf::io::json
