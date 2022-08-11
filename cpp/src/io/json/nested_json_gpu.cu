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

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include "thrust/functional.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/transform_output_iterator.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sequence.h"
#include "thrust/sort.h"
#include "thrust/tabulate.h"
#include <thrust/gather.h>
#include <thrust/scan.h>

#include <stack>

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
  pda_tlt[static_cast<StateT>(pda_state_t::PD_BOA)] = {
    {{token_t::ErrorBegin},
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
     {token_t::StructMemberBegin, token_t::FieldNameBegin},
     {token_t::ErrorBegin},
     {token_t::ErrorBegin},
     {token_t::ErrorBegin},
     {},
     {token_t::ErrorBegin}}};
  pda_tlt[static_cast<StateT>(pda_state_t::PD_LON)] = {
    {{token_t::ErrorBegin},
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
     {token_t::ValueEnd, token_t::StructMemberEnd, token_t::StructEnd},
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
  pda_tlt[static_cast<StateT>(pda_state_t::PD_PVL)] = {
    {{token_t::ErrorBegin},
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
     {token_t::StructMemberEnd, token_t::StructEnd},
     {token_t::ErrorBegin},
     {token_t::ErrorBegin},
     {token_t::ErrorBegin},
     {token_t::StructMemberEnd},
     {token_t::ErrorBegin},
     {},
     {token_t::ErrorBegin}}};
  pda_tlt[static_cast<StateT>(pda_state_t::PD_BFN)] = {
    {{token_t::ErrorBegin},
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
     {token_t::StructMemberBegin, token_t::FieldNameBegin},
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

tree_meta_t get_tree_representation(host_span<SymbolT const> input, rmm::cuda_stream_view stream)
{
  constexpr std::size_t single_item = 1;
  hostdevice_vector<PdaTokenT> tokens_gpu{input.size(), stream};
  hostdevice_vector<SymbolOffsetT> token_indices_gpu{input.size(), stream};
  hostdevice_vector<SymbolOffsetT> num_tokens_out{single_item, stream};

  rmm::device_uvector<SymbolT> d_input{input.size(), stream};
  cudaMemcpyAsync(
    d_input.data(), input.data(), input.size() * sizeof(input[0]), cudaMemcpyHostToDevice, stream);

  // Parse the JSON and get the token stream
  cudf::io::json::detail::get_token_stream(
    cudf::device_span<SymbolT>{d_input.data(), d_input.size()},
    tokens_gpu.device_ptr(),
    token_indices_gpu.device_ptr(),
    num_tokens_out.device_ptr(),
    stream);

  // Copy the JSON tokens to the host
  token_indices_gpu.device_to_host(stream);
  tokens_gpu.device_to_host(stream);
  num_tokens_out.device_to_host(stream);

  // Make sure tokens have been copied to the host
  stream.synchronize();

  auto to_token_str = [](PdaTokenT token) {
    switch (token) {
      case token_t::StructBegin: return " {";
      case token_t::StructEnd: return " }";
      case token_t::ListBegin: return " [";
      case token_t::ListEnd: return " ]";
      case token_t::FieldNameBegin: return "FB";
      case token_t::FieldNameEnd: return "FE";
      case token_t::StringBegin: return "SB";
      case token_t::StringEnd: return "SE";
      case token_t::ErrorBegin: return "er";
      case token_t::ValueBegin: return "VB";
      case token_t::ValueEnd: return "VE";
      case token_t::StructMemberBegin: return " <";
      case token_t::StructMemberEnd: return " >";
      default: return ".";
    }
  };

  // Whether a token does represent a node in the tree representation
  auto is_node = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin:
      case token_t::StringBegin:
      case token_t::ValueBegin:
      case token_t::FieldNameBegin:
      case token_t::ErrorBegin: return true;
      default: return false;
    };
  };

  // The node that a token represents
  auto token_to_node = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructBegin: return NC_STRUCT;
      case token_t::ListBegin: return NC_LIST;
      case token_t::StringBegin: return NC_STR;
      case token_t::ValueBegin: return NC_VAL;
      case token_t::FieldNameBegin: return NC_FN;
      default: return NC_ERR;
    };
  };

  auto get_token_index = [](PdaTokenT const token, SymbolOffsetT const token_index) {
    constexpr SymbolOffsetT skip_quote_char = 1;
    switch (token) {
      case token_t::StringBegin: return token_index + skip_quote_char;
      case token_t::FieldNameBegin: return token_index + skip_quote_char;
      default: return token_index;
    };
  };

  // Whether a token expects to be followed by its respective end-of-* token partner
  auto is_begin_of_section = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StringBegin:
      case token_t::ValueBegin:
      case token_t::FieldNameBegin: return true;
      default: return false;
    };
  };

  // The end-of-* partner token for a given beginning-of-* token
  auto end_of_partner = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StringBegin: return token_t::StringEnd;
      case token_t::ValueBegin: return token_t::ValueEnd;
      case token_t::FieldNameBegin: return token_t::FieldNameEnd;
      default: return token_t::ErrorBegin;
    };
  };

  // Whether the token pops from the parent node stack
  auto does_pop = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructEnd:
      case token_t::ListEnd: return true;
      default: return false;
    };
  };

  // Whether the token pushes onto the parent node stack
  auto does_push = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin: return true;
      default: return false;
    };
  };

  // The node id sitting on top of the stack becomes the node's parent
  // The full stack represents the path from the root to the current node
  std::stack<std::pair<NodeIndexT, bool>> parent_stack;

  constexpr bool field_name_node    = true;
  constexpr bool no_field_name_node = false;

  std::vector<NodeT> node_categories;
  std::vector<NodeIndexT> parent_node_ids;
  std::vector<TreeDepthT> node_levels;
  std::vector<SymbolOffsetT> node_range_begin;
  std::vector<SymbolOffsetT> node_range_end;

  std::size_t node_id = 0;
  for (std::size_t i = 0; i < num_tokens_out[0]; i++) {
    auto token = tokens_gpu[i];

    // The section from the original JSON input that this token demarcates
    std::size_t range_begin = get_token_index(token, token_indices_gpu[i]);
    std::size_t range_end   = range_begin + 1;

    // Identify this node's parent node id
    std::size_t parent_node_id =
      (parent_stack.size() > 0) ? parent_stack.top().first : parent_node_sentinel;

    // If this token is the beginning-of-{value, string, field name}, also consume the next end-of-*
    // token
    if (is_begin_of_section(token)) {
      if ((i + 1) < num_tokens_out[0] && end_of_partner(token) == tokens_gpu[i + 1]) {
        // Update the range_end for this pair of tokens
        range_end = token_indices_gpu[i + 1];
        // We can skip the subsequent end-of-* token
        i++;
      }
    }

    // Emit node if this token becomes a node in the tree
    if (is_node(token)) {
      node_categories.push_back(token_to_node(token));
      parent_node_ids.push_back(parent_node_id);
      node_levels.push_back(parent_stack.size());
      node_range_begin.push_back(range_begin);
      node_range_end.push_back(range_end);
    }

    // Modify the stack if needed
    if (token == token_t::FieldNameBegin) {
      parent_stack.push({node_id, field_name_node});
    } else {
      if (does_push(token)) {
        parent_stack.push({node_id, no_field_name_node});
      } else if (does_pop(token)) {
        CUDF_EXPECTS(parent_stack.size() >= 1, "Invalid JSON input.");
        parent_stack.pop();
      }

      // If what we're left with is a field name on top of stack, we need to pop it
      if (parent_stack.size() >= 1 && parent_stack.top().second == field_name_node) {
        parent_stack.pop();
      }
    }

    // Update node_id
    if (is_node(token)) { node_id++; }
  }

  // DEBUG prints
  auto print_cat = [](auto const& gpu, auto const& cpu, auto const name) {
    auto to_cat = [](auto v) {
      switch (v) {
        case NC_STRUCT: return " S";
        case NC_LIST: return " L";
        case NC_STR: return " \"";
        case NC_VAL: return " V";
        case NC_FN: return " F";
        case NC_ERR: return "ER";
        default: return "UN";
      };
    };
    for (auto const& v : cpu)
      printf("%s,", to_cat(v));
    std::cout << name << "(CPU):" << std::endl;
    for (auto const& v : gpu)
      printf("%s,", to_cat(v));
    std::cout << name << "(GPU):" << std::endl;
    if (!std::equal(gpu.begin(), gpu.end(), cpu.begin())) {
      for (auto i = 0lu; i < cpu.size(); i++)
        printf("%2s,", (gpu[i] == cpu[i] ? " " : "x"));
      std::cout << std::endl;
    }
  };
  bool mismatch  = false;
  auto print_vec = [&](auto const& gpu, auto const& cpu, auto const name) {
    for (auto const& v : cpu)
      printf("%2d,", int(v));
    std::cout << name << "(CPU):" << std::endl;
    for (auto const& v : gpu)
      printf("%2d,", int(v));
    std::cout << name << "(GPU):" << std::endl;
    if (!std::equal(gpu.begin(), gpu.end(), cpu.begin())) {
      for (auto i = 0lu; i < cpu.size(); i++) {
        mismatch |= (gpu[i] != cpu[i]);
        printf("%2s,", (gpu[i] == cpu[i] ? " " : "x"));
      }
      std::cout << std::endl;
    }
  };

#define PRINT_VEC(vec) print_vec(value.vec, vec, #vec);
  auto value = get_tree_representation_gpu(d_input, stream);
  // PRINT_VEC(node_categories); //Works
  print_cat(value.node_categories, node_categories, "node_categories");
  PRINT_VEC(node_levels);       // Works
  PRINT_VEC(node_range_begin);  // Works
  PRINT_VEC(node_range_end);    // Works
  PRINT_VEC(parent_node_ids);   // Works
  CUDF_EXPECTS(!mismatch, "Mismatch in GPU and CPU tree representation");
  std::cout << "Mismatch: " << mismatch << std::endl;

#undef PRINT_VEC
  return {std::move(node_categories),
          std::move(parent_node_ids),
          std::move(node_levels),
          std::move(node_range_begin),
          std::move(node_range_end)};
}

// The node that a token represents
struct token_to_node {
  __device__ auto operator()(PdaTokenT const token) -> NodeT
  {
    switch (token) {
      case token_t::StructBegin: return NC_STRUCT;
      case token_t::ListBegin: return NC_LIST;
      case token_t::StringBegin: return NC_STR;
      case token_t::ValueBegin: return NC_VAL;
      case token_t::FieldNameBegin: return NC_FN;
      default: return NC_ERR;
    };
  }
};

// convert token indices to node range for each vaid node.
template <typename T1, typename T2, typename T3>
struct node_ranges {
  T1 tokens_gpu;
  T2 token_indices_gpu;
  T3 num_tokens;
  __device__ auto operator()(size_type i) -> thrust::tuple<SymbolOffsetT, SymbolOffsetT>
  {
    // Whether a token expects to be followed by its respective end-of-* token partner
    auto is_begin_of_section = [] __device__(PdaTokenT const token) {
      switch (token) {
        case token_t::StringBegin:
        case token_t::ValueBegin:
        case token_t::FieldNameBegin: return true;
        default: return false;
      };
    };
    // The end-of-* partner token for a given beginning-of-* token
    auto end_of_partner = [] __device__(PdaTokenT const token) {
      switch (token) {
        case token_t::StringBegin: return token_t::StringEnd;
        case token_t::ValueBegin: return token_t::ValueEnd;
        case token_t::FieldNameBegin: return token_t::FieldNameEnd;
        default: return token_t::ErrorBegin;
      };
    };
    auto get_token_index = [] __device__(PdaTokenT const token, SymbolOffsetT const token_index) {
      constexpr SymbolOffsetT skip_quote_char = 1;
      switch (token) {
        case token_t::StringBegin: return token_index + skip_quote_char;
        case token_t::FieldNameBegin: return token_index + skip_quote_char;
        default: return token_index;
      };
    };
    PdaTokenT const token = tokens_gpu[i];
    // The section from the original JSON input that this token demarcates
    SymbolOffsetT range_begin = get_token_index(token, token_indices_gpu[i]);
    SymbolOffsetT range_end   = range_begin + 1;
    if (is_begin_of_section(token)) {
      if ((i + 1) < num_tokens && end_of_partner(token) == tokens_gpu[i + 1]) {
        // Update the range_end for this pair of tokens
        range_end = token_indices_gpu[i + 1];
      }
    }
    return thrust::make_tuple(range_begin, range_end);
  }
};

// GPU version of get_tree_representation
tree_meta_t get_tree_representation_gpu(device_span<SymbolT const> d_input,
                                        rmm::cuda_stream_view stream)
{
  constexpr std::size_t single_item = 1;
  rmm::device_uvector<PdaTokenT> tokens_gpu{d_input.size(), stream};
  rmm::device_uvector<SymbolOffsetT> token_indices_gpu{d_input.size(), stream};
  hostdevice_vector<SymbolOffsetT> num_tokens_out{single_item, stream};

  // Parse the JSON and get the token stream
  cudf::io::json::detail::get_token_stream(
    d_input, tokens_gpu.data(), token_indices_gpu.data(), num_tokens_out.device_ptr(), stream);

  // Copy the JSON token count to the host
  num_tokens_out.device_to_host(stream);

  // Make sure tokens have been copied to the host
  stream.synchronize();

  // Whether a token does represent a node in the tree representation
  auto is_node = [] __device__(PdaTokenT const token) -> size_type {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin:
      case token_t::StringBegin:
      case token_t::ValueBegin:
      case token_t::FieldNameBegin:
      case token_t::ErrorBegin: return 1;
      default: return 0;
    };
  };

  // Whether the token pops from the parent node stack
  auto does_pop = [] __device__(PdaTokenT const token) {
    switch (token) {
      case token_t::StructMemberEnd:
      case token_t::StructEnd:
      case token_t::ListEnd: return true;
      default: return false;
    };
  };

  // Whether the token pushes onto the parent node stack
  auto does_push = [] __device__(PdaTokenT const token) {
    switch (token) {
      // case token_t::StructMemberBegin: //TODO: Either use FieldNameBegin here or change the
      // token_to_node function
      case token_t::FieldNameBegin:
      case token_t::StructBegin:
      case token_t::ListBegin: return true;
      default: return false;
    };
  };

  auto num_tokens = num_tokens_out[0];
  auto is_node_it = thrust::make_transform_iterator(tokens_gpu.begin(), is_node);
  auto num_nodes  = thrust::reduce(rmm::exec_policy(stream), is_node_it, is_node_it + num_tokens);

  // Node categories: copy_if with transform.
  rmm::device_uvector<NodeT> node_categories(num_nodes, stream);
  auto node_categories_it =
    thrust::make_transform_output_iterator(node_categories.begin(), token_to_node{});
  auto node_categories_end = thrust::copy_if(rmm::exec_policy(stream),
                                             tokens_gpu.begin(),
                                             tokens_gpu.begin() + num_tokens,
                                             node_categories_it,
                                             is_node);
  CUDF_EXPECTS(node_categories_end - node_categories_it == num_nodes,
               "node category count mismatch");

  // Node levels: transform_exclusive_scan, copy_if.
  rmm::device_uvector<size_type> token_levels(num_tokens, stream);
  auto push_pop_it = thrust::make_transform_iterator(
    tokens_gpu.begin(), [does_push, does_pop] __device__(PdaTokenT const token) -> size_type {
      return does_push(token) ? 1 : (does_pop(token) ? -1 : 0);
    });
  thrust::exclusive_scan(
    rmm::exec_policy(stream), push_pop_it, push_pop_it + num_tokens, token_levels.begin());

  rmm::device_uvector<TreeDepthT> node_levels(num_nodes, stream);
  auto node_levels_end = thrust::copy_if(rmm::exec_policy(stream),
                                         token_levels.begin(),
                                         token_levels.begin() + num_tokens,
                                         tokens_gpu.begin(),
                                         node_levels.begin(),
                                         is_node);
  CUDF_EXPECTS(node_levels_end - node_levels.begin() == num_nodes, "node level count mismatch");

  // Node ranges: copy_if with transform.
  rmm::device_uvector<SymbolOffsetT> node_range_begin(num_nodes, stream);
  rmm::device_uvector<SymbolOffsetT> node_range_end(num_nodes, stream);
  auto node_range_tuple_it =
    thrust::make_zip_iterator(node_range_begin.begin(), node_range_end.begin());
  using node_ranges_t    = node_ranges<decltype(tokens_gpu.begin()),
                                    decltype(token_indices_gpu.begin()),
                                    decltype(num_tokens)>;
  auto node_range_out_it = thrust::make_transform_output_iterator(
    node_range_tuple_it, node_ranges_t{tokens_gpu.begin(), token_indices_gpu.begin(), num_tokens});

  auto node_range_out_end =
    thrust::copy_if(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(0) + num_tokens,
                    node_range_out_it,
                    [is_node, tokens_gpu = tokens_gpu.begin()] __device__(size_type i) -> bool {
                      PdaTokenT const token = tokens_gpu[i];
                      return is_node(token);
                    });
  CUDF_EXPECTS(node_range_out_end - node_range_out_it == num_nodes, "node range count mismatch");

  // Node parent ids: previous push token_id transform, stable sort, segmented scan with Max,
  // copy_if. This one is sort of logical stack. But more generalized. TODO: make it own function.
  rmm::device_uvector<size_type> parent_token_ids(num_tokens, stream);  // XXX: fill with 0?
  rmm::device_uvector<size_type> initial_order(num_tokens, stream);
  thrust::sequence(rmm::exec_policy(stream), initial_order.begin(), initial_order.end());
  thrust::tabulate(rmm::exec_policy(stream),
                   parent_token_ids.begin(),
                   parent_token_ids.end(),
                   [does_push, tokens_gpu = tokens_gpu.begin()] __device__(auto i) -> size_type {
                     if (i == 0)
                       return -1;
                     else
                       return does_push(tokens_gpu[i - 1]) ? i - 1 : -1;  // XXX: -1 or 0?
                   });
  auto out_pid = thrust::make_zip_iterator(parent_token_ids.data(), initial_order.data());
  // TODO: use radix sort.
  thrust::stable_sort_by_key(rmm::exec_policy(stream),
                             token_levels.data(),
                             token_levels.data() + token_levels.size(),
                             out_pid);
  // SegmentedScan Max.
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                token_levels.data(),
                                token_levels.data() + token_levels.size(),
                                parent_token_ids.data(),
                                parent_token_ids.data(),  // size_type{-1},
                                thrust::equal_to<size_type>{},
                                thrust::maximum<size_type>{});
  // TODO: Avoid sorting again by  gather_if on a transform iterator. or scatter.
  thrust::sort_by_key(rmm::exec_policy(stream),
                      initial_order.data(),
                      initial_order.data() + initial_order.size(),
                      parent_token_ids.data());

  rmm::device_uvector<size_type> node_ids_gpu(num_tokens, stream);
  thrust::exclusive_scan(
    rmm::exec_policy(stream), is_node_it, is_node_it + num_tokens, node_ids_gpu.begin());
  rmm::device_uvector<NodeIndexT> parent_node_ids(num_nodes, stream);
  auto parent_node_ids_it = thrust::make_transform_iterator(
    parent_token_ids.begin(),
    [node_ids_gpu = node_ids_gpu.begin()] __device__(size_type const pid) -> NodeIndexT {
      return pid < 0 ? pid : node_ids_gpu[pid];
    });
  auto parent_node_ids_end = thrust::copy_if(rmm::exec_policy(stream),
                                             parent_node_ids_it,
                                             parent_node_ids_it + parent_token_ids.size(),
                                             tokens_gpu.begin(),
                                             parent_node_ids.begin(),
                                             is_node);
  CUDF_EXPECTS(parent_node_ids_end - parent_node_ids.begin() == num_nodes,
               "parent node id gather mismatch");

  return {cudf::detail::make_std_vector_async(node_categories, stream),
          cudf::detail::make_std_vector_async(parent_node_ids, stream),
          cudf::detail::make_std_vector_async(node_levels, stream),
          cudf::detail::make_std_vector_async(node_range_begin, stream),
          cudf::detail::make_std_vector_async(node_range_end, stream)};
}
}  // namespace detail
}  // namespace cudf::io::json
