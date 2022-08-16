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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <stack>

// Debug print flag
#ifndef NJP_DEBUG_PRINT
//#define NJP_DEBUG_PRINT
#endif

namespace {

/**
 * @brief While parsing the token stream, we use a stack of tree_nodes to maintain all the
 * information about the data path that is relevant.
 */
struct tree_node {
  // The column that this node is associated with
  cudf::io::json::json_column* column;

  // The row offset that this node belongs to within the given column
  uint32_t row_index;

  // Selected child column
  // E.g., if this is a struct node, and we subsequently encountered the field name "a", then this
  // point's to the struct's "a" child column
  cudf::io::json::json_column* current_selected_col = nullptr;

  std::size_t num_children = 0;
};
}  // namespace

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

/**
 * @brief Parses the given JSON string and generates a tree representation of the given input.
 *
 * @param[in,out] root_column The root column of the hierarchy of columns into which data is parsed
 * @param[in,out] current_data_path The stack represents the path from the JSON root node to the
 * first node encountered in \p input
 * @param[in] input The JSON input in host memory
 * @param[in] d_input The JSON input in device memory
 * @param[in] stream The CUDA stream to which kernels are dispatched
 * @return The columnar representation of the data from the given JSON input
 */
void make_json_column(json_column& root_column,
                      std::stack<tree_node>& current_data_path,
                      host_span<SymbolT const> input,
                      device_span<SymbolT const> d_input,
                      rmm::cuda_stream_view stream)
{
  // Default name for a list's child column
  std::string const list_child_name = "element";

  constexpr std::size_t single_item = 1;
  hostdevice_vector<PdaTokenT> tokens_gpu{input.size(), stream};
  hostdevice_vector<SymbolOffsetT> token_indices_gpu{input.size(), stream};
  hostdevice_vector<SymbolOffsetT> num_tokens_out{single_item, stream};

  // Parse the JSON and get the token stream
  get_token_stream(d_input,
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

  // Whether this token is the valid token to begin the JSON document with
  auto is_valid_root_token = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin:
      case token_t::StringBegin:
      case token_t::ValueBegin: return true;
      default: return false;
    };
  };

  // Returns the token's corresponding column type
  auto token_to_column_type = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructBegin: return json_col_t::StructColumn;
      case token_t::ListBegin: return json_col_t::ListColumn;
      case token_t::StringBegin: return json_col_t::StringColumn;
      case token_t::ValueBegin: return json_col_t::StringColumn;
      default: return json_col_t::Unknown;
    };
  };

  // Whether this token is a beginning-of-list or beginning-of-struct token
  auto is_nested_token = [](PdaTokenT const token) {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin: return true;
      default: return false;
    };
  };

  // Skips the quote char if the token is a beginning-of-string or beginning-of-field-name token
  auto get_token_index = [](PdaTokenT const token, SymbolOffsetT const token_index) {
    constexpr SymbolOffsetT skip_quote_char = 1;
    switch (token) {
      case token_t::StringBegin: return token_index + skip_quote_char;
      case token_t::FieldNameBegin: return token_index + skip_quote_char;
      default: return token_index;
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

#ifdef NJP_DEBUG_PRINT
  auto column_type_string = [](json_col_t column_type) {
    switch (column_type) {
      case json_col_t::Unknown: return "Unknown";
      case json_col_t::ListColumn: return "List";
      case json_col_t::StructColumn: return "Struct";
      case json_col_t::StringColumn: return "String";
      default: return "Unknown";
    }
  };

  auto token_to_string = [](PdaTokenT token_type) {
    switch (token_type) {
      case token_t::StructBegin: return "StructBegin";
      case token_t::StructEnd: return "StructEnd";
      case token_t::ListBegin: return "ListBegin";
      case token_t::ListEnd: return "ListEnd";
      case token_t::FieldNameBegin: return "FieldNameBegin";
      case token_t::FieldNameEnd: return "FieldNameEnd";
      case token_t::StringBegin: return "StringBegin";
      case token_t::StringEnd: return "StringEnd";
      case token_t::ValueBegin: return "ValueBegin";
      case token_t::ValueEnd: return "ValueEnd";
      case token_t::ErrorBegin: return "ErrorBegin";
      default: return "Unknown";
    }
  };
#endif

  /**
   * @brief Updates the given row in the given column with a new string_end and child_count. In
   * particular, updating the child count is relevant for list columns.
   */
  auto update_row =
    [](json_column* column, uint32_t row_index, uint32_t string_end, uint32_t child_count) {
#ifdef NJP_DEBUG_PRINT
      std::cout << "  -> update_row()\n";
      std::cout << "  ---> col@" << column << "\n";
      std::cout << "  ---> row #" << row_index << "\n";
      std::cout << "  ---> string_lengths = " << (string_end - column->string_offsets[row_index])
                << "\n";
      std::cout << "  ---> child_offsets = " << (column->child_offsets[row_index + 1] + child_count)
                << "\n";
#endif
      column->string_lengths[row_index]    = column->child_offsets[row_index + 1] + child_count;
      column->child_offsets[row_index + 1] = column->child_offsets[row_index + 1] + child_count;
    };

  /**
   * @brief Gets the currently selected child column given a \p current_data_path.
   *
   * That is, if \p current_data_path top-of-stack is
   * (a) a struct, the selected child column corresponds to the child column of the last field name
   * node encountered.
   * (b) a list, the selected child column corresponds to single child column of
   * the list column. In this case, the child column may not exist yet.
   */
  auto get_selected_column = [&list_child_name](std::stack<tree_node>& current_data_path) {
    json_column* selected_col = current_data_path.top().current_selected_col;

    // If the node does not have a selected column yet
    if (selected_col == nullptr) {
      // We're looking at the child column of a list column
      if (current_data_path.top().column->type == json_col_t::ListColumn) {
        CUDF_EXPECTS(current_data_path.top().column->child_columns.size() <= 1,
                     "Encountered a list column with more than a single child column");
        // The child column has yet to be created
        if (current_data_path.top().column->child_columns.size() == 0) {
          current_data_path.top().column->child_columns.emplace(std::string{list_child_name},
                                                                json_column{json_col_t::Unknown});
        }
        current_data_path.top().current_selected_col =
          &current_data_path.top().column->child_columns.begin()->second;
        selected_col = current_data_path.top().current_selected_col;
      } else {
        CUDF_FAIL("Trying to retrieve child column without encountering a field name.");
      }
    }
#ifdef NJP_DEBUG_PRINT
    std::cout << "  -> get_selected_column()\n";
    std::cout << "  ---> selected col@" << selected_col << "\n";
#endif
    return selected_col;
  };

  /**
   * @brief Returns a pointer to the child column with the given \p field_name within the current
   * struct column.
   */
  auto select_column = [](std::stack<tree_node>& current_data_path, std::string const& field_name) {
#ifdef NJP_DEBUG_PRINT
    std::cout << "  -> select_column(" << field_name << ")\n";
#endif
    // The field name's parent struct node
    auto& current_struct_node = current_data_path.top();

    // Verify that the field name node is actually a child of a struct
    CUDF_EXPECTS(current_data_path.top().column->type == json_col_t::StructColumn,
                 "Invalid JSON token sequence");

    json_column* struct_col  = current_struct_node.column;
    auto const& child_col_it = struct_col->child_columns.find(field_name);

    // The field name's column exists already, select that as the struct node's currently selected
    // child column
    if (child_col_it != struct_col->child_columns.end()) { return &child_col_it->second; }

    // The field name's column does not exist yet, so we have to append the child column to the
    // struct column
    return &struct_col->child_columns.emplace(field_name, json_column{}).first->second;
  };

  /**
   * @brief Gets the row offset at which to insert. I.e., for a child column of a list column, we
   * just have to append the row to the end. Otherwise we have to propagate the row offset from the
   * parent struct column.
   */
  auto get_target_row_index = [](std::stack<tree_node> const& current_data_path,
                                 json_column* target_column) {
#ifdef NJP_DEBUG_PRINT
    std::cout << " -> target row: "
              << ((current_data_path.top().column->type == json_col_t::ListColumn)
                    ? target_column->current_offset
                    : current_data_path.top().row_index)
              << "\n";
#endif
    return (current_data_path.top().column->type == json_col_t::ListColumn)
             ? target_column->current_offset
             : current_data_path.top().row_index;
  };

  // The offset of the token currently being processed
  std::size_t offset = 0;

  // Giving names to magic constants
  constexpr uint32_t row_offset_zero  = 0;
  constexpr uint32_t zero_child_count = 0;

  //--------------------------------------------------------------------------------
  // INITIALIZE JSON ROOT NODE
  //--------------------------------------------------------------------------------
  // The JSON root may only be a struct, list, string, or value node
  CUDF_EXPECTS(num_tokens_out[0] > 0, "Empty JSON input not supported");
  CUDF_EXPECTS(is_valid_root_token(tokens_gpu[offset]), "Invalid beginning of JSON document");

  // The JSON root is either a struct or list
  if (is_nested_token(tokens_gpu[offset])) {
    // Initialize the root column and append this row to it
    root_column.append_row(row_offset_zero,
                           token_to_column_type(tokens_gpu[offset]),
                           get_token_index(tokens_gpu[offset], token_indices_gpu[offset]),
                           get_token_index(tokens_gpu[offset], token_indices_gpu[offset]),
                           0);

    // Push the root node onto the stack for the data path
    current_data_path.push({&root_column, row_offset_zero, nullptr, zero_child_count});

    // Continue with the next token from the token stream
    offset++;
  }
  // The JSON is a simple scalar value -> create simple table and return
  else {
    constexpr SymbolOffsetT max_tokens_for_scalar_value = 2;
    CUDF_EXPECTS(num_tokens_out[0] <= max_tokens_for_scalar_value,
                 "Invalid JSON format. Expected just a scalar value.");

    // If this isn't the only token, verify the subsequent token is the correct end-of-* partner
    if ((offset + 1) < num_tokens_out[0]) {
      CUDF_EXPECTS(tokens_gpu[offset + 1] == end_of_partner(tokens_gpu[offset]),
                   "Invalid JSON token sequence");
    }

    // The offset to the first symbol from the JSON input associated with the current token
    auto const& token_begin_offset = get_token_index(tokens_gpu[offset], token_indices_gpu[offset]);

    // The offset to one past the last symbol associated with the current token
    // Literals without trailing space are missing the corresponding end-of-* counterpart.
    auto const& token_end_offset =
      (offset + 1 < num_tokens_out[0])
        ? get_token_index(tokens_gpu[offset + 1], token_indices_gpu[offset + 1])
        : input.size();

    root_column.append_row(row_offset_zero,
                           json_col_t::StringColumn,
                           token_begin_offset,
                           token_end_offset,
                           zero_child_count);
    return;
  }

  while (offset < num_tokens_out[0]) {
    // Verify there's at least the JSON root node left on the stack to which we can append data
    CUDF_EXPECTS(current_data_path.size() > 0, "Invalid JSON structure");

    // Verify that the current node in the tree (which becomes this nodes parent) can have children
    CUDF_EXPECTS(current_data_path.top().column->type == json_col_t::ListColumn or
                   current_data_path.top().column->type == json_col_t::StructColumn,
                 "Invalid JSON structure");

    // The token we're currently parsing
    auto const& token = tokens_gpu[offset];

#ifdef NJP_DEBUG_PRINT
    std::cout << "[" << token_to_string(token) << "]\n";
#endif

    // StructBegin token
    if (token == token_t::StructBegin) {
      // Get this node's column. That is, the parent node's selected column:
      // (a) if parent is a list, then this will (create and) return the list's only child column
      // (b) if parent is a struct, then this will return the column selected by the last field name
      // encountered.
      json_column* selected_col = get_selected_column(current_data_path);

      // Get the row offset at which to insert
      auto const target_row_index = get_target_row_index(current_data_path, selected_col);

      // Increment parent's child count and insert this struct node into the data path
      current_data_path.top().num_children++;
      current_data_path.push({selected_col, target_row_index, nullptr, zero_child_count});

      // Add this struct node to the current column
      selected_col->append_row(target_row_index,
                               token_to_column_type(tokens_gpu[offset]),
                               get_token_index(tokens_gpu[offset], token_indices_gpu[offset]),
                               get_token_index(tokens_gpu[offset], token_indices_gpu[offset]),
                               zero_child_count);
    }

    // StructEnd token
    else if (token == token_t::StructEnd) {
      // Verify that this node in fact a struct node (i.e., it was part of a struct column)
      CUDF_EXPECTS(current_data_path.top().column->type == json_col_t::StructColumn,
                   "Broken invariant while parsing JSON");
      CUDF_EXPECTS(current_data_path.top().column != nullptr,
                   "Broken invariant while parsing JSON");

      // Update row to account for string offset
      update_row(current_data_path.top().column,
                 current_data_path.top().row_index,
                 get_token_index(tokens_gpu[offset], token_indices_gpu[offset]),
                 current_data_path.top().num_children);

      // Pop struct from the path stack
      current_data_path.pop();
    }

    // ListBegin token
    else if (token == token_t::ListBegin) {
      // Get the selected column
      json_column* selected_col = get_selected_column(current_data_path);

      // Get the row offset at which to insert
      auto const target_row_index = get_target_row_index(current_data_path, selected_col);

      // Increment parent's child count and insert this struct node into the data path
      current_data_path.top().num_children++;
      current_data_path.push({selected_col, target_row_index, nullptr, zero_child_count});

      // Add this struct node to the current column
      selected_col->append_row(target_row_index,
                               token_to_column_type(tokens_gpu[offset]),
                               get_token_index(tokens_gpu[offset], token_indices_gpu[offset]),
                               get_token_index(tokens_gpu[offset], token_indices_gpu[offset]),
                               zero_child_count);
    }

    // ListEnd token
    else if (token == token_t::ListEnd) {
      // Verify that this node in fact a list node (i.e., it was part of a list column)
      CUDF_EXPECTS(current_data_path.top().column->type == json_col_t::ListColumn,
                   "Broken invariant while parsing JSON");
      CUDF_EXPECTS(current_data_path.top().column != nullptr,
                   "Broken invariant while parsing JSON");

      // Update row to account for string offset
      update_row(current_data_path.top().column,
                 current_data_path.top().row_index,
                 get_token_index(tokens_gpu[offset], token_indices_gpu[offset]),
                 current_data_path.top().num_children);

      // Pop list from the path stack
      current_data_path.pop();
    }

    // Error token
    else if (token == token_t::ErrorBegin) {
#ifdef NJP_DEBUG_PRINT
      std::cout << "[ErrorBegin]\n";
#endif
      CUDF_FAIL("Parser encountered an invalid format.");
    }

    // FieldName, String, or Value (begin, end)-pair
    else if (token == token_t::FieldNameBegin or token == token_t::StringBegin or
             token == token_t::ValueBegin) {
      // Verify that this token has the right successor to build a correct (being, end) token pair
      CUDF_EXPECTS((offset + 1) < num_tokens_out[0], "Invalid JSON token sequence");
      CUDF_EXPECTS(tokens_gpu[offset + 1] == end_of_partner(token), "Invalid JSON token sequence");

      // The offset to the first symbol from the JSON input associated with the current token
      auto const& token_begin_offset =
        get_token_index(tokens_gpu[offset], token_indices_gpu[offset]);

      // The offset to one past the last symbol associated with the current token
      auto const& token_end_offset =
        get_token_index(tokens_gpu[offset + 1], token_indices_gpu[offset + 1]);

      // FieldNameBegin
      // For the current struct node in the tree, select the child column corresponding to this
      // field name
      if (token == token_t::FieldNameBegin) {
        std::string field_name{input.data() + token_begin_offset,
                               (token_end_offset - token_begin_offset)};
        current_data_path.top().current_selected_col = select_column(current_data_path, field_name);
      }
      // StringBegin
      // ValueBegin
      // As we currently parse to string columns there's no further differentiation
      else if (token == token_t::StringBegin or token == token_t::ValueBegin) {
        // Get the selected column
        json_column* selected_col = get_selected_column(current_data_path);

        // Get the row offset at which to insert
        auto const target_row_index = get_target_row_index(current_data_path, selected_col);

        current_data_path.top().num_children++;

        selected_col->append_row(target_row_index,
                                 token_to_column_type(token),
                                 token_begin_offset,
                                 token_end_offset,
                                 zero_child_count);
      } else {
        CUDF_FAIL("Unknown JSON token");
      }

      // As we've also consumed the end-of-* token, we advance the processed token offset by one
      offset++;
    }

    offset++;
  }

  // Make sure all of a struct's child columns have the same length
  root_column.level_child_cols_recursively(root_column.current_offset);
}

std::pair<std::unique_ptr<column>, std::vector<column_name_info>> json_column_to_cudf_column(
  json_column const& json_col,
  device_span<SymbolT const> d_input,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto make_validity =
    [stream, mr](json_column const& json_col) -> std::pair<rmm::device_buffer, size_type> {
    if (json_col.current_offset == json_col.valid_count) { return {rmm::device_buffer{}, 0}; }

    return {rmm::device_buffer{json_col.validity.data(),
                               bitmask_allocation_size_bytes(json_col.current_offset),
                               stream,
                               mr},
            json_col.current_offset - json_col.valid_count};
  };

  switch (json_col.type) {
    case json_col_t::StringColumn: {
      // move string_offsets to GPU and transform to string column
      auto const col_size      = json_col.string_offsets.size();
      using char_length_pair_t = thrust::pair<const char*, size_type>;
      CUDF_EXPECTS(json_col.string_offsets.size() == json_col.string_lengths.size(),
                   "string offset, string length mismatch");
      rmm::device_uvector<char_length_pair_t> d_string_data(col_size, stream);
      rmm::device_uvector<json_column::row_offset_t> d_string_offsets =
        cudf::detail::make_device_uvector_async(json_col.string_offsets, stream);
      rmm::device_uvector<json_column::row_offset_t> d_string_lengths =
        cudf::detail::make_device_uvector_async(json_col.string_lengths, stream);
      auto offset_length_it =
        thrust::make_zip_iterator(d_string_offsets.begin(), d_string_lengths.begin());
      thrust::transform(rmm::exec_policy(stream),
                        offset_length_it,
                        offset_length_it + col_size,
                        d_string_data.data(),
                        [data = d_input.data()] __device__(auto ip) {
                          return char_length_pair_t{data + thrust::get<0>(ip), thrust::get<1>(ip)};
                        });
      auto str_col_ptr                  = make_strings_column(d_string_data, stream, mr);
      auto [result_bitmask, null_count] = make_validity(json_col);
      str_col_ptr->set_null_mask(result_bitmask, null_count);
      return {std::move(str_col_ptr), {{"offsets"}, {"chars"}}};
      break;
    }
    case json_col_t::StructColumn: {
      std::vector<std::unique_ptr<column>> child_columns;
      std::vector<column_name_info> column_names{};
      size_type num_rows{json_col.current_offset};
      // Create children columns
      for (auto const& col : json_col.child_columns) {
        column_names.emplace_back(col.first);
        auto const& child_col      = col.second;
        auto [child_column, names] = json_column_to_cudf_column(child_col, d_input, stream, mr);
        CUDF_EXPECTS(num_rows == child_column->size(),
                     "All children columns must have the same size");
        child_columns.push_back(std::move(child_column));
        column_names.back().children = names;
      }
      auto [result_bitmask, null_count] = make_validity(json_col);
      return {
        make_structs_column(
          num_rows, std::move(child_columns), null_count, std::move(result_bitmask), stream, mr),
        column_names};
      break;
    }
    case json_col_t::ListColumn: {
      size_type num_rows = json_col.child_offsets.size();
      std::vector<column_name_info> column_names{};
      column_names.emplace_back("offsets");
      column_names.emplace_back(json_col.child_columns.begin()->first);

      rmm::device_uvector<json_column::row_offset_t> d_offsets =
        cudf::detail::make_device_uvector_async(json_col.child_offsets, stream, mr);
      auto offsets_column =
        std::make_unique<column>(data_type{type_id::INT32}, num_rows, d_offsets.release());
      // Create children column
      auto [child_column, names] =
        json_column_to_cudf_column(json_col.child_columns.begin()->second, d_input, stream, mr);
      column_names.back().children      = names;
      auto [result_bitmask, null_count] = make_validity(json_col);
      return {make_lists_column(num_rows - 1,
                                std::move(offsets_column),
                                std::move(child_column),
                                null_count,
                                std::move(result_bitmask),
                                stream,
                                mr),
              std::move(column_names)};
      break;
    }
    default: CUDF_FAIL("Unsupported column type, yet to be implemented"); break;
  }

  return {};
}

table_with_metadata parse_nested_json(host_span<SymbolT const> input,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  // Allocate device memory for the JSON input & copy over to device
  rmm::device_uvector<SymbolT> d_input = cudf::detail::make_device_uvector_async(input, stream);

  // Get internal JSON column
  json_column root_column{};
  std::stack<tree_node> data_path{};
  make_json_column(root_column, data_path, input, d_input, stream);

  // Verify that we were in fact given a list of structs (or in JSON speech: an array of objects)
  auto constexpr single_child_col_count = 1;
  CUDF_EXPECTS(root_column.type == json_col_t::ListColumn and
                 root_column.child_columns.size() == single_child_col_count and
                 root_column.child_columns.begin()->second.type == json_col_t::StructColumn,
               "Currently the nested JSON parser only supports an array of (nested) objects");

  // Slice off the root list column, which has only a single row that contains all the structs
  auto const& root_struct_col = root_column.child_columns.begin()->second;

  // Initialize meta data to be populated while recursing through the tree of columns
  std::vector<std::unique_ptr<column>> out_columns;
  std::vector<column_name_info> out_column_names;

  // Iterate over the struct's child columns and convert to cudf column
  for (auto const& [col_name, json_col] : root_struct_col.child_columns) {
    // Insert this columns name into the schema
    out_column_names.emplace_back(col_name);

    // Get this JSON column's cudf column and schema info
    auto [cudf_col, col_name_info]   = json_column_to_cudf_column(json_col, d_input, stream, mr);
    out_column_names.back().children = std::move(col_name_info);
    out_columns.emplace_back(std::move(cudf_col));
  }

  return table_with_metadata{std::make_unique<table>(std::move(out_columns)),
                             {{}, out_column_names}};
}

}  // namespace detail
}  // namespace cudf::io::json

// Debug print flag
#undef NJP_DEBUG_PRINT
