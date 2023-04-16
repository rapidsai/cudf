/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <io/utilities/parsing_utils.cuh>
#include <io/utilities/type_inference.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/utilities/visitor_overload.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/io/detail/data_casting.cuh>
#include <cudf/io/json.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <limits>
#include <stack>

// Debug print flag
#ifndef NJP_DEBUG_PRINT
// #define NJP_DEBUG_PRINT
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

/**
 * @brief Verifies that the JSON input can be handled without corrupted data due to offset
 * overflows.
 *
 * @param input_size The JSON inputs size in bytes
 */
void check_input_size(std::size_t input_size)
{
  // Transduce() writes symbol offsets that may be as large input_size-1
  CUDF_EXPECTS(input_size == 0 ||
                 (input_size - 1) <= std::numeric_limits<cudf::io::json::SymbolOffsetT>::max(),
               "Given JSON input is too large");
}
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
   /* TT_OOS    */ {{{'{'}, {'['}, {'}'}, {']'}, {}, {}, {}}},
   /* TT_STR    */ {{{}, {}, {}, {}, {}, {}, {}}},
   /* TT_ESC    */ {{{}, {}, {}, {}, {}, {}, {}}}}};

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
  /// Linebreak
  LINE_BREAK,
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
  static_cast<PdaSymbolGroupIdT>(symbol_group_id::LINE_BREAK),
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

/**
 * @brief Getting the transition table
 */
auto get_transition_table(bool newline_delimited_json)
{
  static_assert(static_cast<PdaStackSymbolGroupIdT>(stack_symbol_group_id::STACK_ROOT) == 0);
  static_assert(static_cast<PdaStackSymbolGroupIdT>(stack_symbol_group_id::STACK_LIST) == 1);
  static_assert(static_cast<PdaStackSymbolGroupIdT>(stack_symbol_group_id::STACK_STRUCT) == 2);

  // In case of newline-delimited JSON, multiple newlines are ignored, similar to whitespace.
  // Thas is, empty lines are ignored
  auto const PD_ANL = newline_delimited_json ? PD_BOV : PD_PVL;
  std::array<std::array<pda_state_t, NUM_PDA_SGIDS>, PD_NUM_STATES> pda_tt;
  //  {       [       }       ]       "       \       ,       :     space   newline other
  pda_tt[static_cast<StateT>(pda_state_t::PD_BOV)] = {
    PD_BOA, PD_BOA, PD_ERR, PD_ERR, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_BOV, PD_LON,
    PD_BOA, PD_BOA, PD_ERR, PD_PVL, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_BOV, PD_LON,
    PD_BOA, PD_BOA, PD_ERR, PD_ERR, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_BOV, PD_LON};
  pda_tt[static_cast<StateT>(pda_state_t::PD_BOA)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_BOA, PD_BOA, PD_ERR, PD_PVL, PD_STR, PD_ERR, PD_ERR, PD_ERR, PD_BOA, PD_BOA, PD_LON,
    PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_FLN, PD_ERR, PD_ERR, PD_ERR, PD_BOA, PD_BOA, PD_ERR};
  pda_tt[static_cast<StateT>(pda_state_t::PD_LON)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_PVL, PD_LON,
    PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_BOV, PD_ERR, PD_PVL, PD_PVL, PD_LON,
    PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_ERR, PD_BFN, PD_ERR, PD_PVL, PD_PVL, PD_LON};
  pda_tt[static_cast<StateT>(pda_state_t::PD_STR)] = {
    PD_STR, PD_STR, PD_STR, PD_STR, PD_PVL, PD_SCE, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR,
    PD_STR, PD_STR, PD_STR, PD_STR, PD_PVL, PD_SCE, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR,
    PD_STR, PD_STR, PD_STR, PD_STR, PD_PVL, PD_SCE, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR};
  pda_tt[static_cast<StateT>(pda_state_t::PD_SCE)] = {
    PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR,
    PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR,
    PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR, PD_STR};
  pda_tt[static_cast<StateT>(pda_state_t::PD_PVL)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_ANL, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_BOV, PD_ERR, PD_PVL, PD_PVL, PD_ERR,
    PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_ERR, PD_ERR, PD_BFN, PD_ERR, PD_PVL, PD_PVL, PD_ERR};
  pda_tt[static_cast<StateT>(pda_state_t::PD_BFN)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_PVL, PD_ERR, PD_FLN, PD_ERR, PD_ERR, PD_ERR, PD_BFN, PD_BFN, PD_ERR};
  pda_tt[static_cast<StateT>(pda_state_t::PD_FLN)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_PFN, PD_FNE, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN};
  pda_tt[static_cast<StateT>(pda_state_t::PD_FNE)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN, PD_FLN};
  pda_tt[static_cast<StateT>(pda_state_t::PD_PFN)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_BOV, PD_PFN, PD_PFN, PD_ERR};
  pda_tt[static_cast<StateT>(pda_state_t::PD_ERR)] = {
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR,
    PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR, PD_ERR};
  return pda_tt;
}

/**
 * @brief Getting the translation table
 */
auto get_translation_table()
{
  constexpr auto StructBegin       = token_t::StructBegin;
  constexpr auto StructEnd         = token_t::StructEnd;
  constexpr auto ListBegin         = token_t::ListBegin;
  constexpr auto ListEnd           = token_t::ListEnd;
  constexpr auto StructMemberBegin = token_t::StructMemberBegin;
  constexpr auto StructMemberEnd   = token_t::StructMemberEnd;
  constexpr auto FieldNameBegin    = token_t::FieldNameBegin;
  constexpr auto FieldNameEnd      = token_t::FieldNameEnd;
  constexpr auto StringBegin       = token_t::StringBegin;
  constexpr auto StringEnd         = token_t::StringEnd;
  constexpr auto ValueBegin        = token_t::ValueBegin;
  constexpr auto ValueEnd          = token_t::ValueEnd;
  constexpr auto ErrorBegin        = token_t::ErrorBegin;

  std::array<std::array<std::vector<char>, NUM_PDA_SGIDS>, PD_NUM_STATES> pda_tlt;
  pda_tlt[static_cast<StateT>(pda_state_t::PD_BOV)] = {{                /*ROOT*/
                                                        {StructBegin},  // OPENING_BRACE
                                                        {ListBegin},    // OPENING_BRACKET
                                                        {ErrorBegin},   // CLOSING_BRACE
                                                        {ErrorBegin},   // CLOSING_BRACKET
                                                        {StringBegin},  // QUOTE
                                                        {ErrorBegin},   // ESCAPE
                                                        {ErrorBegin},   // COMMA
                                                        {ErrorBegin},   // COLON
                                                        {},             // WHITE_SPACE
                                                        {},             // LINE_BREAK
                                                        {ValueBegin},   // OTHER
                                                        /*LIST*/
                                                        {StructBegin},  // OPENING_BRACE
                                                        {ListBegin},    // OPENING_BRACKET
                                                        {ErrorBegin},   // CLOSING_BRACE
                                                        {ListEnd},      // CLOSING_BRACKET
                                                        {StringBegin},  // QUOTE
                                                        {ErrorBegin},   // ESCAPE
                                                        {ErrorBegin},   // COMMA
                                                        {ErrorBegin},   // COLON
                                                        {},             // WHITE_SPACE
                                                        {},             // LINE_BREAK
                                                        {ValueBegin},   // OTHER
                                                        /*STRUCT*/
                                                        {StructBegin},   // OPENING_BRACE
                                                        {ListBegin},     // OPENING_BRACKET
                                                        {ErrorBegin},    // CLOSING_BRACE
                                                        {ErrorBegin},    // CLOSING_BRACKET
                                                        {StringBegin},   // QUOTE
                                                        {ErrorBegin},    // ESCAPE
                                                        {ErrorBegin},    // COMMA
                                                        {ErrorBegin},    // COLON
                                                        {},              // WHITE_SPACE
                                                        {},              // LINE_BREAK
                                                        {ValueBegin}}};  // OTHER
  pda_tlt[static_cast<StateT>(pda_state_t::PD_BOA)] = {
    {                                                                    /*ROOT*/
     {ErrorBegin},                                                       // OPENING_BRACE
     {ErrorBegin},                                                       // OPENING_BRACKET
     {ErrorBegin},                                                       // CLOSING_BRACE
     {ErrorBegin},                                                       // CLOSING_BRACKET
     {ErrorBegin},                                                       // QUOTE
     {ErrorBegin},                                                       // ESCAPE
     {ErrorBegin},                                                       // COMMA
     {ErrorBegin},                                                       // COLON
     {ErrorBegin},                                                       // WHITE_SPACE
     {ErrorBegin},                                                       // LINE_BREAK
     {ErrorBegin},                                                       // OTHER
     /*LIST*/
     {StructBegin},  // OPENING_BRACE
     {ListBegin},    // OPENING_BRACKET
     {ErrorBegin},   // CLOSING_BRACE
     {ListEnd},      // CLOSING_BRACKET
     {StringBegin},  // QUOTE
     {ErrorBegin},   // ESCAPE
     {ErrorBegin},   // COMMA
     {ErrorBegin},   // COLON
     {},             // WHITE_SPACE
     {},             // LINE_BREAK
     {ValueBegin},   // OTHER
     /*STRUCT*/
     {ErrorBegin},                         // OPENING_BRACE
     {ErrorBegin},                         // OPENING_BRACKET
     {StructEnd},                          // CLOSING_BRACE
     {ErrorBegin},                         // CLOSING_BRACKET
     {StructMemberBegin, FieldNameBegin},  // QUOTE
     {ErrorBegin},                         // ESCAPE
     {ErrorBegin},                         // COMMA
     {ErrorBegin},                         // COLON
     {},                                   // WHITE_SPACE
     {},                                   // LINE_BREAK
     {ErrorBegin}}};                       // OTHER
  pda_tlt[static_cast<StateT>(pda_state_t::PD_LON)] = {
    {                                      /*ROOT*/
     {ErrorBegin},                         // OPENING_BRACE
     {ErrorBegin},                         // OPENING_BRACKET
     {ErrorBegin},                         // CLOSING_BRACE
     {ErrorBegin},                         // CLOSING_BRACKET
     {ErrorBegin},                         // QUOTE
     {ErrorBegin},                         // ESCAPE
     {ErrorBegin},                         // COMMA
     {ErrorBegin},                         // COLON
     {ValueEnd},                           // WHITE_SPACE
     {ValueEnd},                           // LINE_BREAK
     {},                                   // OTHER
     /*LIST*/
     {ErrorBegin},         // OPENING_BRACE
     {ErrorBegin},         // OPENING_BRACKET
     {ErrorBegin},         // CLOSING_BRACE
     {ValueEnd, ListEnd},  // CLOSING_BRACKET
     {ErrorBegin},         // QUOTE
     {ErrorBegin},         // ESCAPE
     {ValueEnd},           // COMMA
     {ErrorBegin},         // COLON
     {ValueEnd},           // WHITE_SPACE
     {ValueEnd},           // LINE_BREAK
     {},                   // OTHER
     /*STRUCT*/
     {ErrorBegin},                                                    // OPENING_BRACE
     {ErrorBegin},                                                    // OPENING_BRACKET
     {ValueEnd, StructMemberEnd, StructEnd},                          // CLOSING_BRACE
     {ErrorBegin},                                                    // CLOSING_BRACKET
     {ErrorBegin},                                                    // QUOTE
     {ErrorBegin},                                                    // ESCAPE
     {ValueEnd, StructMemberEnd},                                     // COMMA
     {ErrorBegin},                                                    // COLON
     {ValueEnd},                                                      // WHITE_SPACE
     {ValueEnd},                                                      // LINE_BREAK
     {}}};                                                            // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_STR)] = {{              /*ROOT*/
                                                        {},           // OPENING_BRACE
                                                        {},           // OPENING_BRACKET
                                                        {},           // CLOSING_BRACE
                                                        {},           // CLOSING_BRACKET
                                                        {StringEnd},  // QUOTE
                                                        {},           // ESCAPE
                                                        {},           // COMMA
                                                        {},           // COLON
                                                        {},           // WHITE_SPACE
                                                        {},           // LINE_BREAK
                                                        {},           // OTHER
                                                        /*LIST*/
                                                        {},           // OPENING_BRACE
                                                        {},           // OPENING_BRACKET
                                                        {},           // CLOSING_BRACE
                                                        {},           // CLOSING_BRACKET
                                                        {StringEnd},  // QUOTE
                                                        {},           // ESCAPE
                                                        {},           // COMMA
                                                        {},           // COLON
                                                        {},           // WHITE_SPACE
                                                        {},           // LINE_BREAK
                                                        {},           // OTHER
                                                        /*STRUCT*/
                                                        {},           // OPENING_BRACE
                                                        {},           // OPENING_BRACKET
                                                        {},           // CLOSING_BRACE
                                                        {},           // CLOSING_BRACKET
                                                        {StringEnd},  // QUOTE
                                                        {},           // ESCAPE
                                                        {},           // COMMA
                                                        {},           // COLON
                                                        {},           // WHITE_SPACE
                                                        {},           // LINE_BREAK
                                                        {}}};         // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_SCE)] = {{              /*ROOT*/
                                                        {},           // OPENING_BRACE
                                                        {},           // OPENING_BRACKET
                                                        {},           // CLOSING_BRACE
                                                        {},           // CLOSING_BRACKET
                                                        {},           // QUOTE
                                                        {},           // ESCAPE
                                                        {},           // COMMA
                                                        {},           // COLON
                                                        {},           // WHITE_SPACE
                                                        {},           // LINE_BREAK
                                                        {},           // OTHER
                                                        /*LIST*/
                                                        {},  // OPENING_BRACE
                                                        {},  // OPENING_BRACKET
                                                        {},  // CLOSING_BRACE
                                                        {},  // CLOSING_BRACKET
                                                        {},  // QUOTE
                                                        {},  // ESCAPE
                                                        {},  // COMMA
                                                        {},  // COLON
                                                        {},  // WHITE_SPACE
                                                        {},  // LINE_BREAK
                                                        {},  // OTHER
                                                        /*STRUCT*/
                                                        {},    // OPENING_BRACE
                                                        {},    // OPENING_BRACKET
                                                        {},    // CLOSING_BRACE
                                                        {},    // CLOSING_BRACKET
                                                        {},    // QUOTE
                                                        {},    // ESCAPE
                                                        {},    // COMMA
                                                        {},    // COLON
                                                        {},    // WHITE_SPACE
                                                        {},    // LINE_BREAK
                                                        {}}};  // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_PVL)] = {
    {               /*ROOT*/
     {ErrorBegin},  // OPENING_BRACE
     {ErrorBegin},  // OPENING_BRACKET
     {ErrorBegin},  // CLOSING_BRACE
     {ErrorBegin},  // CLOSING_BRACKET
     {ErrorBegin},  // QUOTE
     {ErrorBegin},  // ESCAPE
     {ErrorBegin},  // COMMA
     {ErrorBegin},  // COLON
     {},            // WHITE_SPACE
     {},            // LINE_BREAK
     {ErrorBegin},  // OTHER
     /*LIST*/
     {ErrorBegin},  // OPENING_BRACE
     {ErrorBegin},  // OPENING_BRACKET
     {ErrorBegin},  // CLOSING_BRACE
     {ListEnd},     // CLOSING_BRACKET
     {ErrorBegin},  // QUOTE
     {ErrorBegin},  // ESCAPE
     {},            // COMMA
     {ErrorBegin},  // COLON
     {},            // WHITE_SPACE
     {},            // LINE_BREAK
     {ErrorBegin},  // OTHER
     /*STRUCT*/
     {ErrorBegin},                  // OPENING_BRACE
     {ErrorBegin},                  // OPENING_BRACKET
     {StructMemberEnd, StructEnd},  // CLOSING_BRACE
     {ErrorBegin},                  // CLOSING_BRACKET
     {ErrorBegin},                  // QUOTE
     {ErrorBegin},                  // ESCAPE
     {StructMemberEnd},             // COMMA
     {ErrorBegin},                  // COLON
     {},                            // WHITE_SPACE
     {},                            // LINE_BREAK
     {ErrorBegin}}};                // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_BFN)] = {
    {               /*ROOT*/
     {ErrorBegin},  // OPENING_BRACE
     {ErrorBegin},  // OPENING_BRACKET
     {ErrorBegin},  // CLOSING_BRACE
     {ErrorBegin},  // CLOSING_BRACKET
     {ErrorBegin},  // QUOTE
     {ErrorBegin},  // ESCAPE
     {ErrorBegin},  // COMMA
     {ErrorBegin},  // COLON
     {ErrorBegin},  // WHITE_SPACE
     {ErrorBegin},  // LINE_BREAK
     {ErrorBegin},  // OTHER
     /*LIST*/
     {ErrorBegin},  // OPENING_BRACE
     {ErrorBegin},  // OPENING_BRACKET
     {ErrorBegin},  // CLOSING_BRACE
     {ErrorBegin},  // CLOSING_BRACKET
     {ErrorBegin},  // QUOTE
     {ErrorBegin},  // ESCAPE
     {ErrorBegin},  // COMMA
     {ErrorBegin},  // COLON
     {ErrorBegin},  // WHITE_SPACE
     {ErrorBegin},  // LINE_BREAK
     {ErrorBegin},  // OTHER
     /*STRUCT*/
     {ErrorBegin},                                                     // OPENING_BRACE
     {ErrorBegin},                                                     // OPENING_BRACKET
     {StructEnd},                                                      // CLOSING_BRACE
     {ErrorBegin},                                                     // CLOSING_BRACKET
     {StructMemberBegin, FieldNameBegin},                              // QUOTE
     {ErrorBegin},                                                     // ESCAPE
     {ErrorBegin},                                                     // COMMA
     {ErrorBegin},                                                     // COLON
     {},                                                               // WHITE_SPACE
     {},                                                               // LINE_BREAK
     {ErrorBegin}}};                                                   // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_FLN)] = {{               /*ROOT*/
                                                        {ErrorBegin},  // OPENING_BRACE
                                                        {ErrorBegin},  // OPENING_BRACKET
                                                        {ErrorBegin},  // CLOSING_BRACE
                                                        {ErrorBegin},  // CLOSING_BRACKET
                                                        {ErrorBegin},  // QUOTE
                                                        {ErrorBegin},  // ESCAPE
                                                        {ErrorBegin},  // COMMA
                                                        {ErrorBegin},  // COLON
                                                        {ErrorBegin},  // WHITE_SPACE
                                                        {ErrorBegin},  // LINE_BREAK
                                                        {ErrorBegin},  // OTHER
                                                        /*LIST*/
                                                        {ErrorBegin},  // OPENING_BRACE
                                                        {ErrorBegin},  // OPENING_BRACKET
                                                        {ErrorBegin},  // CLOSING_BRACE
                                                        {ErrorBegin},  // CLOSING_BRACKET
                                                        {ErrorBegin},  // QUOTE
                                                        {ErrorBegin},  // ESCAPE
                                                        {ErrorBegin},  // COMMA
                                                        {ErrorBegin},  // COLON
                                                        {ErrorBegin},  // WHITE_SPACE
                                                        {ErrorBegin},  // LINE_BREAK
                                                        {ErrorBegin},  // OTHER
                                                        /*STRUCT*/
                                                        {},              // OPENING_BRACE
                                                        {},              // OPENING_BRACKET
                                                        {},              // CLOSING_BRACE
                                                        {},              // CLOSING_BRACKET
                                                        {FieldNameEnd},  // QUOTE
                                                        {},              // ESCAPE
                                                        {},              // COMMA
                                                        {},              // COLON
                                                        {},              // WHITE_SPACE
                                                        {},              // LINE_BREAK
                                                        {}}};            // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_FNE)] = {{                 /*ROOT*/
                                                        {ErrorBegin},    // OPENING_BRACE
                                                        {ErrorBegin},    // OPENING_BRACKET
                                                        {ErrorBegin},    // CLOSING_BRACE
                                                        {ErrorBegin},    // CLOSING_BRACKET
                                                        {ErrorBegin},    // QUOTE
                                                        {ErrorBegin},    // ESCAPE
                                                        {ErrorBegin},    // COMMA
                                                        {ErrorBegin},    // COLON
                                                        {ErrorBegin},    // WHITE_SPACE
                                                        {ErrorBegin},    // LINE_BREAK
                                                        {ErrorBegin},    // OTHER
                                                        /*LIST*/
                                                        {ErrorBegin},  // OPENING_BRACE
                                                        {ErrorBegin},  // OPENING_BRACKET
                                                        {ErrorBegin},  // CLOSING_BRACE
                                                        {ErrorBegin},  // CLOSING_BRACKET
                                                        {ErrorBegin},  // QUOTE
                                                        {ErrorBegin},  // ESCAPE
                                                        {ErrorBegin},  // COMMA
                                                        {ErrorBegin},  // COLON
                                                        {ErrorBegin},  // WHITE_SPACE
                                                        {ErrorBegin},  // LINE_BREAK
                                                        {ErrorBegin},  // OTHER
                                                        /*STRUCT*/
                                                        {},            // OPENING_BRACE
                                                        {},            // OPENING_BRACKET
                                                        {},            // CLOSING_BRACE
                                                        {},            // CLOSING_BRACKET
                                                        {},            // QUOTE
                                                        {},            // ESCAPE
                                                        {},            // COMMA
                                                        {},            // COLON
                                                        {},            // WHITE_SPACE
                                                        {},            // LINE_BREAK
                                                        {}}};          // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_PFN)] = {{               /*ROOT*/
                                                        {ErrorBegin},  // OPENING_BRACE
                                                        {ErrorBegin},  // OPENING_BRACKET
                                                        {ErrorBegin},  // CLOSING_BRACE
                                                        {ErrorBegin},  // CLOSING_BRACKET
                                                        {ErrorBegin},  // QUOTE
                                                        {ErrorBegin},  // ESCAPE
                                                        {ErrorBegin},  // COMMA
                                                        {ErrorBegin},  // COLON
                                                        {ErrorBegin},  // WHITE_SPACE
                                                        {ErrorBegin},  // LINE_BREAK
                                                        {ErrorBegin},  // OTHER
                                                        /*LIST*/
                                                        {ErrorBegin},  // OPENING_BRACE
                                                        {ErrorBegin},  // OPENING_BRACKET
                                                        {ErrorBegin},  // CLOSING_BRACE
                                                        {ErrorBegin},  // CLOSING_BRACKET
                                                        {ErrorBegin},  // QUOTE
                                                        {ErrorBegin},  // ESCAPE
                                                        {ErrorBegin},  // COMMA
                                                        {ErrorBegin},  // COLON
                                                        {ErrorBegin},  // WHITE_SPACE
                                                        {ErrorBegin},  // LINE_BREAK
                                                        {ErrorBegin},  // OTHER
                                                        /*STRUCT*/
                                                        {ErrorBegin},    // OPENING_BRACE
                                                        {ErrorBegin},    // OPENING_BRACKET
                                                        {ErrorBegin},    // CLOSING_BRACE
                                                        {ErrorBegin},    // CLOSING_BRACKET
                                                        {ErrorBegin},    // QUOTE
                                                        {ErrorBegin},    // ESCAPE
                                                        {ErrorBegin},    // COMMA
                                                        {},              // COLON
                                                        {},              // WHITE_SPACE
                                                        {},              // LINE_BREAK
                                                        {ErrorBegin}}};  // OTHER

  pda_tlt[static_cast<StateT>(pda_state_t::PD_ERR)] = {{                 /*ROOT*/
                                                        {},              // OPENING_BRACE
                                                        {},              // OPENING_BRACKET
                                                        {},              // CLOSING_BRACE
                                                        {},              // CLOSING_BRACKET
                                                        {},              // QUOTE
                                                        {},              // ESCAPE
                                                        {},              // COMMA
                                                        {},              // COLON
                                                        {},              // WHITE_SPACE
                                                        {},              // LINE_BREAK
                                                        {},              // OTHER
                                                        /*LIST*/
                                                        {},  // OPENING_BRACE
                                                        {},  // OPENING_BRACKET
                                                        {},  // CLOSING_BRACE
                                                        {},  // CLOSING_BRACKET
                                                        {},  // QUOTE
                                                        {},  // ESCAPE
                                                        {},  // COMMA
                                                        {},  // COLON
                                                        {},  // WHITE_SPACE
                                                        {},  // LINE_BREAK
                                                        {},  // OTHER
                                                        /*STRUCT*/
                                                        {},    // OPENING_BRACE
                                                        {},    // OPENING_BRACKET
                                                        {},    // CLOSING_BRACE
                                                        {},    // CLOSING_BRACKET
                                                        {},    // QUOTE
                                                        {},    // ESCAPE
                                                        {},    // COMMA
                                                        {},    // COLON
                                                        {},    // WHITE_SPACE
                                                        {},    // LINE_BREAK
                                                        {}}};  // OTHER
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

void json_column::null_fill(row_offset_t up_to_row_offset)
{
  // Fill all the rows up to up_to_row_offset with "empty"/null rows
  validity.resize(word_index(up_to_row_offset) + 1);
  std::fill_n(std::back_inserter(string_offsets),
              up_to_row_offset - string_offsets.size(),
              (string_offsets.size() > 0) ? string_offsets.back() : 0);
  std::fill_n(std::back_inserter(string_lengths), up_to_row_offset - string_lengths.size(), 0);
  std::fill_n(std::back_inserter(child_offsets),
              up_to_row_offset + 1 - child_offsets.size(),
              (child_offsets.size() > 0) ? child_offsets.back() : 0);
  current_offset = up_to_row_offset;
}

void json_column::level_child_cols_recursively(row_offset_t min_row_count)
{
  // Fill this columns with nulls up to the given row count
  null_fill(min_row_count);

  // If this is a struct column, we need to level all its child columns
  if (type == json_col_t::StructColumn) {
    for (auto it = std::begin(child_columns); it != std::end(child_columns); it++) {
      it->second.level_child_cols_recursively(min_row_count);
    }
  }
  // If this is a list column, we need to make sure that its child column levels its children
  else if (type == json_col_t::ListColumn) {
    auto it = std::begin(child_columns);
    // Make that child column fill its child columns up to its own row count
    if (it != std::end(child_columns)) {
      it->second.level_child_cols_recursively(it->second.current_offset);
    }
  }
};

void json_column::append_row(uint32_t row_index,
                             json_col_t row_type,
                             uint32_t string_offset,
                             uint32_t string_end,
                             uint32_t child_count)
{
  // If, thus far, the column's type couldn't be inferred, we infer it to the given type
  if (type == json_col_t::Unknown) {
    type = row_type;
  }
  // If, at some point within a column, we encounter a nested type (list or struct),
  // we change that column's type to that respective nested type and invalidate all previous rows
  else if (type == json_col_t::StringColumn &&
           (row_type == json_col_t::ListColumn || row_type == json_col_t::StructColumn)) {
    // Change the column type
    type = row_type;

    // Invalidate all previous entries, as they were _not_ of the nested type to which we just
    // converted
    std::fill_n(validity.begin(), validity.size(), 0);
    valid_count = 0U;
  }
  // If this is a nested column but we're trying to insert either (a) a list node into a struct
  // column or (b) a struct node into a list column, we fail
  CUDF_EXPECTS(not((type == json_col_t::ListColumn and row_type == json_col_t::StructColumn) or
                   (type == json_col_t::StructColumn and row_type == json_col_t::ListColumn)),
               "A mix of lists and structs within the same column is not supported");

  // We shouldn't run into this, as we shouldn't be asked to append an "unknown" row type
  CUDF_EXPECTS(type != json_col_t::Unknown, "Encountered invalid JSON token sequence");

  // Fill all the omitted rows with "empty"/null rows (if needed)
  null_fill(row_index);

  // Table listing what we intend to use for a given column type and row type combination
  // col type | row type  => {valid, FAIL, null}
  // -----------------------------------------------
  // List     | List      => valid
  // List     | Struct    => FAIL
  // List     | String    => null
  // Struct   | List      => FAIL
  // Struct   | Struct    => valid
  // Struct   | String    => null
  // String   | List      => valid (we switch col type to list, null'ing all previous rows)
  // String   | Struct    => valid (we switch col type to list, null'ing all previous rows)
  // String   | String    => valid
  bool const is_valid = (type == row_type);
  if (static_cast<size_type>(validity.size()) < word_index(current_offset)) validity.push_back({});
  if (is_valid) { set_bit_unsafe(&validity.back(), intra_word_index(current_offset)); }
  valid_count += (is_valid) ? 1U : 0U;
  string_offsets.push_back(string_offset);
  string_lengths.push_back(string_end - string_offset);
  child_offsets.push_back((child_offsets.size() > 0) ? child_offsets.back() + child_count : 0);
  current_offset++;
};

namespace detail {

void get_stack_context(device_span<SymbolT const> json_in,
                       SymbolT* d_top_of_stack,
                       rmm::cuda_stream_view stream)
{
  check_input_size(json_in.size());

  // Range of encapsulating function that comprises:
  // -> DFA simulation for filtering out brackets and braces inside of quotes
  // -> Logical stack to infer the stack context
  CUDF_FUNC_RANGE();

  // Symbol representing the JSON-root (i.e., we're at nesting level '0')
  constexpr StackSymbolT root_symbol = '_';
  // This can be any stack symbol from the stack alphabet that does not push onto stack
  constexpr StackSymbolT read_symbol = 'x';

  // Number of stack operations in the input (i.e., number of '{', '}', '[', ']' outside of quotes)
  rmm::device_scalar<SymbolOffsetT> d_num_stack_ops(stream);

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
                                  d_num_stack_ops.data(),
                                  to_stack_op::start_state,
                                  stream);

  // Copy back to actual number of stack operations
  auto const num_stack_ops = d_num_stack_ops.value(stream);

  // stack operations with indices are converted to top of the stack for each character in the input
  fst::sparse_stack_op_to_top_of_stack<StackLevelT>(
    stack_ops.data(),
    device_span<SymbolOffsetT>{stack_op_indices.data(), num_stack_ops},
    JSONToStackOp{},
    d_top_of_stack,
    root_symbol,
    read_symbol,
    json_in.size(),
    stream);
}

std::pair<rmm::device_uvector<PdaTokenT>, rmm::device_uvector<SymbolOffsetT>> get_token_stream(
  device_span<SymbolT const> json_in,
  cudf::io::json_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  check_input_size(json_in.size());

  // Range of encapsulating function that parses to internal columnar data representation
  CUDF_FUNC_RANGE();

  auto const new_line_delimited_json = options.is_enabled_lines();

  // Prepare for PDA transducer pass, merging input symbols with stack symbols
  rmm::device_uvector<PdaSymbolGroupIdT> pda_sgids = [json_in, stream]() {
    // Memory holding the top-of-stack stack context for the input
    rmm::device_uvector<StackSymbolT> stack_op_indices{json_in.size(), stream};

    // Identify what is the stack context for each input character (JSON-root, struct, or list)
    get_stack_context(json_in, stack_op_indices.data(), stream);

    rmm::device_uvector<PdaSymbolGroupIdT> pda_sgids{json_in.size(), stream};
    auto zip_in = thrust::make_zip_iterator(json_in.data(), stack_op_indices.data());
    thrust::transform(rmm::exec_policy(stream),
                      zip_in,
                      zip_in + json_in.size(),
                      pda_sgids.data(),
                      tokenizer_pda::PdaSymbolToSymbolGroupId{});
    return pda_sgids;
  }();

  // PDA transducer alias
  using ToTokenStreamFstT =
    cudf::io::fst::detail::Dfa<StackSymbolT,
                               tokenizer_pda::NUM_PDA_SGIDS,
                               static_cast<tokenizer_pda::StateT>(
                                 tokenizer_pda::pda_state_t::PD_NUM_STATES)>;

  // Instantiating PDA transducer
  std::vector<std::vector<char>> pda_sgid_identity{tokenizer_pda::NUM_PDA_SGIDS};
  std::generate(std::begin(pda_sgid_identity),
                std::end(pda_sgid_identity),
                [i = char{0}]() mutable { return std::vector<char>{i++}; });
  ToTokenStreamFstT json_to_tokens_fst{pda_sgid_identity,
                                       tokenizer_pda::get_transition_table(new_line_delimited_json),
                                       tokenizer_pda::get_translation_table(),
                                       stream};

  // Perform a PDA-transducer pass
  // Compute the maximum amount of tokens that can possibly be emitted for a given input size
  // Worst case ratio of tokens per input char is given for a struct with an empty field name, that
  // may be arbitrarily deeply nested: {"":_}, where '_' is a placeholder for any JSON value,
  // possibly another such struct. That is, 6 tokens for 5 chars (plus chars and tokens of '_')
  std::size_t constexpr min_chars_per_struct  = 5;
  std::size_t constexpr max_tokens_per_struct = 6;
  auto const max_token_out_count =
    cudf::util::div_rounding_up_safe(json_in.size(), min_chars_per_struct) * max_tokens_per_struct;
  rmm::device_scalar<std::size_t> num_written_tokens{stream};
  rmm::device_uvector<PdaTokenT> tokens{max_token_out_count, stream, mr};
  rmm::device_uvector<SymbolOffsetT> tokens_indices{max_token_out_count, stream, mr};

  json_to_tokens_fst.Transduce(pda_sgids.begin(),
                               static_cast<SymbolOffsetT>(json_in.size()),
                               tokens.data(),
                               tokens_indices.data(),
                               num_written_tokens.data(),
                               tokenizer_pda::start_state,
                               stream);

  auto const num_total_tokens = num_written_tokens.value(stream);
  tokens.resize(num_total_tokens, stream);
  tokens_indices.resize(num_total_tokens, stream);

  CUDF_EXPECTS(num_total_tokens <= max_token_out_count,
               "Generated token count exceeds the expected token count");

  return std::make_pair(std::move(tokens), std::move(tokens_indices));
}

/**
 * @brief Parses the given JSON string and generates a tree representation of the given input.
 *
 * @param[in,out] root_column The root column of the hierarchy of columns into which data is parsed
 * @param[in,out] current_data_path The stack represents the path from the JSON root node to the
 * first node encountered in \p input
 * @param[in] input The JSON input in host memory
 * @param[in] d_input The JSON input in device memory
 * @param[in] options Parsing options specifying the parsing behaviour
 * @param[in] include_quote_char Whether to include the original quote chars around string values,
 * allowing to distinguish string values from numeric and literal values
 * @param[in] stream The CUDA stream to which kernels are dispatched
 * @param[in] mr Optional, resource with which to allocate
 * @return The columnar representation of the data from the given JSON input
 */
void make_json_column(json_column& root_column,
                      std::stack<tree_node>& current_data_path,
                      host_span<SymbolT const> input,
                      device_span<SymbolT const> d_input,
                      cudf::io::json_reader_options const& options,
                      bool include_quote_char,
                      rmm::cuda_stream_view stream,
                      rmm::mr::device_memory_resource* mr)
{
  // Range of encapsulating function that parses to internal columnar data representation
  CUDF_FUNC_RANGE();

  // Parse the JSON and get the token stream
  const auto [d_tokens_gpu, d_token_indices_gpu] = get_token_stream(d_input, options, stream, mr);

  // Copy the JSON tokens to the host
  thrust::host_vector<PdaTokenT> tokens =
    cudf::detail::make_host_vector_async(d_tokens_gpu, stream);
  thrust::host_vector<SymbolOffsetT> token_indices_gpu =
    cudf::detail::make_host_vector_async(d_token_indices_gpu, stream);

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

  // Depending on whether we want to include the quotes of strings or not, respectively, we:
  // (a) strip off the beginning quote included in StringBegin and FieldNameBegin or
  // (b) include of the end quote excluded from in StringEnd and strip off the beginning quote
  // included FieldNameBegin
  auto get_token_index = [include_quote_char](PdaTokenT const token,
                                              SymbolOffsetT const token_index) {
    constexpr SymbolOffsetT quote_char_size = 1;
    switch (token) {
      // Optionally strip off quote char included for StringBegin
      case token_t::StringBegin: return token_index + (include_quote_char ? 0 : quote_char_size);
      // Optionally include trailing quote char for string values excluded for StringEnd
      case token_t::StringEnd: return token_index + (include_quote_char ? quote_char_size : 0);
      // Strip off quote char included for FieldNameBegin
      case token_t::FieldNameBegin: return token_index + quote_char_size;
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
      case token_t::StructMemberBegin: return "StructMemberBegin";
      case token_t::StructMemberEnd: return "StructMemberEnd";
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
  auto get_selected_column = [](std::stack<tree_node>& current_data_path) {
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
          current_data_path.top().column->column_order.push_back(list_child_name);
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
    struct_col->column_order.push_back(field_name);
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
  constexpr uint32_t zero_child_count = 0;

  CUDF_EXPECTS(tokens.size() == token_indices_gpu.size(),
               "Unexpected mismatch in number of token types and token indices");
  CUDF_EXPECTS(tokens.size() > 0, "Empty JSON input not supported");

  // The JSON root may only be a struct, list, string, or value node
  CUDF_EXPECTS(is_valid_root_token(tokens[offset]), "Invalid beginning of JSON document");

  while (offset < tokens.size()) {
    // Verify there's at least the JSON root node left on the stack to which we can append data
    CUDF_EXPECTS(current_data_path.size() > 0, "Invalid JSON structure");

    // Verify that the current node in the tree (which becomes this nodes parent) can have children
    CUDF_EXPECTS(current_data_path.top().column->type == json_col_t::ListColumn or
                   current_data_path.top().column->type == json_col_t::StructColumn,
                 "Invalid JSON structure");

    // The token we're currently parsing
    auto const& token = tokens[offset];

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
                               token_to_column_type(tokens[offset]),
                               get_token_index(tokens[offset], token_indices_gpu[offset]),
                               get_token_index(tokens[offset], token_indices_gpu[offset]),
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
                 get_token_index(tokens[offset], token_indices_gpu[offset]),
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
                               token_to_column_type(tokens[offset]),
                               get_token_index(tokens[offset], token_indices_gpu[offset]),
                               get_token_index(tokens[offset], token_indices_gpu[offset]),
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
                 get_token_index(tokens[offset], token_indices_gpu[offset]),
                 current_data_path.top().num_children);

      // Pop list from the path stack
      current_data_path.pop();
    }

    // Error token
    else if (token == token_t::ErrorBegin) {
#ifdef NJP_DEBUG_PRINT
      std::cout << "[ErrorBegin]\n";
      std::cout << "@" << get_token_index(tokens[offset], token_indices_gpu[offset]);
#endif
      CUDF_FAIL("Parser encountered an invalid format.");
    }

    // FieldName, String, or Value (begin, end)-pair
    else if (token == token_t::FieldNameBegin or token == token_t::StringBegin or
             token == token_t::ValueBegin) {
      // Verify that this token has the right successor to build a correct (being, end) token pair
      CUDF_EXPECTS((offset + 1) < tokens.size(), "Invalid JSON token sequence");
      CUDF_EXPECTS(tokens[offset + 1] == end_of_partner(token), "Invalid JSON token sequence");

      // The offset to the first symbol from the JSON input associated with the current token
      auto const& token_begin_offset = get_token_index(tokens[offset], token_indices_gpu[offset]);

      // The offset to one past the last symbol associated with the current token
      auto const& token_end_offset =
        get_token_index(tokens[offset + 1], token_indices_gpu[offset + 1]);

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

/**
 * @brief Retrieves the parse_options to be used for type inference and type casting
 *
 * @param options The reader options to influence the relevant type inference and type casting
 * options
 */
auto parsing_options(cudf::io::json_reader_options const& options)
{
  auto parse_opts = cudf::io::parse_options{',', '\n', '\"', '.'};

  auto const stream     = cudf::get_default_stream();
  parse_opts.dayfirst   = options.is_enabled_dayfirst();
  parse_opts.keepquotes = options.is_enabled_keep_quotes();
  parse_opts.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  parse_opts.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  parse_opts.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);
  return parse_opts;
}

std::pair<std::unique_ptr<column>, std::vector<column_name_info>> json_column_to_cudf_column(
  json_column const& json_col,
  device_span<SymbolT const> d_input,
  cudf::io::json_reader_options const& options,
  std::optional<schema_element> schema,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  // Range of orchestrating/encapsulating function
  CUDF_FUNC_RANGE();

  auto make_validity =
    [stream, mr](json_column const& json_col) -> std::pair<rmm::device_buffer, size_type> {
    return {rmm::device_buffer{json_col.validity.data(),
                               bitmask_allocation_size_bytes(json_col.current_offset),
                               stream,
                               mr},
            json_col.current_offset - json_col.valid_count};
  };

  auto get_child_schema = [schema](auto child_name) -> std::optional<schema_element> {
    if (schema.has_value()) {
      auto const result = schema.value().child_types.find(child_name);
      if (result != std::end(schema.value().child_types)) { return result->second; }
    }
    return {};
  };

  switch (json_col.type) {
    case json_col_t::StringColumn: {
      auto const col_size = json_col.string_offsets.size();
      CUDF_EXPECTS(json_col.string_offsets.size() == json_col.string_lengths.size(),
                   "string offset, string length mismatch");

      // Move string_offsets and string_lengths to GPU
      rmm::device_uvector<json_column::row_offset_t> d_string_offsets =
        cudf::detail::make_device_uvector_async(
          json_col.string_offsets, stream, rmm::mr::get_current_device_resource());
      rmm::device_uvector<json_column::row_offset_t> d_string_lengths =
        cudf::detail::make_device_uvector_async(
          json_col.string_lengths, stream, rmm::mr::get_current_device_resource());

      // Prepare iterator that returns (string_offset, string_length)-tuples
      auto offset_length_it =
        thrust::make_zip_iterator(d_string_offsets.begin(), d_string_lengths.begin());

      // Prepare iterator that returns (string_offset, string_length)-pairs needed by inference
      auto string_ranges_it =
        thrust::make_transform_iterator(offset_length_it, [] __device__(auto ip) {
          return thrust::pair<json_column::row_offset_t, std::size_t>{
            thrust::get<0>(ip), static_cast<std::size_t>(thrust::get<1>(ip))};
        });

      // Prepare iterator that returns (string_ptr, string_length)-pairs needed by type conversion
      auto string_spans_it = thrust::make_transform_iterator(
        offset_length_it, [data = d_input.data()] __device__(auto ip) {
          return thrust::pair<const char*, std::size_t>{
            data + thrust::get<0>(ip), static_cast<std::size_t>(thrust::get<1>(ip))};
        });

      data_type target_type{};

      if (schema.has_value()) {
#ifdef NJP_DEBUG_PRINT
        std::cout << "-> explicit type: "
                  << (schema.has_value() ? std::to_string(static_cast<int>(schema->type.id()))
                                         : "n/a");
#endif
        target_type = schema.value().type;
      }
      // Infer column type, if we don't have an explicit type for it
      else {
        target_type = cudf::io::detail::infer_data_type(
          parsing_options(options).json_view(), d_input, string_ranges_it, col_size, stream);
      }

      // Convert strings to the inferred data type
      auto col = experimental::detail::parse_data(string_spans_it,
                                                  col_size,
                                                  target_type,
                                                  make_validity(json_col).first,
                                                  parsing_options(options).view(),
                                                  stream,
                                                  mr);

      // Reset nullable if we do not have nulls
      // This is to match the existing JSON reader's behaviour:
      // - Non-string columns will always be returned as nullable
      // - String columns will be returned as nullable, iff there's at least one null entry
      if (target_type.id() == type_id::STRING and col->null_count() == 0) {
        col->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);
      }

      // For string columns return ["offsets", "char"] schema
      if (target_type.id() == type_id::STRING) {
        return {std::move(col), {{"offsets"}, {"chars"}}};
      }
      // Non-string leaf-columns (e.g., numeric) do not have child columns in the schema
      else {
        return {std::move(col), {}};
      }
      break;
    }
    case json_col_t::StructColumn: {
      std::vector<std::unique_ptr<column>> child_columns;
      std::vector<column_name_info> column_names{};
      size_type num_rows{json_col.current_offset};
      // Create children columns
      for (auto const& col_name : json_col.column_order) {
        auto const& col = json_col.child_columns.find(col_name);
        column_names.emplace_back(col->first);
        auto const& child_col      = col->second;
        auto [child_column, names] = json_column_to_cudf_column(
          child_col, d_input, options, get_child_schema(col_name), stream, mr);
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
      column_names.emplace_back(
        json_col.child_columns.empty() ? list_child_name : json_col.child_columns.begin()->first);

      rmm::device_uvector<json_column::row_offset_t> d_offsets =
        cudf::detail::make_device_uvector_async(json_col.child_offsets, stream, mr);
      auto offsets_column =
        std::make_unique<column>(data_type{type_id::INT32}, num_rows, d_offsets.release());
      // Create children column
      auto [child_column, names] =
        json_col.child_columns.empty()
          ? std::pair<std::unique_ptr<column>,
                      std::vector<column_name_info>>{std::make_unique<column>(), {}}
          : json_column_to_cudf_column(json_col.child_columns.begin()->second,
                                       d_input,
                                       options,
                                       get_child_schema(json_col.child_columns.begin()->first),
                                       stream,
                                       mr);
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

table_with_metadata host_parse_nested_json(device_span<SymbolT const> d_input,
                                           cudf::io::json_reader_options const& options,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  // Range of orchestrating/encapsulating function
  CUDF_FUNC_RANGE();

  auto const h_input = cudf::detail::make_std_vector_async(d_input, stream);

  auto const new_line_delimited_json = options.is_enabled_lines();

  // Get internal JSON column
  json_column root_column{};
  std::stack<tree_node> data_path{};

  constexpr uint32_t row_offset_zero            = 0;
  constexpr uint32_t token_begin_offset_zero    = 0;
  constexpr uint32_t token_end_offset_zero      = 0;
  constexpr uint32_t node_init_child_count_zero = 0;

  // Whether the tokenizer stage should keep quote characters for string values
  // If the tokenizer keeps the quote characters, they may be stripped during type casting
  constexpr bool include_quote_chars = true;

  // We initialize the very root node and root column, which represent the JSON document being
  // parsed. That root node is a list node and that root column is a list column. The column has the
  // root node as its only row. The values parsed from the JSON input will be treated as follows:
  // (1) For JSON lines: we expect to find a list of JSON values that all
  // will be inserted into this root list column. (2) For regular JSON: we expect to have only a
  // single value (list, struct, string, number, literal) that will be inserted into this root
  // column.
  root_column.append_row(
    row_offset_zero, json_col_t::ListColumn, token_begin_offset_zero, token_end_offset_zero, 1);

  // Push the root node onto the stack for the data path
  data_path.push({&root_column, row_offset_zero, nullptr, node_init_child_count_zero});

  make_json_column(
    root_column, data_path, h_input, d_input, options, include_quote_chars, stream, mr);

  // data_root refers to the root column of the data represented by the given JSON string
  auto const& data_root =
    new_line_delimited_json ? root_column : root_column.child_columns.begin()->second;

  // Zero row entries
  if (data_root.type == json_col_t::ListColumn && data_root.child_columns.size() == 0) {
    return table_with_metadata{std::make_unique<table>(std::vector<std::unique_ptr<column>>{})};
  }

  // Verify that we were in fact given a list of structs (or in JSON speech: an array of objects)
  auto constexpr single_child_col_count = 1;
  CUDF_EXPECTS(data_root.type == json_col_t::ListColumn and
                 data_root.child_columns.size() == single_child_col_count and
                 data_root.child_columns.begin()->second.type == json_col_t::StructColumn,
               "Currently the nested JSON parser only supports an array of (nested) objects");

  // Slice off the root list column, which has only a single row that contains all the structs
  auto const& root_struct_col = data_root.child_columns.begin()->second;

  // Initialize meta data to be populated while recursing through the tree of columns
  std::vector<std::unique_ptr<column>> out_columns;
  std::vector<column_name_info> out_column_names;

  // Iterate over the struct's child columns and convert to cudf column
  size_type column_index = 0;
  for (auto const& col_name : root_struct_col.column_order) {
    auto const& json_col = root_struct_col.child_columns.find(col_name)->second;
    // Insert this columns name into the schema
    out_column_names.emplace_back(col_name);

    std::optional<schema_element> child_schema_element = std::visit(
      cudf::detail::visitor_overload{
        [column_index](const std::vector<data_type>& user_dtypes) -> std::optional<schema_element> {
          auto ret = (static_cast<std::size_t>(column_index) < user_dtypes.size())
                       ? std::optional<schema_element>{{user_dtypes[column_index]}}
                       : std::optional<schema_element>{};
#ifdef NJP_DEBUG_PRINT
          std::cout << "Column by index: #" << column_index << ", type id: "
                    << (ret.has_value() ? std::to_string(static_cast<int>(ret->type.id())) : "n/a")
                    << ", with " << (ret.has_value() ? ret->child_types.size() : 0) << " children"
                    << "\n";
#endif
          return ret;
        },
        [col_name](
          std::map<std::string, data_type> const& user_dtypes) -> std::optional<schema_element> {
          auto ret = (user_dtypes.find(col_name) != std::end(user_dtypes))
                       ? std::optional<schema_element>{{user_dtypes.find(col_name)->second}}
                       : std::optional<schema_element>{};
#ifdef NJP_DEBUG_PRINT
          std::cout << "Column by flat name: '" << col_name << "', type id: "
                    << (ret.has_value() ? std::to_string(static_cast<int>(ret->type.id())) : "n/a")
                    << ", with " << (ret.has_value() ? ret->child_types.size() : 0) << " children"
                    << "\n";
#endif
          return ret;
        },
        [col_name](std::map<std::string, schema_element> const& user_dtypes)
          -> std::optional<schema_element> {
          auto ret = (user_dtypes.find(col_name) != std::end(user_dtypes))
                       ? user_dtypes.find(col_name)->second
                       : std::optional<schema_element>{};
#ifdef NJP_DEBUG_PRINT
          std::cout << "Column by nested name: #" << col_name << ", type id: "
                    << (ret.has_value() ? std::to_string(static_cast<int>(ret->type.id())) : "n/a")
                    << ", with " << (ret.has_value() ? ret->child_types.size() : 0) << " children"
                    << "\n";
#endif
          return ret;
        }},
      options.get_dtypes());

    // Get this JSON column's cudf column and schema info
    auto [cudf_col, col_name_info] =
      json_column_to_cudf_column(json_col, d_input, options, child_schema_element, stream, mr);
    out_column_names.back().children = std::move(col_name_info);
    out_columns.emplace_back(std::move(cudf_col));

    column_index++;
  }

  return table_with_metadata{std::make_unique<table>(std::move(out_columns)), {out_column_names}};
}

}  // namespace detail
}  // namespace cudf::io::json

// Debug print flag
#undef NJP_DEBUG_PRINT
