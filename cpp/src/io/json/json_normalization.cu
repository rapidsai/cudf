/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "io/fst/lookup_tables.cuh"

#include <cudf/io/detail/json.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/discard_iterator.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace cudf::io::json {

// Type used to represent the atomic symbol type used within the finite-state machine
using SymbolT = char;
using StateT  = char;

// Type sufficiently large to index symbols within the input and output (may be unsigned)
using SymbolOffsetT = uint32_t;

namespace normalize_quotes {

enum class dfa_states : StateT { TT_OOS = 0U, TT_DQS, TT_SQS, TT_DEC, TT_SEC, TT_NUM_STATES };
enum class dfa_symbol_group_id : uint32_t {
  DOUBLE_QUOTE_CHAR,  ///< Quote character SG: "
  SINGLE_QUOTE_CHAR,  ///< Quote character SG: '
  ESCAPE_CHAR,        ///< Escape character SG: '\'
  NEWLINE_CHAR,       ///< Newline character SG: '\n'
  OTHER_SYMBOLS,      ///< SG implicitly matching all other characters
  NUM_SYMBOL_GROUPS   ///< Total number of symbol groups
};

// Aliases for readability of the transition table
constexpr auto TT_OOS            = dfa_states::TT_OOS;
constexpr auto TT_DQS            = dfa_states::TT_DQS;
constexpr auto TT_SQS            = dfa_states::TT_SQS;
constexpr auto TT_DEC            = dfa_states::TT_DEC;
constexpr auto TT_SEC            = dfa_states::TT_SEC;
constexpr auto TT_NUM_STATES     = static_cast<StateT>(dfa_states::TT_NUM_STATES);
constexpr auto NUM_SYMBOL_GROUPS = static_cast<uint32_t>(dfa_symbol_group_id::NUM_SYMBOL_GROUPS);

// The i-th string representing all the characters of a symbol group
std::array<std::vector<SymbolT>, NUM_SYMBOL_GROUPS - 1> const qna_sgs{
  {{'\"'}, {'\''}, {'\\'}, {'\n'}}};

// Transition table
std::array<std::array<dfa_states, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const qna_state_tt{{
  /* IN_STATE      "       '       \       \n    OTHER  */
  /* TT_OOS */ {{TT_DQS, TT_SQS, TT_OOS, TT_OOS, TT_OOS}},
  /* TT_DQS */ {{TT_OOS, TT_DQS, TT_DEC, TT_OOS, TT_DQS}},
  /* TT_SQS */ {{TT_SQS, TT_OOS, TT_SEC, TT_OOS, TT_SQS}},
  /* TT_DEC */ {{TT_DQS, TT_DQS, TT_DQS, TT_OOS, TT_DQS}},
  /* TT_SEC */ {{TT_SQS, TT_SQS, TT_SQS, TT_OOS, TT_SQS}},
}};

// The DFA's starting state
constexpr auto start_state = static_cast<StateT>(TT_OOS);

struct TransduceToNormalizedQuotes {
  /**
   * @brief Returns the <relative_offset>-th output symbol on the transition (state_id, match_id).
   */
  template <typename StateT, typename SymbolGroupT, typename RelativeOffsetT, typename SymbolT>
  constexpr CUDF_HOST_DEVICE SymbolT operator()(StateT const state_id,
                                                SymbolGroupT const match_id,
                                                RelativeOffsetT const relative_offset,
                                                SymbolT const read_symbol) const
  {
    // -------- TRANSLATION TABLE ------------
    //      Let the alphabet set be Sigma
    // ---------------------------------------
    // ---------- NON-SPECIAL CASES: ----------
    //      Output symbol same as input symbol <s>
    // state | read_symbol <s> -> output_symbol <s>
    // DQS   | Sigma           -> Sigma
    // DEC   | Sigma           -> Sigma
    // OOS   | Sigma\{'}       -> Sigma\{'}
    // SQS   | Sigma\{', "}    -> Sigma\{', "}
    // ---------- SPECIAL CASES: --------------
    //    Input symbol translates to output symbol
    // OOS   | {'}             -> {"}
    // SQS   | {'}             -> {"}
    // SQS   | {"}             -> {\"}
    // SQS   | {\}             -> <nop>
    // DQS   | {\}             -> <nop>
    // SEC   | {'}             -> {'}
    // SEC   | Sigma\{'}       -> {\*}
    // DEC   | {'}             -> {'}
    // DEC   | Sigma\{'}       -> {\*}

    // Whether this transition translates to the escape sequence: \"
    bool const outputs_escape_sequence =
      (state_id == static_cast<StateT>(dfa_states::TT_SQS)) &&
      (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::DOUBLE_QUOTE_CHAR));
    // Case when a double quote needs to be replaced by the escape sequence: \"
    if (outputs_escape_sequence) { return (relative_offset == 0) ? '\\' : '"'; }
    // Case when a single quote needs to be replaced by a double quote
    if ((match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::SINGLE_QUOTE_CHAR)) &&
        ((state_id == static_cast<StateT>(dfa_states::TT_SQS)) ||
         (state_id == static_cast<StateT>(dfa_states::TT_OOS)))) {
      return '"';
    }
    // Case when the read symbol is an escape character - the actual translation for \<s> for some
    // symbol <s> is handled by transitions from SEC. The same logic applies for the transition from
    // DEC. For now, there is no output for this transition
    if (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::ESCAPE_CHAR) &&
        (state_id == static_cast<StateT>(dfa_states::TT_SQS) ||
         state_id == static_cast<StateT>(dfa_states::TT_DQS))) {
      return 0;
    }
    // Case when an escaped single quote in an input single-quoted or double-quoted string needs
    // to be replaced by an unescaped single quote
    if (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::SINGLE_QUOTE_CHAR) &&
        (state_id == static_cast<StateT>(dfa_states::TT_SEC) ||
         state_id == static_cast<StateT>(dfa_states::TT_DEC))) {
      return '\'';
    }
    // Case when an escaped symbol <s> that is not a single-quote needs to be replaced with \<s>
    if (state_id == static_cast<StateT>(dfa_states::TT_SEC) ||
        state_id == static_cast<StateT>(dfa_states::TT_DEC)) {
      return (relative_offset == 0) ? '\\' : read_symbol;
    }
    // In all other cases we simply output the input symbol
    return read_symbol;
  }

  /**
   * @brief Returns the number of output characters for a given transition. During quote
   * normalization, we always emit one output character (i.e., either the input character or the
   * single quote-input replaced by a double quote), except when we need to escape a double quote
   * that was previously inside a single-quoted string.
   */
  template <typename StateT, typename SymbolGroupT, typename SymbolT>
  constexpr CUDF_HOST_DEVICE int32_t operator()(StateT const state_id,
                                                SymbolGroupT const match_id,
                                                SymbolT const read_symbol) const
  {
    // Whether this transition translates to the escape sequence: \"
    bool const sqs_outputs_escape_sequence =
      (state_id == static_cast<StateT>(dfa_states::TT_SQS)) &&
      (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::DOUBLE_QUOTE_CHAR));
    // Number of characters to output on this transition
    if (sqs_outputs_escape_sequence) { return 2; }

    // Whether this transition translates to the escape sequence \<s> or unescaped '
    bool const sec_dec_outputs_escape_sequence =
      (state_id == static_cast<StateT>(dfa_states::TT_SEC) ||
       state_id == static_cast<StateT>(dfa_states::TT_DEC)) &&
      (match_id != static_cast<SymbolGroupT>(dfa_symbol_group_id::SINGLE_QUOTE_CHAR));
    // Number of characters to output on this transition
    if (sec_dec_outputs_escape_sequence) { return 2; }

    // Whether this transition translates to no output <nop>
    bool const sqs_dqs_outputs_nop =
      (state_id == static_cast<StateT>(dfa_states::TT_SQS) ||
       state_id == static_cast<StateT>(dfa_states::TT_DQS)) &&
      (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::ESCAPE_CHAR));
    // Number of characters to output on this transition
    if (sqs_dqs_outputs_nop) { return 0; }

    return 1;
  }
};

}  // namespace normalize_quotes

namespace normalize_whitespace {

enum class dfa_symbol_group_id : uint32_t {
  DOUBLE_QUOTE_CHAR,   ///< Quote character SG: "
  ESCAPE_CHAR,         ///< Escape character SG: '\\'
  NEWLINE_CHAR,        ///< Newline character SG: '\n'
  WHITESPACE_SYMBOLS,  ///< Whitespace characters SG: '\t' or ' '
  OTHER_SYMBOLS,       ///< SG implicitly matching all other characters
  NUM_SYMBOL_GROUPS    ///< Total number of symbol groups
};
// Alias for readability of symbol group ids
constexpr auto NUM_SYMBOL_GROUPS = static_cast<uint32_t>(dfa_symbol_group_id::NUM_SYMBOL_GROUPS);
// The i-th string representing all the characters of a symbol group
std::array<std::vector<SymbolT>, NUM_SYMBOL_GROUPS - 1> const wna_sgs{
  {{'"'}, {'\\'}, {'\n'}, {' ', '\t'}}};

/**
 * -------- FST states ---------
 * -----------------------------
 * TT_OOS | Out-of-string state handling whitespace and non-whitespace chars outside double
 *        |   quotes as well as any other character not enclosed by a string. Also handles
 *        |   newline character present within a string
 * TT_DQS | Double-quoted string state handling all characters within double quotes except
 *        |   newline character
 * TT_DEC | State handling escaped characters inside double-quoted string. Note that this
 *        |   state is necessary to process escaped double-quote characters. Without this
 *        |   state, whitespaces following escaped double quotes inside strings may be removed.
 *
 * NOTE: An important case NOT handled by this FST is that of whitespace following newline
 * characters within a string. Consider the following example
 * Input:           {"a":"x\n y"}
 * FST output:      {"a":"x\ny"}
 * Expected output: {"a":"x\n y"}
 * Such strings are not part of the JSON standard (characters allowed within quotes should
 * have ASCII at least 0x20 i.e. space character and above) but may be encountered while
 * reading JSON files
 */
enum class dfa_states : StateT { TT_OOS = 0U, TT_DQS, TT_DEC, TT_NUM_STATES };
// Aliases for readability of the transition table
constexpr auto TT_OOS        = dfa_states::TT_OOS;
constexpr auto TT_DQS        = dfa_states::TT_DQS;
constexpr auto TT_DEC        = dfa_states::TT_DEC;
constexpr auto TT_NUM_STATES = static_cast<StateT>(dfa_states::TT_NUM_STATES);

// Transition table
std::array<std::array<dfa_states, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const wna_state_tt{
  {/* IN_STATE      "       \       \n    <SPC>   OTHER  */
   /* TT_OOS */ {{TT_DQS, TT_OOS, TT_OOS, TT_OOS, TT_OOS}},
   /* TT_DQS */ {{TT_OOS, TT_DEC, TT_OOS, TT_DQS, TT_DQS}},
   /* TT_DEC */ {{TT_DQS, TT_DQS, TT_DQS, TT_DQS, TT_DQS}}}};

// The DFA's starting state
constexpr StateT start_state = static_cast<StateT>(TT_OOS);

struct TransduceToNormalizedWS {
  /**
   * @brief Returns the <relative_offset>-th output symbol on the transition (state_id, match_id).
   */
  template <typename StateT, typename SymbolGroupT, typename RelativeOffsetT, typename SymbolT>
  constexpr CUDF_HOST_DEVICE SymbolT operator()(StateT const state_id,
                                                SymbolGroupT const match_id,
                                                RelativeOffsetT const relative_offset,
                                                SymbolT const read_symbol) const
  {
    // -------- TRANSLATION TABLE ------------
    //      Let the alphabet set be Sigma
    // ---------------------------------------
    // ---------- NON-SPECIAL CASES: ----------
    //      Output symbol same as input symbol <s>
    // state | read_symbol <s>  -> output_symbol <s>
    // DQS   | Sigma            -> Sigma
    // OOS   | Sigma\{<SPC>,\t} -> Sigma\{<SPC>,\t}
    // DEC   | Sigma            -> Sigma
    // ---------- SPECIAL CASES: --------------
    //    Input symbol translates to output symbol
    // OOS   | {<SPC>}          -> <nop>
    // OOS   | {\t}             -> <nop>

    // Case when read symbol is a space or tab but is unquoted
    // This will be the same condition as in `operator()(state_id, match_id, read_symbol)` function
    // However, since there is no output in this case i.e. the count returned by
    // operator()(state_id, match_id, read_symbol) is zero, this function is never called.
    // So skipping the check for this case.

    // In all other cases, we have an output symbol for the input symbol.
    // We simply output the input symbol
    return read_symbol;
  }

  /**
   * @brief Returns the number of output characters for a given transition.
   * During whitespace normalization, we always emit one output character i.e., the input
   * character, except when we need to remove the space/tab character
   */
  template <typename StateT, typename SymbolGroupT, typename SymbolT>
  constexpr CUDF_HOST_DEVICE uint32_t operator()(StateT const state_id,
                                                 SymbolGroupT const match_id,
                                                 SymbolT const read_symbol) const
  {
    // Case when read symbol is a space or tab but is unquoted
    if (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::WHITESPACE_SYMBOLS) &&
        state_id == static_cast<StateT>(dfa_states::TT_OOS)) {
      return 0;
    }
    return 1;
  }
};

}  // namespace normalize_whitespace

namespace detail {

rmm::device_uvector<SymbolT> normalize_single_quotes(rmm::device_uvector<SymbolT>&& inbuf,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource* mr)
{
  auto parser = fst::detail::make_fst(
    fst::detail::make_symbol_group_lut(normalize_quotes::qna_sgs),
    fst::detail::make_transition_table(normalize_quotes::qna_state_tt),
    fst::detail::make_translation_functor(normalize_quotes::TransduceToNormalizedQuotes{}),
    stream);

  rmm::device_uvector<SymbolT> outbuf(inbuf.size() * 2, stream, mr);
  rmm::device_scalar<SymbolOffsetT> outbuf_size(stream, mr);
  parser.Transduce(inbuf.data(),
                   static_cast<SymbolOffsetT>(inbuf.size()),
                   outbuf.data(),
                   thrust::make_discard_iterator(),
                   outbuf_size.data(),
                   normalize_quotes::start_state,
                   stream);

  outbuf.resize(outbuf_size.value(stream), stream);
  return outbuf;
}

rmm::device_uvector<SymbolT> normalize_whitespace(rmm::device_uvector<SymbolT>&& inbuf,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  auto parser = fst::detail::make_fst(
    fst::detail::make_symbol_group_lut(normalize_whitespace::wna_sgs),
    fst::detail::make_transition_table(normalize_whitespace::wna_state_tt),
    fst::detail::make_translation_functor(normalize_whitespace::TransduceToNormalizedWS{}),
    stream);

  rmm::device_uvector<SymbolT> outbuf(inbuf.size(), stream, mr);
  rmm::device_scalar<SymbolOffsetT> outbuf_size(stream, mr);
  parser.Transduce(inbuf.data(),
                   static_cast<SymbolOffsetT>(inbuf.size()),
                   outbuf.data(),
                   thrust::make_discard_iterator(),
                   outbuf_size.data(),
                   normalize_whitespace::start_state,
                   stream);

  outbuf.resize(outbuf_size.value(stream), stream);
  return outbuf;
}

}  // namespace detail
}  // namespace cudf::io::json
