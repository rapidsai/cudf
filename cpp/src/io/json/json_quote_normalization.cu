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

using SymbolT       = char;
using StateT        = char;
using SymbolOffsetT = uint32_t;

namespace normalize_quotes {

// Type sufficiently large to index symbols within the input and output (may be unsigned)
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
    // SEC   | {'}             -> {'}
    // SEC   | Sigma\{'}       -> {\*}

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
    // symbol <s> is handled by transitions from SEC. For now, there is no output for this
    // transition
    if ((match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::ESCAPE_CHAR)) &&
        ((state_id == static_cast<StateT>(dfa_states::TT_SQS)))) {
      return 0;
    }
    // Case when an escaped single quote in an input single-quoted string needs to be replaced by an
    // unescaped single quote
    if ((match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::SINGLE_QUOTE_CHAR)) &&
        ((state_id == static_cast<StateT>(dfa_states::TT_SEC)))) {
      return '\'';
    }
    // Case when an escaped symbol <s> that is not a single-quote needs to be replaced with \<s>
    if (state_id == static_cast<StateT>(dfa_states::TT_SEC)) {
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
    bool const sec_outputs_escape_sequence =
      (state_id == static_cast<StateT>(dfa_states::TT_SEC)) &&
      (match_id != static_cast<SymbolGroupT>(dfa_symbol_group_id::SINGLE_QUOTE_CHAR));
    // Number of characters to output on this transition
    if (sec_outputs_escape_sequence) { return 2; }
    // Whether this transition translates to no output <nop>
    bool const sqs_outputs_nop =
      (state_id == static_cast<StateT>(dfa_states::TT_SQS)) &&
      (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::ESCAPE_CHAR));
    // Number of characters to output on this transition
    if (sqs_outputs_nop) { return 0; }
    return 1;
  }
};

}  // namespace normalize_quotes

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

}  // namespace detail
}  // namespace cudf::io::json
