/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <io/fst/lookup_tables.cuh>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/discard_iterator.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace {

// Type used to represent the atomic symbol type used within the finite-state machine
// TODO: type aliasing to be declared in a common header for better maintainability and
//        pre-empt future bugs
using SymbolT = char;
using StateT  = char;

// Type sufficiently large to index symbols within the input and output (may be unsigned)
using SymbolOffsetT = uint32_t;
enum class dfa_states : char { TT_OOS = 0U, TT_DQS, TT_SQS, TT_DEC, TT_SEC, TT_NUM_STATES };
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
constexpr auto TT_NUM_STATES     = static_cast<char>(dfa_states::TT_NUM_STATES);
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
constexpr char start_state = static_cast<char>(TT_OOS);

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
    const bool outputs_escape_sequence =
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
    const bool sqs_outputs_escape_sequence =
      (state_id == static_cast<StateT>(dfa_states::TT_SQS)) &&
      (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::DOUBLE_QUOTE_CHAR));
    // Number of characters to output on this transition
    if (sqs_outputs_escape_sequence) { return 2; }
    // Whether this transition translates to the escape sequence \<s> or unescaped '
    const bool sec_outputs_escape_sequence =
      (state_id == static_cast<StateT>(dfa_states::TT_SEC)) &&
      (match_id != static_cast<SymbolGroupT>(dfa_symbol_group_id::SINGLE_QUOTE_CHAR));
    // Number of characters to output on this transition
    if (sec_outputs_escape_sequence) { return 2; }
    // Whether this transition translates to no output <nop>
    const bool sqs_outputs_nop =
      (state_id == static_cast<StateT>(dfa_states::TT_SQS)) &&
      (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::ESCAPE_CHAR));
    // Number of characters to output on this transition
    if (sqs_outputs_nop) { return 0; }
    return 1;
  }
};

}  // namespace

// Base test fixture for tests
struct FstTest : public cudf::test::BaseFixture {};

void run_test(std::string& input, std::string& output)
{
  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  auto parser = cudf::io::fst::detail::make_fst(
    cudf::io::fst::detail::make_symbol_group_lut(qna_sgs),
    cudf::io::fst::detail::make_transition_table(qna_state_tt),
    cudf::io::fst::detail::make_translation_functor(TransduceToNormalizedQuotes{}),
    stream);

  auto d_input_scalar = cudf::make_string_scalar(input, stream_view);
  auto& d_input       = static_cast<cudf::scalar_type_t<std::string>&>(*d_input_scalar);

  // Prepare input & output buffers
  constexpr std::size_t single_item = 1;
  cudf::detail::hostdevice_vector<SymbolT> output_gpu(input.size() * 2, stream_view);
  cudf::detail::hostdevice_vector<SymbolOffsetT> output_gpu_size(single_item, stream_view);

  // Allocate device-side temporary storage & run algorithm
  parser.Transduce(d_input.data(),
                   static_cast<SymbolOffsetT>(d_input.size()),
                   output_gpu.device_ptr(),
                   thrust::make_discard_iterator(),
                   output_gpu_size.device_ptr(),
                   start_state,
                   stream_view);

  // Async copy results from device to host
  output_gpu.device_to_host_async(stream_view);
  output_gpu_size.device_to_host_async(stream_view);

  // Make sure results have been copied back to host
  stream.synchronize();

  // Verify results
  ASSERT_EQ(output_gpu_size[0], output.size());
  CUDF_TEST_EXPECT_VECTOR_EQUAL(output_gpu, output, output.size());
}

TEST_F(FstTest, GroundTruth_QuoteNormalization1)
{
  std::string input  = R"({"A":'TEST"'})";
  std::string output = R"({"A":"TEST\""})";
  run_test(input, output);
}

TEST_F(FstTest, GroundTruth_QuoteNormalization2)
{
  std::string input  = R"({'A':"TEST'"} ['OTHER STUFF'])";
  std::string output = R"({"A":"TEST'"} ["OTHER STUFF"])";
  run_test(input, output);
}

TEST_F(FstTest, GroundTruth_QuoteNormalization3)
{
  std::string input  = R"(['{"A": "B"}',"{'A': 'B'}"])";
  std::string output = R"(["{\"A\": \"B\"}","{'A': 'B'}"])";
  run_test(input, output);
}

TEST_F(FstTest, GroundTruth_QuoteNormalization4)
{
  std::string input = R"({"ain't ain't a word and you ain't supposed to say it":'"""""""""""'})";
  std::string output =
    R"({"ain't ain't a word and you ain't supposed to say it":"\"\"\"\"\"\"\"\"\"\"\""})";
  run_test(input, output);
}

TEST_F(FstTest, GroundTruth_QuoteNormalization5)
{
  std::string input  = R"({"\"'\"'\"'\"'":'"\'"\'"\'"\'"'})";
  std::string output = R"({"\"'\"'\"'\"'":"\"'\"'\"'\"'\""})";
  run_test(input, output);
}

TEST_F(FstTest, GroundTruth_QuoteNormalization6)
{
  std::string input  = R"([{"ABC':'CBA":'XYZ":"ZXY'}])";
  std::string output = R"([{"ABC':'CBA":"XYZ\":\"ZXY"}])";
  run_test(input, output);
}

TEST_F(FstTest, GroundTruth_QuoteNormalization7)
{
  std::string input  = R"(["\t","\\t","\\","\\\'\"\\\\","\n","\b"])";
  std::string output = R"(["\t","\\t","\\","\\\'\"\\\\","\n","\b"])";
  run_test(input, output);
}

TEST_F(FstTest, GroundTruth_QuoteNormalization8)
{
  std::string input  = R"(['\t','\\t','\\','\\\"\'\\\\','\n','\b','\u0012'])";
  std::string output = R"(["\t","\\t","\\","\\\"'\\\\","\n","\b","\u0012"])";
  run_test(input, output);
}

TEST_F(FstTest, GroundTruth_QuoteNormalization_Invalid1)
{
  std::string input  = R"(["THIS IS A TEST'])";
  std::string output = R"(["THIS IS A TEST'])";
  run_test(input, output);
}

TEST_F(FstTest, GroundTruth_QuoteNormalization_Invalid2)
{
  std::string input  = R"(['THIS IS A TEST"])";
  std::string output = R"(["THIS IS A TEST\"])";
  run_test(input, output);
}

TEST_F(FstTest, GroundTruth_QuoteNormalization_Invalid3)
{
  std::string input  = R"({"MORE TEST'N":'RESUL})";
  std::string output = R"({"MORE TEST'N":"RESUL})";
  run_test(input, output);
}

TEST_F(FstTest, GroundTruth_QuoteNormalization_Invalid4)
{
  std::string input  = R"({"NUMBER":100'0,'STRING':'SOMETHING'})";
  std::string output = R"({"NUMBER":100"0,"STRING":"SOMETHING"})";
  run_test(input, output);
}

TEST_F(FstTest, GroundTruth_QuoteNormalization_Invalid5)
{
  std::string input  = R"({'NUMBER':100"0,"STRING":"SOMETHING"})";
  std::string output = R"({"NUMBER":100"0,"STRING":"SOMETHING"})";
  run_test(input, output);
}

TEST_F(FstTest, GroundTruth_QuoteNormalization_Invalid6)
{
  std::string input  = R"({'a':'\\''})";
  std::string output = R"({"a":"\\""})";
  run_test(input, output);
}

TEST_F(FstTest, GroundTruth_QuoteNormalization_Invalid7)
{
  std::string input  = R"(}'a': 'b'{)";
  std::string output = R"(}"a": "b"{)";
  run_test(input, output);
}

CUDF_TEST_PROGRAM_MAIN()
