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
#include "io/utilities/hostdevice_vector.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/discard_iterator.h>

#include <cstdlib>
#include <string>

namespace {
// Type used to represent the atomic symbol type used within the finite-state machine
using SymbolT = char;
using StateT  = char;

// Type sufficiently large to index symbols within the input and output (may be unsigned)
using SymbolOffsetT = uint32_t;

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
}  // namespace

// Base test fixture for tests
struct JsonWSNormalizationTest : public cudf::test::BaseFixture {};

void run_test(std::string const& input, std::string const& output)
{
  auto parser = cudf::io::fst::detail::make_fst(
    cudf::io::fst::detail::make_symbol_group_lut(wna_sgs),
    cudf::io::fst::detail::make_transition_table(wna_state_tt),
    cudf::io::fst::detail::make_translation_functor(TransduceToNormalizedWS{}),
    cudf::test::get_default_stream());

  auto d_input_scalar = cudf::make_string_scalar(input, cudf::test::get_default_stream());
  auto& d_input       = static_cast<cudf::scalar_type_t<std::string>&>(*d_input_scalar);

  // Prepare input & output buffers
  constexpr std::size_t single_item = 1;
  cudf::detail::hostdevice_vector<SymbolT> output_gpu(input.size(),
                                                      cudf::test::get_default_stream());
  cudf::detail::hostdevice_vector<SymbolOffsetT> output_gpu_size(single_item,
                                                                 cudf::test::get_default_stream());

  // Allocate device-side temporary storage & run algorithm
  parser.Transduce(d_input.data(),
                   static_cast<SymbolOffsetT>(d_input.size()),
                   output_gpu.device_ptr(),
                   thrust::make_discard_iterator(),
                   output_gpu_size.device_ptr(),
                   start_state,
                   cudf::test::get_default_stream());

  // Async copy results from device to host
  output_gpu.device_to_host_async(cudf::test::get_default_stream());
  output_gpu_size.device_to_host_async(cudf::test::get_default_stream());

  // Make sure results have been copied back to host
  cudf::test::get_default_stream().synchronize();

  // Verify results
  ASSERT_EQ(output_gpu_size[0], output.size());
  CUDF_TEST_EXPECT_VECTOR_EQUAL(output_gpu, output, output.size());
}

TEST_F(JsonWSNormalizationTest, GroundTruth_Spaces)
{
  std::string input  = R"({ "A" : "TEST" })";
  std::string output = R"({"A":"TEST"})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_MoreSpaces)
{
  std::string input  = R"({"a": [1, 2, 3, 4, 5, 6, 7, 8], "b": {"c": "d"}})";
  std::string output = R"({"a":[1,2,3,4,5,6,7,8],"b":{"c":"d"}})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_SpacesInString)
{
  std::string input  = R"({" a ":50})";
  std::string output = R"({" a ":50})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_NewlineInString)
{
  std::string input  = "{\"a\" : \"x\ny\"}\n{\"a\" : \"x\\ny\"}";
  std::string output = "{\"a\":\"x\ny\"}\n{\"a\":\"x\\ny\"}";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_Tabs)
{
  std::string input  = "{\"a\":\t\"b\"}";
  std::string output = R"({"a":"b"})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_SpacesAndTabs)
{
  std::string input  = "{\"A\" : \t\"TEST\" }";
  std::string output = R"({"A":"TEST"})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_MultilineJSONWithSpacesAndTabs)
{
  std::string input =
    "{ \"foo rapids\": [1,2,3], \"bar\trapids\": 123 }\n\t{ \"foo rapids\": { \"a\": 1 }, "
    "\"bar\trapids\": 456 }";
  std::string output =
    "{\"foo rapids\":[1,2,3],\"bar\trapids\":123}\n{\"foo rapids\":{\"a\":1},\"bar\trapids\":456}";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_PureJSONExample)
{
  std::string input  = R"([{"a":50}, {"a" : 60}])";
  std::string output = R"([{"a":50},{"a":60}])";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_NoNormalizationRequired)
{
  std::string input  = R"({"a\\n\r\a":50})";
  std::string output = R"({"a\\n\r\a":50})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_InvalidInput)
{
  std::string input  = "{\"a\" : \"b }\n{ \"c \" :\t\"d\"}";
  std::string output = "{\"a\":\"b }\n{\"c \":\"d\"}";
  run_test(input, output);
}

CUDF_TEST_PROGRAM_MAIN()
