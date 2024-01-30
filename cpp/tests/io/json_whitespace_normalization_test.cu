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
#include <io/fst/lookup_tables.cuh>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

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

enum class dfa_states : StateT { TT_OOS = 0U, TT_DQS, TT_DEC, TT_WS, TT_NUM_STATES };
enum class dfa_symbol_group_id : uint32_t {
  DOUBLE_QUOTE_CHAR,  ///< Quote character SG: "
  ESCAPE_CHAR,        ///< Escape character SG: '\\'
  NEWLINE_CHAR,       ///< Newline character SG: '\n'
  WHITESPACE_CHAR,    ///< Whitespace character SG: ' '
  TABSPACE_CHAR,      ///< Tabspace character SG: '\t'
  OTHER_SYMBOLS,      ///< SG implicitly matching all other characters
  NUM_SYMBOL_GROUPS   ///< Total number of symbol groups
};

// Aliases for readability of the transition table
constexpr auto TT_OOS            = dfa_states::TT_OOS;
constexpr auto TT_DQS            = dfa_states::TT_DQS;
constexpr auto TT_DEC            = dfa_states::TT_DEC;
constexpr auto TT_WS             = dfa_states::TT_WS;
constexpr auto TT_NUM_STATES     = static_cast<StateT>(dfa_states::TT_NUM_STATES);
constexpr auto NUM_SYMBOL_GROUPS = static_cast<uint32_t>(dfa_symbol_group_id::NUM_SYMBOL_GROUPS);

// The i-th string representing all the characters of a symbol group
std::array<std::vector<SymbolT>, NUM_SYMBOL_GROUPS - 1> const wna_sgs{
  {{'"'}, {'\\'}, {'\n'}, {' '}, {'\t'}}};

// Transition table
std::array<std::array<dfa_states, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const wna_state_tt{
  {/* IN_STATE      "       \       \n    <SPC>    \t      OTHER  */
   /* TT_OOS */ {{TT_DQS, TT_OOS, TT_OOS, TT_WS, TT_WS, TT_OOS}},
   /* TT_DQS */ {{TT_OOS, TT_DEC, TT_DQS, TT_DQS, TT_DQS, TT_DQS}},
   /* TT_DEC */ {{TT_DQS, TT_DQS, TT_DQS, TT_DQS, TT_DQS, TT_DQS}},
   /* TT_WS  */ {{TT_DQS, TT_OOS, TT_OOS, TT_WS, TT_WS, TT_OOS}}}};

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
    // WS    | Sigma\{<SPC>,\t} -> Sigma\{<SPC>,\t}
    // OOS   | Sigma\{<SPC>,\t} -> Sigma\{<SPC>,\t}
    // ---------- SPECIAL CASES: --------------
    //    Input symbol translates to output symbol
    // OOS   | {<SPC>}          -> <nop>
    // OOS   | {\t}             -> <nop>
    // WS    | {<SPC>}          -> <nop>
    // WS    | {\t}             -> <nop>

    // Case when read symbol is a whitespace or tabspace but is unquoted
    if (((match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::WHITESPACE_CHAR)) ||
         (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::TABSPACE_CHAR))) &&
        ((state_id == static_cast<StateT>(dfa_states::TT_OOS)) ||
         (state_id == static_cast<StateT>(dfa_states::TT_WS)))) {
      return 0;
    }
    // In all other cases we simply output the input symbol
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
    // Case when read symbol is a whitespace or tabspace but is unquoted
    if (((match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::WHITESPACE_CHAR)) ||
         (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::TABSPACE_CHAR))) &&
        ((state_id == static_cast<StateT>(dfa_states::TT_OOS)) ||
         (state_id == static_cast<StateT>(dfa_states::TT_WS)))) {
      return 0;
    }
    return 1;
  }
};
}  // namespace

// Base test fixture for tests
struct JsonWSNormalizationTest : public cudf::test::BaseFixture {};

void run_test(const std::string& input, const std::string& output)
{
  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  auto parser = cudf::io::fst::detail::make_fst(
    cudf::io::fst::detail::make_symbol_group_lut(wna_sgs),
    cudf::io::fst::detail::make_transition_table(wna_state_tt),
    cudf::io::fst::detail::make_translation_functor(TransduceToNormalizedWS{}),
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

  std::cout << "Expected output: " << output << std::endl << "Computed output: ";
  for (size_t i = 0; i < output_gpu_size[0]; i++)
    std::cout << output_gpu[i];
  std::cout << std::endl;
  // Verify results
  ASSERT_EQ(output_gpu_size[0], output.size());
  CUDF_TEST_EXPECT_VECTOR_EQUAL(output_gpu, output, output.size());
}

TEST_F(JsonWSNormalizationTest, GroundTruth_WSNormalization1)
{
  std::string input  = R"({"A" : "TEST" })";
  std::string output = R"({"A":"TEST"})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_WSNormalization2)
{
  std::string input  = R"({"A" :   "TEST" })";
  std::string output = R"({"A":"TEST"})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_WSNormalization3)
{
  std::string input  = R"({ "foo rapids": [1,2,3], "bar  rapids": 123 }
                          { "foo rapids": { "a": 1 }, "bar  rapids": 456 })";
  std::string output = R"({"foo rapids":[1,2,3],"bar  rapids":123}
{"foo rapids":{"a":1},"bar  rapids":456})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_WSNormalization4)
{
  std::string input  = "{\"a\":\t\"b\"}";
  std::string output  = R"({"a":"b"})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_WSNormalization5)
{
  std::string input  = R"({"a": [1, 2, 3, 4, 5, 6, 7, 8], "b": {"c": "d"}})";
  std::string output  = R"({"a":[1,2,3,4,5,6,7,8],"b":{"c":"d"}})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_WSNormalization6)
{
  std::string input  = R"({" a ":50})";
  std::string output  = R"({" a ":50})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_WSNormalization7)
{
  std::string input  = R"( { "a" : 50 })";
  std::string output  = R"({"a":50})";
  run_test(input, output);
}

TEST_F(JsonWSNormalizationTest, GroundTruth_WSNormalization8)
{
  std::string input  = R"({"a":50})";
  std::string output  = R"({"a":50})";
  run_test(input, output);
}

CUDF_TEST_PROGRAM_MAIN()
