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

#include <io/fst/lookup_tables.cuh>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdlib>
#include <string>
#include <vector>

namespace {

//------------------------------------------------------------------------------
// CPU-BASED IMPLEMENTATIONS FOR VERIFICATION
//------------------------------------------------------------------------------
/**
 * @brief CPU-based implementation of a finite-state transducer (FST).
 *
 * @tparam InputItT Forward input iterator type to symbols fed into the FST
 * @tparam StateT Type representing states of the finite-state machine
 * @tparam SymbolGroupLutT Sequence container of symbol groups. Each symbol group is a sequence
 * container to symbols within that group.
 * @tparam TransitionTableT Two-dimensional container type
 * @tparam TransducerTableT Two-dimensional container type
 * @tparam OutputItT Forward output iterator type
 * @tparam IndexOutputItT Forward output iterator type
 * @param[in] begin Forward iterator to the beginning of the symbol sequence
 * @param[in] end Forward iterator to one past the last element of the symbol sequence
 * @param[in] init_state The starting state of the finite-state machine
 * @param[in] symbol_group_lut Sequence container of symbol groups. Each symbol group is a sequence
 * container to symbols within that group. The index of the symbol group containing a symbol being
 * read will be used as symbol_gid of the transition and translation tables.
 * @param[in] transition_table The two-dimensional transition table, i.e.,
 * transition_table[state][symbol_gid] -> new_state
 * @param[in] translation_table The two-dimensional transducer table, i.e.,
 * translation_table[state][symbol_gid] -> range_of_output_symbols
 * @param[out] out_tape A forward output iterator to which the transduced input will be written
 * @param[out] out_index_tape A forward output iterator to which indexes of the symbols that
 * actually caused some output are written to
 * @return A pair of iterators to one past the last element of (1) the transduced output symbol
 * sequence and (2) the indexes of
 */
template <typename InputItT,
          typename StateT,
          typename SymbolGroupLutT,
          typename TransitionTableT,
          typename TransducerTableT,
          typename OutputItT,
          typename IndexOutputItT>
static std::pair<OutputItT, IndexOutputItT> fst_baseline(InputItT begin,
                                                         InputItT end,
                                                         StateT const& init_state,
                                                         SymbolGroupLutT symbol_group_lut,
                                                         TransitionTableT transition_table,
                                                         TransducerTableT translation_table,
                                                         OutputItT out_tape,
                                                         IndexOutputItT out_index_tape)
{
  // Initialize "FSM" with starting state
  StateT state = init_state;

  // To track the symbol offset within the input that caused the FST to output
  std::size_t in_offset = 0;
  for (auto it = begin; it < end; it++) {
    // The symbol currently being read
    auto const& symbol = *it;

    // Iterate over symbol groups and search for the first symbol group containing the current
    // symbol, if no match is found we use cend(symbol_group_lut) as the "catch-all" symbol group
    auto symbol_group_it =
      std::find_if(std::cbegin(symbol_group_lut), std::cend(symbol_group_lut), [symbol](auto& sg) {
        return std::find(std::cbegin(sg), std::cend(sg), symbol) != std::cend(sg);
      });
    auto symbol_group = std::distance(std::cbegin(symbol_group_lut), symbol_group_it);

    // Output the translated symbols to the output tape
    out_tape = std::copy(std::cbegin(translation_table[state][symbol_group]),
                         std::cend(translation_table[state][symbol_group]),
                         out_tape);

    auto out_size = std::distance(std::cbegin(translation_table[state][symbol_group]),
                                  std::cend(translation_table[state][symbol_group]));

    out_index_tape = std::fill_n(out_index_tape, out_size, in_offset);

    // Transition the state of the finite-state machine
    state = static_cast<char>(transition_table[state][symbol_group]);

    // Continue with next symbol from input tape
    in_offset++;
  }
  return {out_tape, out_index_tape};
}
}  // namespace

// Base test fixture for tests
struct FstTest : public cudf::test::BaseFixture {};

void run_test(std::string& input)
{
  // Type used to represent the atomic symbol type used within the finite-state machine
  using SymbolT = char;

  // Type sufficiently large to index symbols within the input and output (may be unsigned)
  using SymbolOffsetT = uint32_t;

  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  // Run algorithm
  enum class dfa_states : char { TT_OOS = 0U, TT_DQS, TT_SQS, TT_DEC, TT_SEC, TT_NUM_STATES };

  enum class dfa_symbol_group_id : uint32_t {
    OPENING_BRACE,      ///< Opening brace SG: {
    OPENING_BRACKET,    ///< Opening bracket SG: [
    CLOSING_BRACE,      ///< Closing brace SG: }
    CLOSING_BRACKET,    ///< Closing bracket SG: ]
    DOUBLE_QUOTE_CHAR,  ///< Quote character SG: "
    SINGLE_QUOTE_CHAR,  ///< Quote character SG: '
    ESCAPE_CHAR,        ///< Escape character SG: '\'
    OTHER_SYMBOLS,      ///< SG implicitly matching all other characters
    NUM_SYMBOL_GROUPS   ///< Total number of symbol groups
  };

  // Aliases for readability of the transition table
  constexpr auto TT_OOS = dfa_states::TT_OOS;
  constexpr auto TT_DQS = dfa_states::TT_DQS;
  constexpr auto TT_SQS = dfa_states::TT_SQS;
  constexpr auto TT_DEC = dfa_states::TT_DEC;
  constexpr auto TT_SEC = dfa_states::TT_SEC;

  constexpr auto TT_NUM_STATES     = static_cast<char>(dfa_states::TT_NUM_STATES);
  constexpr auto NUM_SYMBOL_GROUPS = static_cast<uint32_t>(dfa_symbol_group_id::NUM_SYMBOL_GROUPS);

  // The i-th string representing all the characters of a symbol group
  std::array<std::string, NUM_SYMBOL_GROUPS - 1> const qna_sgs{"{", "[", "}", "]", "\"", "'", "\\"};

  // Transition table
  // Does not support JSON lines
  std::array<std::array<dfa_states, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const qna_state_tt{{
    /* IN_STATE    {       [       }       ]       "       '       \       OTHER */
    /* TT_OOS */ {{TT_OOS, TT_OOS, TT_OOS, TT_OOS, TT_DQS, TT_SQS, TT_OOS, TT_OOS}},
    /* TT_DQS */ {{TT_DQS, TT_DQS, TT_DQS, TT_DQS, TT_OOS, TT_DQS, TT_DEC, TT_DQS}},
    /* TT_SQS */ {{TT_SQS, TT_SQS, TT_SQS, TT_SQS, TT_SQS, TT_OOS, TT_SEC, TT_SQS}},
    /* TT_DEC */ {{TT_DQS, TT_DQS, TT_DQS, TT_DQS, TT_DQS, TT_DQS, TT_DQS, TT_DQS}},
    /* TT_SEC */ {{TT_SQS, TT_SQS, TT_SQS, TT_SQS, TT_SQS, TT_SQS, TT_SQS, TT_SQS}},
  }};

  // Translation table (i.e., for each transition, what are the symbols that we output)
  std::array<std::array<std::vector<char>, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const qna_out_tt{
    {/* IN_STATE         {      [      }      ]      "              '      \    OTHER */
     /* TT_OOS    */ {{{'{'}, {'['}, {'}'}, {']'}, {'"'}, {'"'}, {'\\'}, {'x'}}},
     /* TT_DQS    */ {{{'{'}, {'['}, {'}'}, {']'}, {'"'}, {'\''}, {'\\'}, {'x'}}},
     /* TT_SQS    */ {{{'{'}, {'['}, {'}'}, {']'}, {'\\', '"'}, {'"'}, {'\\'}, {'x'}}},
     /* TT_DEC    */ {{{'{'}, {'['}, {'}'}, {']'}, {'"'}, {'\''}, {'\\'}, {'x'}}},
     /* TT_SEC    */ {{{'{'}, {'['}, {'}'}, {']'}, {'"'}, {'\''}, {'\\'}, {'x'}}}}};

  // The DFA's starting state
  constexpr char start_state = static_cast<char>(TT_OOS);

  auto parser = cudf::io::fst::detail::make_fst(
    cudf::io::fst::detail::make_symbol_group_lut(qna_sgs),
    cudf::io::fst::detail::make_transition_table(qna_state_tt),
    cudf::io::fst::detail::make_translation_table<TT_NUM_STATES * NUM_SYMBOL_GROUPS>(qna_out_tt),
    stream);

  auto d_input_scalar = cudf::make_string_scalar(input);
  auto& d_input       = static_cast<cudf::scalar_type_t<std::string>&>(*d_input_scalar);

  // Prepare input & output buffers
  constexpr std::size_t single_item = 1;
  cudf::detail::hostdevice_vector<SymbolT> output_gpu(input.size() * 2, stream_view);
  cudf::detail::hostdevice_vector<SymbolOffsetT> output_gpu_size(single_item, stream_view);
  cudf::detail::hostdevice_vector<SymbolOffsetT> out_indexes_gpu(input.size(), stream_view);

  // Allocate device-side temporary storage & run algorithm
  parser.Transduce(d_input.data(),
                   static_cast<SymbolOffsetT>(d_input.size()),
                   output_gpu.device_ptr(),
                   out_indexes_gpu.device_ptr(),
                   output_gpu_size.device_ptr(),
                   start_state,
                   stream.value());

  // Async copy results from device to host
  output_gpu.device_to_host_async(stream.view());
  out_indexes_gpu.device_to_host_async(stream.view());
  output_gpu_size.device_to_host_async(stream.view());

  // Prepare CPU-side results for verification
  std::string output_cpu{};
  std::vector<SymbolOffsetT> out_index_cpu{};
  output_cpu.reserve(input.size());
  out_index_cpu.reserve(input.size());

  // Run CPU-side algorithm
  fst_baseline(std::begin(input),
               std::end(input),
               start_state,
               qna_sgs,
               qna_state_tt,
               qna_out_tt,
               std::back_inserter(output_cpu),
               std::back_inserter(out_index_cpu));

  // Make sure results have been copied back to host
  stream.synchronize();

  // Verify results
  ASSERT_EQ(output_gpu_size[0], output_cpu.size());
  std::cout << output_cpu << std::endl;
  CUDF_TEST_EXPECT_VECTOR_EQUAL(output_gpu, output_cpu, output_cpu.size());
  // TODO: indexing for multicharacter translations
  // CUDF_TEST_EXPECT_VECTOR_EQUAL(out_indexes_gpu, out_index_cpu, output_cpu.size());
}

TEST_F(FstTest, GroundTruth_QuoteNormalizationSimple1)
{
  std::string input = R"({"A":'TEST"'})";
  run_test(input);
}
TEST_F(FstTest, GroundTruth_QuoteNormalizationSimple2)
{
  std::string input = R"({'A':"TEST'"} ['OTHER STUFF'])";
  run_test(input);
}
TEST_F(FstTest, GroundTruth_QuoteNormalizationSimple3)
{
  std::string input = R"(['{"A": "B"}',"{'A': 'B'}"])";
  run_test(input);
}
TEST_F(FstTest, GroundTruth_QuoteNormalizationSimple4)
{
  std::string input = R"({"ain't ain't a word and you ain't supposed to say it":'"""""""""""'})";
  run_test(input);
}
TEST_F(FstTest, GroundTruth_QuoteNormalizationSimple5)
{
  std::string input = R"({"\"'\"'\"'\"'":'"\'"\'"\'"\'"'})";
  run_test(input);
}
TEST_F(FstTest, GroundTruth_QuoteNormalizationSimple6)
{
  std::string input = R"([{"ABC':'CBA":'XYZ":"ZXY'}])";
  run_test(input);
}
TEST_F(FstTest, GroundTruth_QuoteNormalizationSimple7)
{
  std::string input = R"(["\t","\\t","\\","\\\'\"\\\\","\n","\b"])";
  run_test(input);
}
CUDF_TEST_PROGRAM_MAIN()
