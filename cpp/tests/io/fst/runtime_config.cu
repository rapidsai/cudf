/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/discard_iterator.h>

#include <random>
#include <string>
#include <vector>

// Type used to represent the atomic symbol type used within the finite-state machine
using SymbolT = char;
using StateT  = char;

// Type sufficiently large to index symbols within the input and output (may be unsigned)
using SymbolOffsetT = uint32_t;

namespace runtime_config {

/*
 * Contrived FST that rings a bell whenever a delimiter character
 * (randomly chosen at runtime from a set of possible delimiters) is read
 */
std::vector<SymbolT> const delimiters{'\n', '\b', '\v', '\t', '\f', '\r'};
enum class dfa_states : StateT { TT_OOS = 0U, TT_STR, TT_ESC, TT_NUM_STATES };
enum class dfa_symbol_group_id : uint32_t {
  QUOTE_CHAR,        ///< Quote character SG: "
  ESCAPE_CHAR,       ///< Escape character SG: '\'
  DELIMITER_CHAR,    ///< Delimiter character SG \in delimiters = {\n, \b, \v, \t, \f, \r}
  OTHER_SYMBOLS,     ///< SG implicitly matching all other characters
  NUM_SYMBOL_GROUPS  ///< Total number of symbol groups
};

// Aliases for readability of the transition table
constexpr auto TT_OOS            = dfa_states::TT_OOS;
constexpr auto TT_STR            = dfa_states::TT_STR;
constexpr auto TT_ESC            = dfa_states::TT_ESC;
constexpr auto TT_NUM_STATES     = static_cast<StateT>(dfa_states::TT_NUM_STATES);
constexpr auto NUM_SYMBOL_GROUPS = static_cast<uint32_t>(dfa_symbol_group_id::NUM_SYMBOL_GROUPS);

// Transition table
std::array<std::array<dfa_states, NUM_SYMBOL_GROUPS>, TT_NUM_STATES> const pda_state_tt{{
  /* IN_STATE      "       \    <delim>   OTHER  */
  /* TT_OOS */ {{TT_STR, TT_OOS, TT_OOS, TT_OOS}},
  /* TT_STR */ {{TT_OOS, TT_ESC, TT_OOS, TT_STR}},
  /* TT_ESC */ {{TT_STR, TT_STR, TT_OOS, TT_STR}},
}};

// The DFA's starting state
constexpr auto pda_start_state = static_cast<StateT>(TT_OOS);

struct TransduceToRuntimeConfig {
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
    // OOS   | Sigma           -> Sigma
    // STR   | Sigma\{<delim>} -> Sigma\{<delim>}
    // ESC   | Sigma\{<delim>  -> Sigma\{<delim>}
    // ---------- SPECIAL CASES: --------------
    //    Input symbol translates to output symbol
    // STR   | {<delim>}       -> {\a}
    // ESC   | {<delim>}       -> {\a}

    // Case when delimiter character needs to be replaced by alarm character
    if (match_id == static_cast<SymbolGroupT>(dfa_symbol_group_id::DELIMITER_CHAR) &&
        (state_id == static_cast<StateT>(dfa_states::TT_STR) ||
         state_id == static_cast<StateT>(dfa_states::TT_ESC))) {
      return '\a';
    }
    // In all other cases we simply output the input symbol
    return read_symbol;
  }

  /**
   * @brief Returns the number of output characters for a given transition.
   */
  template <typename StateT, typename SymbolGroupT, typename SymbolT>
  constexpr CUDF_HOST_DEVICE int32_t operator()(StateT const state_id,
                                                SymbolGroupT const match_id,
                                                SymbolT const read_symbol) const
  {
    return 1;
  }
};

}  // namespace runtime_config

// Base test fixture for tests
struct RuntimeConfigFstTest : public cudf::test::BaseFixture {};

TEST_F(RuntimeConfigFstTest, SimpleInput)
{
  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  // The i-th string representing all the characters of a symbol group
  std::array<std::vector<SymbolT>, runtime_config::NUM_SYMBOL_GROUPS - 1> pda_sgs{
    {{'"'}, {'\\'}, {'d'}}};

  std::array<SymbolT, 6> delimiter_chars{{'\n', '\b', '\v', '\t', '\f', '\r'}};
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, delimiter_chars.size());
  SymbolT random_delimiter       = distrib(gen);
  pda_sgs[pda_sgs.size() - 1][0] = random_delimiter;

  // Test input
  std::string input             = R"({)"
                                  R"("category": "reference",)"
                                  R"("index:" [4,{},null,{"a":[]}],)"
                                  R"("author": "Nigel Rees",)"
                                  R"("title": "Sayings of the Century",)"
                                  R"("price": 8.95)"
                                  R"(})";
  std::size_t const string_size = 40000;
  std::size_t const log_repetitions =
    static_cast<std::size_t>(std::ceil(std::log2(string_size / input.size())));
  for (std::size_t i = 0; i < log_repetitions; i++)
    input = input + random_delimiter + input;
  rmm::device_uvector<char> d_input = cudf::detail::make_device_uvector_sync(
    input, stream_view, rmm::mr::get_current_device_resource());

  auto parser = cudf::io::fst::detail::make_fst(
    cudf::io::fst::detail::make_symbol_group_lut(pda_sgs),
    cudf::io::fst::detail::make_transition_table(runtime_config::pda_state_tt),
    cudf::io::fst::detail::make_translation_functor(runtime_config::TransduceToRuntimeConfig{}),
    stream_view);

  rmm::device_uvector<SymbolT> d_output(
    input.size(), stream_view, rmm::mr::get_current_device_resource());
  rmm::device_scalar<SymbolOffsetT> d_output_size(stream_view,
                                                  rmm::mr::get_current_device_resource());
  parser.Transduce(d_input.data(),
                   static_cast<SymbolOffsetT>(d_input.size()),
                   d_output.data(),
                   thrust::make_discard_iterator(),
                   d_output_size.data(),
                   runtime_config::pda_start_state,
                   stream_view);

  // Copy results from device to host
  stream_view.synchronize();
  std::size_t output_size = d_output_size.value(stream_view);
  std::string output(output_size, 0);
  ;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    output.data(), d_output.data(), output.size(), cudaMemcpyDeviceToHost, stream_view));
  stream_view.synchronize();

  // Verify results
  std::string expected_output = input;
  for (std::size_t i = 0; i < log_repetitions; i++)
    expected_output = expected_output + '\a' + expected_output;

  CUDF_TEST_EXPECT_VECTOR_EQUAL(output, expected_output, output.size());
}

CUDF_TEST_PROGRAM_MAIN()
