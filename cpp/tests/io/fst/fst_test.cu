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

#include <io/fst/lookup_tables.cuh>
#include <io/utilities/hostdevice_vector.hpp>
#include <tests/io/fst/common.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/types.hpp>

#include <thrust/iterator/discard_iterator.h>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdlib>
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
 * @return The final state after parsing the entire input
 */
template <typename InputItT,
          typename StateT,
          typename SymbolGroupLutT,
          typename TransitionTableT,
          typename TransducerTableT,
          typename OutputItT,
          typename StateItT,
          typename IndexOutputItT>
static StateT fst_baseline(InputItT begin,
                                                         InputItT end,
                                                         StateT const& init_state,
                                                         SymbolGroupLutT symbol_group_lut,
                                                         TransitionTableT transition_table,
                                                         TransducerTableT translation_table,
                                                         OutputItT out_tape,
                                                         StateItT out_state_tape,
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

    *out_state_tape++ = state;

    // Transition the state of the finite-state machine
    state = static_cast<char>(transition_table[state][symbol_group]);

    // Continue with next symbol from input tape
    in_offset++;
  }
  return state;
}

using namespace cudf::test::io::json;
}  // namespace

// Base test fixture for tests
struct FstTest : public cudf::test::BaseFixture {
};

TEST_F(FstTest, GroundTruth)
{
  // Type used to represent the atomic symbol type used within the finite-state machine
  using SymbolT = char;

  // Type sufficiently large to index symbols within the input and output (may be unsigned)
  using SymbolOffsetT = uint32_t;

  // Helper class to set up transition table, symbol group lookup table, and translation table
  using DfaFstT = cudf::io::fst::detail::Dfa<char, NUM_SYMBOL_GROUPS, TT_NUM_STATES>;

  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  // Test input
  std::string input = R"(  {)"
                      R"("category": "reference",)"
                      R"("index:" [4,12,42],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "Sayings of the Century",)"
                      R"("price": 8.95)"
                      R"(}  )"
                      R"({)"
                      R"("category": "reference",)"
                      R"("index:" [4,{},null,{"a":[]}],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "Sayings of the Century",)"
                      R"("price": 8.95)"
                      R"(}  {} [] [ ])";

  size_t string_size                 = input.size() * (1 << 10);
  auto d_input_scalar                = cudf::make_string_scalar(input);
  auto& d_string_scalar              = static_cast<cudf::string_scalar&>(*d_input_scalar);
  const cudf::size_type repeat_times = string_size / input.size();
  auto d_input_string                = cudf::strings::repeat_string(d_string_scalar, repeat_times);
  auto& d_input = static_cast<cudf::scalar_type_t<std::string>&>(*d_input_string);
  input         = d_input.to_string(stream);

  // Prepare input & output buffers
  constexpr std::size_t single_item = 1;
  hostdevice_vector<SymbolT> output_gpu(input.size(), stream_view);
  hostdevice_vector<SymbolOffsetT> output_gpu_size(single_item, stream_view);
  hostdevice_vector<SymbolOffsetT> out_indexes_gpu(input.size(), stream_view);

  // Run algorithm
  DfaFstT parser{pda_sgs, pda_state_tt, pda_out_tt, stream.value()};

  // Allocate device-side temporary storage & run algorithm
  auto final_state = parser.Transduce(d_input.data(),
                   static_cast<SymbolOffsetT>(d_input.size()),
                   output_gpu.device_ptr(),
                   out_indexes_gpu.device_ptr(),
                   output_gpu_size.device_ptr(),
                   start_state,
                   stream.value());

  // Async copy results from device to host
  output_gpu.device_to_host(stream.view());
  out_indexes_gpu.device_to_host(stream.view());
  output_gpu_size.device_to_host(stream.view());

  // Prepare CPU-side results for verification
  std::string output_cpu{};
  std::vector<SymbolOffsetT> out_index_cpu{};
  output_cpu.reserve(input.size());
  out_index_cpu.reserve(input.size());

  // Run CPU-side algorithm
  auto final_state_ref = fst_baseline(std::begin(input),
               std::end(input),
               start_state,
               pda_sgs,
               pda_state_tt,
               pda_out_tt,
               std::back_inserter(output_cpu),
               thrust::make_discard_iterator(),
               std::back_inserter(out_index_cpu));

  // Make sure results have been copied back to host
  stream.synchronize();

  // Verify results
  ASSERT_EQ(output_gpu_size[0], output_cpu.size());
  ASSERT_EQ(final_state, final_state_ref);
  CUDF_TEST_EXPECT_VECTOR_EQUAL(output_gpu, output_cpu, output_cpu.size());
  CUDF_TEST_EXPECT_VECTOR_EQUAL(out_indexes_gpu, out_index_cpu, output_cpu.size());
}

TEST_F(FstTest, GroudTruthPartial)
{
  // Type used to represent the atomic symbol type used within the finite-state machine
  using SymbolT = char;

  // Type sufficiently large to index symbols within the input and output (may be unsigned)
  using SymbolOffsetT = uint32_t;

  // Helper class to set up transition table, symbol group lookup table, and translation table
  using DfaFstT = cudf::io::fst::detail::Dfa<char, NUM_SYMBOL_GROUPS, TT_NUM_STATES>;

  // Prepare cuda stream for data transfers & kernels
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);

  // Test input
  std::string input = R"(  {)"
                      R"("category": "reference",)"
                      R"("index:" [4,12,42],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "Sayings of the Century",)"
                      R"("price": 8.95)"
                      R"(}  )"
                      R"({)"
                      R"("category": "reference",)"
                      R"("index:" [4,{},null,{"a":[]}],)"
                      R"("author": "Nigel Rees",)"
                      R"("title": "Sayings of the Century",)"
                      R"("price": 8.95)"
                      R"(} "\")"; // to make sure we land in different states in each input section

  auto d_input_scalar                = cudf::make_string_scalar(input);
  auto& d_string_scalar              = static_cast<cudf::string_scalar&>(*d_input_scalar);
  cudf::size_type const repeat_times = 1 << 15;
  auto d_input_string                = cudf::strings::repeat_string(d_string_scalar, repeat_times);
  auto& d_input = static_cast<cudf::scalar_type_t<std::string>&>(*d_input_string);
  input         = d_input.to_string(stream);

  // Prepare input & reference output buffers

  // Run algorithm
  DfaFstT parser{pda_sgs, pda_state_tt, pda_out_tt, stream.value()};

  // Compute a reference solution
  std::string out_cpu{};
  std::vector<char> out_state_cpu{};
  std::vector<SymbolOffsetT> out_index_cpu{};
  out_cpu.reserve(input.size());
  out_state_cpu.reserve(input.size());
  out_index_cpu.reserve(input.size());

  // Run CPU-side algorithm
  auto final_state_ref = fst_baseline(std::begin(input),
               std::end(input),
               start_state,
               pda_sgs,
               pda_state_tt,
               pda_out_tt,
               std::back_inserter(out_cpu),
               std::back_inserter(out_state_cpu),
               std::back_inserter(out_index_cpu));

  // Make sure results have been copied back to host
  stream.synchronize();

  for (SymbolOffsetT start_ofs : {0, 10, 100, 1000, 10000, static_cast<int>(input.size() / 2)}) {
    SCOPED_TRACE(start_ofs);
    std::array<uint32_t, TT_NUM_STATES> state_vector{};
    parser.Reduce(d_input.data(), start_ofs, state_vector, stream.value());
    auto new_start_state = state_vector[start_state];
    ASSERT_EQ(new_start_state, out_state_cpu[start_ofs]);

    constexpr std::size_t single_item = 1;
    auto const local_size = static_cast<SymbolOffsetT>(d_input.size() - start_ofs);
    hostdevice_vector<SymbolT> out_gpu(local_size, stream_view);
    hostdevice_vector<SymbolOffsetT> out_gpu_size(single_item, stream_view);
    hostdevice_vector<SymbolOffsetT> out_index_gpu(local_size, stream_view);
    auto final_state = parser.Transduce(d_input.data() + start_ofs, local_size, out_gpu.device_ptr(),
      out_index_gpu.device_ptr(),
      out_gpu_size.device_ptr(),
      new_start_state,
      stream.value());
      
    // Async copy results from device to host
    out_gpu.device_to_host(stream_view);
    out_index_gpu.device_to_host(stream_view);
    out_gpu_size.device_to_host(stream_view);

    std::string out_cpu_partial{};
    std::vector<SymbolOffsetT> out_index_cpu_partial{};
    out_cpu_partial.reserve(local_size);
    out_index_cpu_partial.reserve(local_size);
    for (std::size_t i = 0; i < out_cpu.size(); i++) {
      if (out_index_cpu[i] >= start_ofs) {
        out_cpu_partial.push_back(out_cpu[i]);
        out_index_cpu_partial.push_back(out_index_cpu[i] - start_ofs);
      }
    }

    ASSERT_EQ(out_gpu_size[0], out_cpu_partial.size());
    ASSERT_EQ(final_state, final_state_ref);
    CUDF_TEST_EXPECT_VECTOR_EQUAL(out_gpu, out_cpu_partial, out_cpu_partial.size());
    CUDF_TEST_EXPECT_VECTOR_EQUAL(out_index_gpu, out_index_cpu_partial, out_cpu_partial.size());
  }
}

CUDF_TEST_PROGRAM_MAIN()
