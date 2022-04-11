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
#pragma once

#include "cub/util_type.cuh"
#include "dispatch_dfa.cuh"
#include <src/io/fst/symbol_lut.cuh>
#include <src/io/fst/transition_table.cuh>
#include <src/io/fst/translation_table.cuh>

#include <cstdint>

namespace cudf {
namespace io {
namespace fst {

/**
 * @brief Uses a deterministic finite automaton to transduce a sequence of symbols from an input
 * iterator to a sequence of transduced output symbols.
 *
 * @tparam SymbolItT Random-access input iterator type to symbols fed into the FST
 * @tparam DfaT The DFA specification
 * @tparam TransducedOutItT Random-access output iterator to which the transduced output will be
 * written
 * @tparam TransducedIndexOutItT Random-access output iterator type to which the indexes of the
 * symbols that caused some output to be written.
 * @tparam TransducedCountOutItT A single-item output iterator type to which the total number of
 * output symbols is written
 * @tparam OffsetT A type large enough to index into either of both: (a) the input symbols and (b)
 * the output symbols
 * @param[in] d_temp_storage Device-accessible allocation of temporary storage.  When NULL, the
 * required allocation size is written to \p temp_storage_bytes and no work is done.
 * @param[in,out] temp_storage_bytes Reference to size in bytes of \p d_temp_storage allocation
 * @param[in] dfa The DFA specifying the number of distinct symbol groups, transition table, and
 * translation table
 * @param[in] d_chars_in Random-access input iterator to the beginning of the sequence of input
 * symbols
 * @param[in] num_chars The total number of input symbols to process
 * @param[out] transduced_out_it Random-access output iterator to which the transduced output is
 * written
 * @param[out] transduced_out_idx_it Random-access output iterator to which, the index i is written
 * iff the i-th input symbol caused some output to be written
 * @param[out] d_num_transduced_out_it A single-item output iterator type to which the total number
 * of output symbols is written
 * @param[in] seed_state The DFA's starting state. For streaming DFAs this corresponds to the
 * "end-state" of the previous invocation of the algorithm.
 * @param[in] stream CUDA stream to launch kernels within. Default is the null-stream.
 */
template <typename DfaT,
          typename SymbolItT,
          typename TransducedOutItT,
          typename TransducedIndexOutItT,
          typename TransducedCountOutItT,
          typename OffsetT>
cudaError_t DeviceTransduce(void* d_temp_storage,
                            size_t& temp_storage_bytes,
                            DfaT dfa,
                            SymbolItT d_chars_in,
                            OffsetT num_chars,
                            TransducedOutItT transduced_out_it,
                            TransducedIndexOutItT transduced_out_idx_it,
                            TransducedCountOutItT d_num_transduced_out_it,
                            uint32_t seed_state = 0,
                            cudaStream_t stream = 0)
{
  using DispatchDfaT = detail::DispatchFSM<DfaT,
                                           SymbolItT,
                                           TransducedOutItT,
                                           TransducedIndexOutItT,
                                           TransducedCountOutItT,
                                           OffsetT>;

  return DispatchDfaT::Dispatch(d_temp_storage,
                                temp_storage_bytes,
                                dfa,
                                seed_state,
                                d_chars_in,
                                num_chars,
                                transduced_out_it,
                                transduced_out_idx_it,
                                d_num_transduced_out_it,
                                stream);
}

/**
 * @brief Helper class to facilitate the specification and instantiation of a DFA (i.e., the
 * transition table and its number of states, the mapping of symbols to symbol groups, and the
 * translation table that specifies which state transitions cause which output to be written).
 *
 * @tparam OutSymbolT The symbol type being output by the finite-state transducer
 * @tparam NUM_SYMBOLS The number of symbol groups amongst which to differentiate (one dimension of
 * the transition table)
 * @tparam TT_NUM_STATES The number of states defined by the DFA (the other dimension of the
 * transition table)
 */
template <typename OutSymbolT, int32_t NUM_SYMBOLS, int32_t TT_NUM_STATES>
class Dfa {
 public:
  // The maximum number of states supported by this DFA instance
  // This is a value queried by the DFA simulation algorithm
  static constexpr int32_t MAX_NUM_STATES = TT_NUM_STATES;

 private:
  // Symbol-group id lookup table
  using MatcherT     = detail::SingleSymbolSmemLUT<char>;
  using MatcherInitT = typename MatcherT::KernelParameter;

  // Transition table
  using TransitionTableT     = detail::TransitionTable<NUM_SYMBOLS + 1, TT_NUM_STATES>;
  using TransitionTableInitT = typename TransitionTableT::KernelParameter;

  // Translation lookup table
  using OutSymbolOffsetT     = uint32_t;
  using TransducerTableT     = detail::TransducerLookupTable<OutSymbolT,
                                                         OutSymbolOffsetT,
                                                         NUM_SYMBOLS + 1,
                                                         TT_NUM_STATES,
                                                         (NUM_SYMBOLS + 1) * TT_NUM_STATES>;
  using TransducerTableInitT = typename TransducerTableT::KernelParameter;

  // Private members (passed between host/device)
  /// Information to initialize the device-side lookup table that maps symbol -> symbol group id
  MatcherInitT symbol_matcher_init;

  /// Information to initialize the device-side transition table
  TransitionTableInitT tt_init;

  /// Information to initialize the device-side translation table
  TransducerTableInitT tt_out_init;

 public:
  //---------------------------------------------------------------------
  // DEVICE-SIDE MEMBER FUNCTIONS
  //---------------------------------------------------------------------
  using SymbolGroupStorageT      = typename MatcherT::TempStorage;
  using TransitionTableStorageT  = typename TransitionTableT::TempStorage;
  using TranslationTableStorageT = typename TransducerTableT::TempStorage;

  __device__ auto InitSymbolGroupLUT(SymbolGroupStorageT& temp_storage)
  {
    return MatcherT(symbol_matcher_init, temp_storage);
  }

  __device__ auto InitTransitionTable(TransitionTableStorageT& temp_storage)
  {
    return TransitionTableT(tt_init, temp_storage);
  }

  __device__ auto InitTranslationTable(TranslationTableStorageT& temp_storage)
  {
    return TransducerTableT(tt_out_init, temp_storage);
  }

  //---------------------------------------------------------------------
  // HOST-SIDE MEMBER FUNCTIONS
  //---------------------------------------------------------------------
  template <typename StateIdT, typename SymbolGroupIdItT>
  cudaError_t Init(SymbolGroupIdItT const& symbol_vec,
                   std::vector<std::vector<StateIdT>> const& tt_vec,
                   std::vector<std::vector<std::vector<OutSymbolT>>> const& out_tt_vec,
                   cudaStream_t stream = 0)
  {
    cudaError_t error = cudaSuccess;

    enum : uint32_t { MEM_SYMBOL_MATCHER = 0, MEM_TT, MEM_OUT_TT, NUM_ALLOCATIONS };

    size_t allocation_sizes[NUM_ALLOCATIONS] = {0};
    void* allocations[NUM_ALLOCATIONS]       = {0};

    // Memory requirements: lookup table
    error = MatcherT::PrepareLUT(
      nullptr, allocation_sizes[MEM_SYMBOL_MATCHER], symbol_vec, symbol_matcher_init);
    if (error) return error;

    // Memory requirements: transition table
    error =
      TransitionTableT::CreateTransitionTable(nullptr, allocation_sizes[MEM_TT], tt_vec, tt_init);
    if (error) return error;

    // Memory requirements: transducer table
    error = TransducerTableT::CreateTransitionTable(
      nullptr, allocation_sizes[MEM_OUT_TT], out_tt_vec, tt_out_init);
    if (error) return error;

    // Memory requirements: total memory
    size_t temp_storage_bytes = 0;
    error = cub::AliasTemporaries(nullptr, temp_storage_bytes, allocations, allocation_sizes);
    if (error) return error;

    // Allocate memory
    void* d_temp_storage = nullptr;
    error                = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (error) return error;

    // Alias memory
    error =
      cub::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes);
    if (error) return error;

    // Initialize symbol group lookup table
    error = MatcherT::PrepareLUT(allocations[MEM_SYMBOL_MATCHER],
                                 allocation_sizes[MEM_SYMBOL_MATCHER],
                                 symbol_vec,
                                 symbol_matcher_init,
                                 stream);
    if (error) return error;

    // Initialize state transition table
    error = TransitionTableT::CreateTransitionTable(
      allocations[MEM_TT], allocation_sizes[MEM_TT], tt_vec, tt_init, stream);
    if (error) return error;

    // Initialize finite-state transducer lookup table
    error = TransducerTableT::CreateTransitionTable(
      allocations[MEM_OUT_TT], allocation_sizes[MEM_OUT_TT], out_tt_vec, tt_out_init, stream);
    if (error) return error;

    return error;
  }

  template <typename SymbolT,
            typename TransducedOutItT,
            typename TransducedIndexOutItT,
            typename TransducedCountOutItT,
            typename OffsetT>
  cudaError_t Transduce(void* d_temp_storage,
                        size_t& temp_storage_bytes,
                        SymbolT const* d_chars,
                        OffsetT num_chars,
                        TransducedOutItT d_out_it,
                        TransducedIndexOutItT d_out_idx_it,
                        TransducedCountOutItT d_num_transduced_out_it,
                        const uint32_t seed_state = 0,
                        cudaStream_t stream       = 0)
  {
    return DeviceTransduce(d_temp_storage,
                           temp_storage_bytes,
                           *this,
                           d_chars,
                           num_chars,
                           d_out_it,
                           d_out_idx_it,
                           d_num_transduced_out_it,
                           seed_state,
                           stream);
  }
};

}  // namespace fst
}  // namespace io
}  // namespace cudf
