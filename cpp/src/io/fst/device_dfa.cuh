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

#include "dispatch_dfa.cuh"

#include <io/utilities/hostdevice_vector.hpp>
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

template <typename SymbolGroupIdLookupT,
          typename TransitionTableT,
          typename TranslationTableT,
          int32_t NUM_STATES>
class dfa_device_view {
 private:
  using sgid_lut_init_t          = typename SymbolGroupIdLookupT::KernelParameter;
  using transition_table_init_t  = typename TransitionTableT::KernelParameter;
  using translation_table_init_t = typename TranslationTableT::KernelParameter;

 public:
  // The maximum number of states supported by this DFA instance
  // This is a value queried by the DFA simulation algorithm
  static constexpr int32_t MAX_NUM_STATES = NUM_STATES;

  //---------------------------------------------------------------------
  // DEVICE-SIDE MEMBER FUNCTIONS
  //---------------------------------------------------------------------
  using SymbolGroupStorageT      = typename SymbolGroupIdLookupT::TempStorage;
  using TransitionTableStorageT  = typename TransitionTableT::TempStorage;
  using TranslationTableStorageT = typename TranslationTableT::TempStorage;

  __device__ auto InitSymbolGroupLUT(SymbolGroupStorageT& temp_storage)
  {
    return SymbolGroupIdLookupT(*d_sgid_lut_init, temp_storage);
  }

  __device__ auto InitTransitionTable(TransitionTableStorageT& temp_storage)
  {
    return TransitionTableT(*d_transition_table_init, temp_storage);
  }

  __device__ auto InitTranslationTable(TranslationTableStorageT& temp_storage)
  {
    return TranslationTableT(*d_translation_table_init, temp_storage);
  }

  dfa_device_view(sgid_lut_init_t const* d_sgid_lut_init,
                  transition_table_init_t const* d_transition_table_init,
                  translation_table_init_t const* d_translation_table_init)
    : d_sgid_lut_init(d_sgid_lut_init),
      d_transition_table_init(d_transition_table_init),
      d_translation_table_init(d_translation_table_init)
  {
  }

 private:
  sgid_lut_init_t const* d_sgid_lut_init;
  transition_table_init_t const* d_transition_table_init;
  translation_table_init_t const* d_translation_table_init;
};

/**
 * @brief Helper class to facilitate the specification and instantiation of a DFA (i.e., the
 * transition table and its number of states, the mapping of symbols to symbol groups, and the
 * translation table that specifies which state transitions cause which output to be written).
 *
 * @tparam OutSymbolT The symbol type being output by the finite-state transducer
 * @tparam NUM_SYMBOLS The number of symbol groups amongst which to differentiate (one dimension of
 * the transition table)
 * @tparam NUM_STATES The number of states defined by the DFA (the other dimension of the
 * transition table)
 */
template <typename OutSymbolT, int32_t NUM_SYMBOLS, int32_t NUM_STATES>
class Dfa {
 public:
  // The maximum number of states supported by this DFA instance
  // This is a value queried by the DFA simulation algorithm
  static constexpr int32_t MAX_NUM_STATES = NUM_STATES;

 private:
  // Symbol-group id lookup table
  using SymbolGroupIdLookupT = detail::SingleSymbolSmemLUT<char>;
  using SymbolGroupIdInitT   = typename SymbolGroupIdLookupT::KernelParameter;

  // Transition table
  using TransitionTableT     = detail::TransitionTable<NUM_SYMBOLS + 1, NUM_STATES>;
  using TransitionTableInitT = typename TransitionTableT::KernelParameter;

  // Translation lookup table
  using OutSymbolOffsetT      = uint32_t;
  using TranslationTableT     = detail::TransducerLookupTable<OutSymbolT,
                                                          OutSymbolOffsetT,
                                                          NUM_SYMBOLS + 1,
                                                          NUM_STATES,
                                                          (NUM_SYMBOLS + 1) * NUM_STATES>;
  using TranslationTableInitT = typename TranslationTableT::KernelParameter;

  auto get_device_view()
  {
    return dfa_device_view<SymbolGroupIdLookupT, TransitionTableT, TranslationTableT, NUM_STATES>{
      sgid_init.d_begin(), transition_table_init.d_begin(), translation_table_init.d_begin()};
  }

 public:
  template <typename StateIdT, typename SymbolGroupIdItT>
  Dfa(SymbolGroupIdItT const& symbol_vec,
      std::vector<std::vector<StateIdT>> const& tt_vec,
      std::vector<std::vector<std::vector<OutSymbolT>>> const& out_tt_vec,
      cudaStream_t stream)
  {
    constexpr std::size_t single_item = 1;

    sgid_init              = hostdevice_vector<SymbolGroupIdInitT>{single_item, stream};
    transition_table_init  = hostdevice_vector<TransitionTableInitT>{single_item, stream};
    translation_table_init = hostdevice_vector<TranslationTableInitT>{single_item, stream};

    // Initialize symbol group id lookup table
    SymbolGroupIdLookupT::InitDeviceSymbolGroupIdLut(sgid_init, symbol_vec, stream);

    // Initialize state transition table
    TransitionTableT::InitDeviceTransitionTable(transition_table_init, tt_vec, stream);

    // Initialize finite-state transducer lookup table
    TranslationTableT::InitDeviceTranslationTable(translation_table_init, out_tt_vec, stream);
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
                           this->get_device_view(),
                           d_chars,
                           num_chars,
                           d_out_it,
                           d_out_idx_it,
                           d_num_transduced_out_it,
                           seed_state,
                           stream);
  }

 private:
  hostdevice_vector<SymbolGroupIdInitT> sgid_init{};
  hostdevice_vector<TransitionTableInitT> transition_table_init{};
  hostdevice_vector<TranslationTableInitT> translation_table_init{};
};
}  // namespace fst
}  // namespace io
}  // namespace cudf
