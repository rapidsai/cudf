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

#pragma once

#include <cudf/types.hpp>
#include <io/fst/device_dfa.cuh>
#include <io/utilities/hostdevice_vector.hpp>

#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cudf::io::fst::detail {

/**
 * @brief Class template that can be plugged into the finite-state machine to look up the symbol
 * group index for a given symbol. Class template does not support multi-symbol lookups (i.e., no
 * look-ahead). The class uses shared memory for the lookups.
 *
 * @tparam SymbolT The symbol type being passed in to lookup the corresponding symbol group id
 */
template <typename SymbolT>
class SingleSymbolSmemLUT {
 private:
  // Type used for representing a symbol group id (i.e., what we return for a given symbol)
  using SymbolGroupIdT = uint8_t;

  // Number of entries for every lookup (e.g., for 8-bit Symbol this is 256)
  static constexpr uint32_t NUM_ENTRIES_PER_LUT = 0x01U << (sizeof(SymbolT) * 8U);

  struct _TempStorage {
    // sym_to_sgid[symbol] -> symbol group index
    SymbolGroupIdT sym_to_sgid[NUM_ENTRIES_PER_LUT];
  };

 public:
  struct KernelParameter {
    // sym_to_sgid[min(symbol,num_valid_entries)] -> symbol group index
    SymbolT num_valid_entries;

    // sym_to_sgid[symbol] -> symbol group index
    SymbolGroupIdT sym_to_sgid[NUM_ENTRIES_PER_LUT];
  };

  using TempStorage = cub::Uninitialized<_TempStorage>;

  /**
   * @brief Initializes the given \p sgid_init with the symbol group lookups defined by \p
   * symbol_strings.
   *
   * @param[out] sgid_init A hostdevice_vector that will be populated
   * @param[in] symbol_strings Array of strings, where the i-th string holds all symbols
   * (characters!) that correspond to the i-th symbol group index
   * @param[in] stream The stream that shall be used to cudaMemcpyAsync the lookup table
   * @return
   */
  template <typename SymbolGroupItT>
  static void InitDeviceSymbolGroupIdLut(
    cudf::detail::hostdevice_vector<KernelParameter>& sgid_init,
    SymbolGroupItT const& symbol_strings,
    rmm::cuda_stream_view stream)
  {
    // The symbol group index to be returned if none of the given symbols match
    SymbolGroupIdT no_match_id = symbol_strings.size();

    // The symbol with the largest value that is mapped to a symbol group id
    SymbolGroupIdT max_base_match_val = 0;

    // Initialize all entries: by default we return the no-match-id
    std::fill(&sgid_init.host_ptr()->sym_to_sgid[0],
              &sgid_init.host_ptr()->sym_to_sgid[NUM_ENTRIES_PER_LUT],
              no_match_id);

    // Set up lookup table
    uint32_t sg_id = 0;
    // Iterate over the symbol groups
    for (auto const& sg_symbols : symbol_strings) {
      // Iterate over all symbols that belong to the current symbol group
      for (auto const& sg_symbol : sg_symbols) {
        max_base_match_val = std::max(max_base_match_val, static_cast<SymbolGroupIdT>(sg_symbol));
        sgid_init.host_ptr()->sym_to_sgid[static_cast<int32_t>(sg_symbol)] = sg_id;
      }
      sg_id++;
    }

    // Initialize the out-of-bounds lookup: sym_to_sgid[max_base_match_val+1] -> no_match_id
    sgid_init.host_ptr()->sym_to_sgid[max_base_match_val + 1] = no_match_id;

    // Alias memory / return memory requirements
    sgid_init.host_ptr()->num_valid_entries = max_base_match_val + 1;

    sgid_init.host_to_device_async(stream);
  }

  _TempStorage& temp_storage;
  SymbolGroupIdT num_valid_entries;

  __device__ __forceinline__ _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  /**
   * @brief Initializes the lookup table, primarily to be invoked from within device code but also
   * provides host-side implementation for verification.
   * @note Synchronizes the thread block, if called from device, and, hence, requires all threads
   * of the thread block to call the constructor
   */
  constexpr CUDF_HOST_DEVICE SingleSymbolSmemLUT(KernelParameter const& kernel_param,
                                                 TempStorage& temp_storage)
    : temp_storage(temp_storage.Alias()), num_valid_entries(kernel_param.num_valid_entries)
  {
    // GPU-side init
#if CUB_PTX_ARCH > 0
    for (int32_t i = threadIdx.x; i < kernel_param.num_valid_entries; i += blockDim.x) {
      this->temp_storage.sym_to_sgid[i] = kernel_param.sym_to_sgid[i];
    }
    __syncthreads();

#else
    // CPU-side init
    std::copy_n(kernel_param.sym_to_sgid, kernel_param.num_luts, this->temp_storage.sym_to_sgid);
#endif
  }

  constexpr CUDF_HOST_DEVICE int32_t operator()(SymbolT const symbol) const
  {
    // Look up the symbol group for given symbol
    return temp_storage
      .sym_to_sgid[min(static_cast<SymbolGroupIdT>(symbol), num_valid_entries - 1U)];
  }
};

/**
 * @brief Lookup table mapping (old_state, symbol_group_id) transitions to a new target state. The
 * class uses shared memory for the lookups.
 *
 * @tparam MAX_NUM_SYMBOLS The maximum number of symbols being output by a single state transition
 * @tparam MAX_NUM_STATES The maximum number of states that this lookup table shall support
 */
template <int32_t MAX_NUM_SYMBOLS, int32_t MAX_NUM_STATES>
class TransitionTable {
 private:
  // Type used
  using ItemT = char;

  struct _TempStorage {
    ItemT transitions[MAX_NUM_STATES * MAX_NUM_SYMBOLS];
  };

 public:
  using TempStorage = cub::Uninitialized<_TempStorage>;

  struct KernelParameter {
    ItemT transitions[MAX_NUM_STATES * MAX_NUM_SYMBOLS];
  };

  template <typename StateIdT>
  static void InitDeviceTransitionTable(
    cudf::detail::hostdevice_vector<KernelParameter>& transition_table_init,
    std::array<std::array<StateIdT, MAX_NUM_SYMBOLS>, MAX_NUM_STATES> const& translation_table,
    rmm::cuda_stream_view stream)
  {
    // translation_table[state][symbol] -> new state
    for (std::size_t state = 0; state < translation_table.size(); ++state) {
      for (std::size_t symbol = 0; symbol < translation_table[state].size(); ++symbol) {
        CUDF_EXPECTS(
          static_cast<int64_t>(translation_table[state][symbol]) <=
            std::numeric_limits<ItemT>::max(),
          "Target state index value exceeds value representable by the transition table's type");
        transition_table_init.host_ptr()->transitions[symbol * MAX_NUM_STATES + state] =
          static_cast<ItemT>(translation_table[state][symbol]);
      }
    }

    // Copy transition table to device
    transition_table_init.host_to_device_async(stream);
  }

  constexpr CUDF_HOST_DEVICE TransitionTable(KernelParameter const& kernel_param,
                                             TempStorage& temp_storage)
    : temp_storage(temp_storage.Alias())
  {
#if CUB_PTX_ARCH > 0
    for (int i = threadIdx.x; i < MAX_NUM_STATES * MAX_NUM_SYMBOLS; i += blockDim.x) {
      this->temp_storage.transitions[i] = kernel_param.transitions[i];
    }
    __syncthreads();
#else
    std::copy_n(
      kernel_param.transitions, MAX_NUM_STATES * MAX_NUM_SYMBOLS, this->temp_storage.transitions);
#endif
  }

  /**
   * @brief Returns a random-access iterator to lookup all the state transitions for one specific
   * symbol from an arbitrary old_state, i.e., it[old_state] -> new_state.
   *
   * @param state_id The DFA's current state index from which we'll transition
   * @param match_id The symbol group id of the symbol that we just read in
   * @return
   */
  template <typename StateIndexT, typename SymbolIndexT>
  constexpr CUDF_HOST_DEVICE int32_t operator()(StateIndexT const state_id,
                                                SymbolIndexT const match_id) const
  {
    return temp_storage.transitions[match_id * MAX_NUM_STATES + state_id];
  }

 private:
  _TempStorage& temp_storage;

  __device__ __forceinline__ _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;

    return private_storage;
  }
};

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
 * @brief Lookup table mapping (old_state, symbol_group_id) transitions to a sequence of symbols
 * that the finite-state transducer is supposed to output for each transition. The class uses shared
 * memory for the lookups.
 *
 * @tparam OutSymbolT The symbol type being output
 * @tparam OutSymbolOffsetT Type sufficiently large to index into the lookup table of output symbols
 * @tparam MAX_NUM_SYMBOLS The maximum number of symbols being output by a single state transition
 * @tparam MAX_NUM_STATES The maximum number of states that this lookup table shall support
 * @tparam MAX_TABLE_SIZE The maximum number of items in the lookup table of output symbols
 */
template <typename OutSymbolT,
          typename OutSymbolOffsetT,
          int32_t MAX_NUM_SYMBOLS,
          int32_t MAX_NUM_STATES,
          int32_t MAX_TABLE_SIZE = (MAX_NUM_SYMBOLS * MAX_NUM_STATES)>
class TransducerLookupTable {
 private:
  struct _TempStorage {
    OutSymbolOffsetT out_offset[MAX_NUM_STATES * MAX_NUM_SYMBOLS + 1];
    OutSymbolT out_symbols[MAX_TABLE_SIZE];
  };

 public:
  using TempStorage = cub::Uninitialized<_TempStorage>;

  struct KernelParameter {
    OutSymbolOffsetT d_out_offsets[MAX_NUM_STATES * MAX_NUM_SYMBOLS + 1];
    OutSymbolT d_out_symbols[MAX_TABLE_SIZE];
  };

  /**
   * @brief Initializes the lookup table, primarily to be invoked from within device code but also
   * provides host-side implementation for verification.
   * @note Synchronizes the thread block, if called from device, and, hence, requires all threads
   * of the thread block to call the constructor
   */
  static void InitDeviceTranslationTable(
    cudf::detail::hostdevice_vector<KernelParameter>& translation_table_init,
    std::array<std::array<std::vector<OutSymbolT>, MAX_NUM_SYMBOLS>, MAX_NUM_STATES> const&
      translation_table,
    rmm::cuda_stream_view stream)
  {
    std::vector<OutSymbolT> out_symbols;
    out_symbols.reserve(MAX_TABLE_SIZE);
    std::vector<OutSymbolOffsetT> out_symbol_offsets;
    out_symbol_offsets.reserve(MAX_NUM_STATES * MAX_NUM_SYMBOLS + 1);
    out_symbol_offsets.push_back(0);

    // Iterate over the states in the transition table
    for (auto const& state_trans : translation_table) {
      uint32_t num_added = 0;
      // Iterate over the symbols in the transition table
      for (auto const& symbol_out : state_trans) {
        // Insert the output symbols for this specific (state, symbol) transition
        out_symbols.insert(std::end(out_symbols), std::begin(symbol_out), std::end(symbol_out));
        out_symbol_offsets.push_back(out_symbols.size());
        num_added++;
      }

      // Copy the last offset for all symbols (to guarantee a proper lookup for omitted symbols of
      // this state)
      if (MAX_NUM_SYMBOLS > num_added) {
        int32_t count = MAX_NUM_SYMBOLS - num_added;
        auto begin_it = std::prev(std::end(out_symbol_offsets));
        std::fill_n(begin_it, count, out_symbol_offsets[0]);
      }
    }

    // Check whether runtime-provided table size exceeds the compile-time given max. table size
    CUDF_EXPECTS(out_symbols.size() <= MAX_TABLE_SIZE, "Unsupported translation table");

    // Prepare host-side data to be copied and passed to the device
    std::copy(std::cbegin(out_symbol_offsets),
              std::cend(out_symbol_offsets),
              translation_table_init.host_ptr()->d_out_offsets);
    std::copy(std::cbegin(out_symbols),
              std::cend(out_symbols),
              translation_table_init.host_ptr()->d_out_symbols);

    // Copy data to device
    translation_table_init.host_to_device_async(stream);
  }

 private:
  _TempStorage& temp_storage;

  __device__ __forceinline__ _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

 public:
  /**
   * @brief Initializes the lookup table, primarily to be invoked from within device code but also
   * provides host-side implementation for verification.
   * @note Synchronizes the thread block, if called from device, and, hence, requires all threads
   * of the thread block to call the constructor
   */
  CUDF_HOST_DEVICE TransducerLookupTable(KernelParameter const& kernel_param,
                                         TempStorage& temp_storage)
    : temp_storage(temp_storage.Alias())
  {
    constexpr uint32_t num_offsets = MAX_NUM_STATES * MAX_NUM_SYMBOLS + 1;
#if CUB_PTX_ARCH > 0
    for (int i = threadIdx.x; i < num_offsets; i += blockDim.x) {
      this->temp_storage.out_offset[i] = kernel_param.d_out_offsets[i];
    }
    // Make sure all threads in the block can read out_symbol_offsets[num_offsets - 1] from shared
    // memory
    __syncthreads();
    for (int i = threadIdx.x; i < this->temp_storage.out_offset[num_offsets - 1]; i += blockDim.x) {
      this->temp_storage.out_symbols[i] = kernel_param.d_out_symbols[i];
    }
    __syncthreads();
#else
    std::copy_n(kernel_param.d_out_offsets, num_offsets, this->temp_storage.out_symbol_offsets);
    std::copy_n(kernel_param.d_out_symbols,
                this->temp_storage.out_symbol_offsets,
                this->temp_storage.out_symbols);
#endif
  }

  template <typename StateIndexT, typename SymbolIndexT, typename RelativeOffsetT>
  constexpr CUDF_HOST_DEVICE OutSymbolT operator()(StateIndexT const state_id,
                                                   SymbolIndexT const match_id,
                                                   RelativeOffsetT const relative_offset) const
  {
    auto offset = temp_storage.out_offset[state_id * MAX_NUM_SYMBOLS + match_id] + relative_offset;
    return temp_storage.out_symbols[offset];
  }

  template <typename StateIndexT, typename SymbolIndexT>
  constexpr CUDF_HOST_DEVICE OutSymbolOffsetT operator()(StateIndexT const state_id,
                                                         SymbolIndexT const match_id) const
  {
    return temp_storage.out_offset[state_id * MAX_NUM_SYMBOLS + match_id + 1] -
           temp_storage.out_offset[state_id * MAX_NUM_SYMBOLS + match_id];
  }
};

/**
 * @brief Helper class to facilitate the specification and instantiation of a DFA (i.e., the
 * transition table and its number of states, the mapping of symbols to symbol groups, and the
 * translation table that specifies which state transitions cause which output to be written).
 *
 * @tparam OutSymbolT The symbol type being output by the finite-state transducer
 * @tparam NUM_SYMBOLS The number of symbol groups amongst which to differentiate including the
 * wildcard symbol group (one dimension of the transition table)
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
  using TransitionTableT     = detail::TransitionTable<NUM_SYMBOLS, NUM_STATES>;
  using TransitionTableInitT = typename TransitionTableT::KernelParameter;

  // Translation lookup table
  using OutSymbolOffsetT      = uint32_t;
  using TranslationTableT     = detail::TransducerLookupTable<OutSymbolT,
                                                          OutSymbolOffsetT,
                                                          NUM_SYMBOLS,
                                                          NUM_STATES,
                                                          NUM_SYMBOLS * NUM_STATES>;
  using TranslationTableInitT = typename TranslationTableT::KernelParameter;

  auto get_device_view()
  {
    return dfa_device_view<SymbolGroupIdLookupT, TransitionTableT, TranslationTableT, NUM_STATES>{
      sgid_init.d_begin(), transition_table_init.d_begin(), translation_table_init.d_begin()};
  }

 public:
  /**
   * @brief Constructs a new DFA.
   *
   * @param symbol_vec Sequence container of symbol groups. Each symbol group is a sequence
   * container to symbols within that group. The index of the symbol group containing a symbol being
   * read will be used as symbol_gid of the transition and translation tables.
   * @param tt_vec The transition table
   * @param out_tt_vec The translation table
   * @param stream The stream to which memory operations and kernels are getting dispatched to
   */
  template <typename StateIdT, typename SymbolGroupIdItT>
  Dfa(SymbolGroupIdItT const& symbol_vec,
      std::array<std::array<StateIdT, NUM_SYMBOLS>, NUM_STATES> const& tt_vec,
      std::array<std::array<std::vector<OutSymbolT>, NUM_SYMBOLS>, NUM_STATES> const& out_tt_vec,
      cudaStream_t stream)
  {
    constexpr std::size_t single_item = 1;

    sgid_init = cudf::detail::hostdevice_vector<SymbolGroupIdInitT>{single_item, stream};
    transition_table_init =
      cudf::detail::hostdevice_vector<TransitionTableInitT>{single_item, stream};
    translation_table_init =
      cudf::detail::hostdevice_vector<TranslationTableInitT>{single_item, stream};

    // Initialize symbol group id lookup table
    SymbolGroupIdLookupT::InitDeviceSymbolGroupIdLut(sgid_init, symbol_vec, stream);

    // Initialize state transition table
    TransitionTableT::InitDeviceTransitionTable(transition_table_init, tt_vec, stream);

    // Initialize finite-state transducer lookup table
    TranslationTableT::InitDeviceTranslationTable(translation_table_init, out_tt_vec, stream);
  }

  /**
   * @brief Dispatches the finite-state transducer algorithm to the GPU.
   *
   * @tparam SymbolT The atomic symbol type from the input tape
   * @tparam TransducedOutItT Random-access output iterator to which the transduced output will be
   * written
   * @tparam TransducedIndexOutItT Random-access output iterator type to which the input symbols'
   * indexes are written.
   * @tparam TransducedCountOutItT A single-item output iterator type to which the total number of
   * output symbols is written
   * @tparam OffsetT A type large enough to index into either of both: (a) the input symbols and (b)
   * the output symbols
   * @param d_chars Pointer to the input string of symbols
   * @param num_chars The total number of input symbols to process
   * @param d_out_it Random-access output iterator to which the transduced output is
   * written
   * @param d_out_idx_it Random-access output iterator to which, the index i is written
   * iff the i-th input symbol caused some output to be written
   * @param d_num_transduced_out_it A single-item output iterator type to which the total number
   * of output symbols is written
   * @param seed_state The DFA's starting state. For streaming DFAs this corresponds to the
   * "end-state" of the previous invocation of the algorithm.
   * @param stream CUDA stream to launch kernels within. Default is the null-stream.
   */
  template <typename SymbolT,
            typename TransducedOutItT,
            typename TransducedIndexOutItT,
            typename TransducedCountOutItT,
            typename OffsetT>
  void Transduce(SymbolT const* d_chars,
                 OffsetT num_chars,
                 TransducedOutItT d_out_it,
                 TransducedIndexOutItT d_out_idx_it,
                 TransducedCountOutItT d_num_transduced_out_it,
                 uint32_t const seed_state,
                 rmm::cuda_stream_view stream)
  {
    std::size_t temp_storage_bytes = 0;
    rmm::device_buffer temp_storage{};
    DeviceTransduce(nullptr,
                    temp_storage_bytes,
                    this->get_device_view(),
                    d_chars,
                    num_chars,
                    d_out_it,
                    d_out_idx_it,
                    d_num_transduced_out_it,
                    seed_state,
                    stream);

    if (temp_storage.size() < temp_storage_bytes) {
      temp_storage.resize(temp_storage_bytes, stream);
    }

    DeviceTransduce(temp_storage.data(),
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
  cudf::detail::hostdevice_vector<SymbolGroupIdInitT> sgid_init{};
  cudf::detail::hostdevice_vector<TransitionTableInitT> transition_table_init{};
  cudf::detail::hostdevice_vector<TranslationTableInitT> translation_table_init{};
};

}  // namespace cudf::io::fst::detail
