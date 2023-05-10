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

#include "agent_dfa.cuh"
#include "in_reg_array.cuh"

#include <cub/cub.cuh>

#include <cstdint>

namespace cudf::io::fst::detail {

/**
 * @brief The tuning policy comprising all the architecture-specific compile-time tuning parameters.
 *
 * @tparam _BLOCK_THREADS Number of threads per block
 * @tparam _ITEMS_PER_THREAD Number of symbols processed by each thread
 */
template <int32_t _BLOCK_THREADS, int32_t _ITEMS_PER_THREAD>
struct AgentDFAPolicy {
  // The number of threads per block
  static constexpr int32_t BLOCK_THREADS = _BLOCK_THREADS;

  // The number of symbols processed by each thread
  static constexpr int32_t ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
};

/**
 * @brief The list of architecture-specific tuning policies. Yet TBD.
 */
struct DeviceFSMPolicy {
  //------------------------------------------------------------------------------
  // Architecture-specific tuning policies
  //------------------------------------------------------------------------------
  struct Policy900 : cub::ChainedPolicy<900, Policy900, Policy900> {
    enum {
      BLOCK_THREADS    = 128,
      ITEMS_PER_THREAD = 32,
    };

    using AgentDFAPolicy = AgentDFAPolicy<BLOCK_THREADS, ITEMS_PER_THREAD>;
  };

  // Top-of-list of the tuning policy "chain"
  using MaxPolicy = Policy900;
};

/**
 * @brief Kernel for initializing single-pass prefix scan tile states
 *
 * @param items_state The tile state
 * @param num_tiles The number of tiles to be initialized
 * @return
 */
template <typename TileState>
__global__ void initialization_pass_kernel(TileState items_state, uint32_t num_tiles)
{
  items_state.InitializeStatus(num_tiles);
}

template <typename DfaT,
          typename SymbolItT,
          typename TransducedOutItT,
          typename TransducedIndexOutItT,
          typename TransducedCountOutItT,
          typename OffsetT>
struct DispatchFSM : DeviceFSMPolicy {
  //------------------------------------------------------------------------------
  // DEFAULT TYPES
  //------------------------------------------------------------------------------
  using StateIndexT  = uint32_t;
  using BlockOffsetT = uint32_t;

  //------------------------------------------------------------------------------
  // DERIVED CONFIGS
  //------------------------------------------------------------------------------
  // DFA-specific configs
  static constexpr int32_t MAX_NUM_STATES  = DfaT::MAX_NUM_STATES;
  static constexpr int32_t MAX_NUM_SYMBOLS = DfaT::MAX_NUM_SYMBOLS;

  // Whether to use a single-pass prefix scan that does all in on
  static constexpr bool SINGLE_PASS_STV = false;

  // Whether this is a finite-state transform
  static constexpr bool IS_FST = true;

  //------------------------------------------------------------------------------
  // TYPEDEFS
  //------------------------------------------------------------------------------
  using StateVectorCompositeOpT = VectorCompositeOp<MAX_NUM_STATES>;

  //------------------------------------------------------------------------------
  // MEMBER VARS
  //------------------------------------------------------------------------------
  void* d_temp_storage;
  size_t& temp_storage_bytes;
  DfaT dfa;
  StateIndexT seed_state;
  SymbolItT d_chars_in;
  OffsetT num_chars;
  TransducedOutItT transduced_out_it;
  TransducedIndexOutItT transduced_out_idx_it;
  TransducedCountOutItT d_num_transduced_out_it;
  cudaStream_t stream;
  int const ptx_version;

  //------------------------------------------------------------------------------
  // CONSTRUCTOR
  //------------------------------------------------------------------------------
  CUB_RUNTIME_FUNCTION __forceinline__ DispatchFSM(void* d_temp_storage,
                                                   size_t& temp_storage_bytes,
                                                   DfaT dfa,
                                                   StateIndexT seed_state,
                                                   SymbolItT d_chars_in,
                                                   OffsetT num_chars,
                                                   TransducedOutItT transduced_out_it,
                                                   TransducedIndexOutItT transduced_out_idx_it,
                                                   TransducedCountOutItT d_num_transduced_out_it,
                                                   cudaStream_t stream,
                                                   int ptx_version)
    : d_temp_storage(d_temp_storage),
      temp_storage_bytes(temp_storage_bytes),
      dfa(dfa),
      seed_state(seed_state),
      d_chars_in(d_chars_in),
      num_chars(num_chars),
      transduced_out_it(transduced_out_it),
      transduced_out_idx_it(transduced_out_idx_it),
      d_num_transduced_out_it(d_num_transduced_out_it),
      stream(stream),
      ptx_version(ptx_version)
  {
  }

  //------------------------------------------------------------------------------
  // DISPATCH INTERFACE
  //------------------------------------------------------------------------------
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DfaT dfa,
    StateIndexT seed_state,
    SymbolItT d_chars_in,
    OffsetT num_chars,
    TransducedOutItT transduced_out_it,
    TransducedIndexOutItT transduced_out_idx_it,
    TransducedCountOutItT d_num_transduced_out_it,
    cudaStream_t stream)
  {
    using MaxPolicyT = DispatchFSM::MaxPolicy;

    cudaError_t error;

    // Get PTX version
    int ptx_version;
    error = cub::PtxVersion(ptx_version);
    if (error != cudaSuccess) return error;

    // Create dispatch functor
    DispatchFSM dispatch(d_temp_storage,
                         temp_storage_bytes,
                         dfa,
                         seed_state,
                         d_chars_in,
                         num_chars,
                         transduced_out_it,
                         transduced_out_idx_it,
                         d_num_transduced_out_it,
                         stream,
                         ptx_version);

    error = MaxPolicyT::Invoke(ptx_version, dispatch);
    return error;
  }

  //------------------------------------------------------------------------------
  // DFA SIMULATION KERNEL INVOCATION
  //------------------------------------------------------------------------------
  template <typename ActivePolicyT,
            typename DFASimulationKernelT,
            typename TileStateT,
            typename FstScanTileStateT,
            typename StateVectorT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t
  InvokeDFASimulationKernel(DFASimulationKernelT dfa_kernel,
                            int32_t sm_count,
                            StateIndexT seed_state,
                            StateVectorT* d_thread_state_transition,
                            TileStateT tile_state,
                            FstScanTileStateT fst_tile_state)

  {
    cudaError_t error = cudaSuccess;
    cub::KernelConfig dfa_simulation_config;

    using PolicyT = typename ActivePolicyT::AgentDFAPolicy;
    if (CubDebug(error = dfa_simulation_config.Init<PolicyT>(dfa_kernel))) return error;

    // Kernel invocation
    uint32_t grid_size = std::max(
      1u, CUB_QUOTIENT_CEILING(num_chars, PolicyT::BLOCK_THREADS * PolicyT::ITEMS_PER_THREAD));
    uint32_t block_threads = dfa_simulation_config.block_threads;

    dfa_kernel<<<grid_size, block_threads, 0, stream>>>(dfa,
                                                        d_chars_in,
                                                        num_chars,
                                                        seed_state,
                                                        d_thread_state_transition,
                                                        tile_state,
                                                        fst_tile_state,
                                                        transduced_out_it,
                                                        transduced_out_idx_it,
                                                        d_num_transduced_out_it);

    // Check for errors
    if (CubDebug(error = cudaPeekAtLastError())) return error;

    return error;
  }

  /**
   * @brief Computes the state-transition vectors
   */
  template <typename ActivePolicyT,
            typename TileStateT,
            typename FstScanTileStateT,
            typename StateVectorT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t
  ComputeStateTransitionVector(uint32_t sm_count,
                               TileStateT tile_state,
                               FstScanTileStateT fst_tile_state,
                               StateVectorT* d_thread_state_transition)
  {
    StateIndexT seed_state = 0;

    return InvokeDFASimulationKernel<ActivePolicyT>(
      SimulateDFAKernel<true,
                        SINGLE_PASS_STV,
                        DfaT,
                        TileStateT,
                        typename ActivePolicyT::AgentDFAPolicy,
                        SymbolItT,
                        OffsetT,
                        StateVectorT,
                        FstScanTileStateT,
                        TransducedOutItT,
                        TransducedIndexOutItT,
                        TransducedCountOutItT>,
      sm_count,
      seed_state,
      d_thread_state_transition,
      tile_state,
      fst_tile_state);
  }

  /**
   * @brief Performs the actual DFA simulation.
   */
  template <typename ActivePolicyT,
            typename TileStateT,
            typename FstScanTileStateT,
            typename StateVectorT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t
  SimulateDFA(uint32_t sm_count,
              TileStateT tile_state,
              FstScanTileStateT fst_tile_state,
              StateIndexT seed_state,
              StateVectorT* d_thread_state_transition)
  {
    return InvokeDFASimulationKernel<ActivePolicyT>(
      SimulateDFAKernel<false,
                        SINGLE_PASS_STV,
                        DfaT,
                        TileStateT,
                        typename ActivePolicyT::AgentDFAPolicy,
                        SymbolItT,
                        OffsetT,
                        StateVectorT,
                        FstScanTileStateT,
                        TransducedOutItT,
                        TransducedIndexOutItT,
                        TransducedCountOutItT>,
      sm_count,
      seed_state,
      d_thread_state_transition,
      tile_state,
      fst_tile_state);
  }

  //------------------------------------------------------------------------------
  // POLICY INVOCATION
  //------------------------------------------------------------------------------
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t Invoke()
  {
    cudaError_t error = cudaSuccess;

    // Get SM count
    int device_ordinal = -1;
    int sm_count       = -1;

    // Get current device
    error = cudaGetDevice(&device_ordinal);
    if (error != cudaSuccess) return error;

    error = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal);
    if (error != cudaSuccess) return error;

    //------------------------------------------------------------------------------
    // DERIVED TYPEDEFS
    //------------------------------------------------------------------------------
    // Type used to represent state-transition vectors
    using StateVectorT = MultiFragmentInRegArray<MAX_NUM_STATES, MAX_NUM_STATES - 1>;

    // Scan tile state used for propagating composed state transition vectors
    using ScanTileStateT = typename cub::ScanTileState<StateVectorT>;

    // Scan tile state used for propagating transduced output offsets
    using FstScanTileStateT = typename cub::ScanTileState<OffsetT>;

    // STATE-TRANSITION IDENTITY VECTOR
    StateVectorT state_identity_vector;
    for (int32_t i = 0; i < MAX_NUM_STATES; ++i) {
      state_identity_vector.Set(i, i);
    }
    StateVectorCompositeOpT state_vector_scan_op;

    //------------------------------------------------------------------------------
    // DERIVED CONFIGS
    //------------------------------------------------------------------------------
    enum {
      BLOCK_THREADS         = ActivePolicyT::BLOCK_THREADS,
      SYMBOLS_PER_THREAD    = ActivePolicyT::ITEMS_PER_THREAD,
      NUM_SYMBOLS_PER_BLOCK = BLOCK_THREADS * SYMBOLS_PER_THREAD
    };

    BlockOffsetT num_blocks = std::max(1u, CUB_QUOTIENT_CEILING(num_chars, NUM_SYMBOLS_PER_BLOCK));
    size_t num_threads      = num_blocks * BLOCK_THREADS;

    //------------------------------------------------------------------------------
    // TEMPORARY MEMORY REQUIREMENTS
    //------------------------------------------------------------------------------
    enum { MEM_STATE_VECTORS = 0, MEM_SCAN, MEM_SINGLE_PASS_STV, MEM_FST_OFFSET, NUM_ALLOCATIONS };

    size_t allocation_sizes[NUM_ALLOCATIONS] = {0};
    void* allocations[NUM_ALLOCATIONS]       = {0};

    size_t vector_scan_storage_bytes = 0;

    // [MEMORY REQUIREMENTS] STATE-TRANSITION SCAN
    cub::DeviceScan::ExclusiveScan(nullptr,
                                   vector_scan_storage_bytes,
                                   static_cast<StateVectorT*>(allocations[MEM_STATE_VECTORS]),
                                   static_cast<StateVectorT*>(allocations[MEM_STATE_VECTORS]),
                                   state_vector_scan_op,
                                   state_identity_vector,
                                   num_threads,
                                   stream);

    allocation_sizes[MEM_STATE_VECTORS] = num_threads * sizeof(StateVectorT);
    allocation_sizes[MEM_SCAN]          = vector_scan_storage_bytes;

    // Bytes needed for tile status descriptors (fusing state-transition vector + DFA simulation)
    if constexpr (SINGLE_PASS_STV) {
      error = ScanTileStateT::AllocationSize(num_blocks, allocation_sizes[MEM_SINGLE_PASS_STV]);
      if (error != cudaSuccess) return error;
    }

    // Bytes needed for tile status descriptors (DFA simulation pass for output size computation +
    // output-generating pass)
    if constexpr (IS_FST) {
      error = FstScanTileStateT::AllocationSize(num_blocks, allocation_sizes[MEM_FST_OFFSET]);
      if (error != cudaSuccess) return error;
    }

    // Alias the temporary allocations from the single storage blob (or compute the necessary size
    // of the blob)
    error =
      cub::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes);
    if (error != cudaSuccess) return error;

    // Return if the caller is simply requesting the size of the storage allocation
    if (d_temp_storage == NULL) return cudaSuccess;

    // Alias memory for state-transition vectors
    StateVectorT* d_thread_state_transition =
      static_cast<StateVectorT*>(allocations[MEM_STATE_VECTORS]);

    //------------------------------------------------------------------------------
    // INITIALIZE SCAN TILE STATES COMPUTING TRANSDUCED OUTPUT OFFSETS
    //------------------------------------------------------------------------------
    FstScanTileStateT fst_offset_tile_state;
    if constexpr (IS_FST) {
      // Construct the tile status (aliases memory internally et al.)
      error = fst_offset_tile_state.Init(
        num_blocks, allocations[MEM_FST_OFFSET], allocation_sizes[MEM_FST_OFFSET]);
      if (error != cudaSuccess) return error;
      constexpr uint32_t FST_INIT_TPB = 256;
      uint32_t num_fst_init_blocks    = CUB_QUOTIENT_CEILING(num_blocks, FST_INIT_TPB);
      initialization_pass_kernel<<<num_fst_init_blocks, FST_INIT_TPB, 0, stream>>>(
        fst_offset_tile_state, num_blocks);
    }

    //------------------------------------------------------------------------------
    // COMPUTE STATE-TRANSITION VECTORS
    //------------------------------------------------------------------------------
    ScanTileStateT stv_tile_state;
    if constexpr (SINGLE_PASS_STV) {
      // Construct the tile status (aliases memory internally et al.)
      error = stv_tile_state.Init(
        num_blocks, allocations[MEM_SINGLE_PASS_STV], allocation_sizes[MEM_SINGLE_PASS_STV]);
      if (error != cudaSuccess) return error;
      constexpr uint32_t STV_INIT_TPB = 256;
      uint32_t num_stv_init_blocks    = CUB_QUOTIENT_CEILING(num_blocks, STV_INIT_TPB);
      initialization_pass_kernel<<<num_stv_init_blocks, STV_INIT_TPB, 0, stream>>>(stv_tile_state,
                                                                                   num_blocks);
    } else {
      // Compute state-transition vectors
      // TODO tag dispatch or constexpr if depending on single-pass config to avoid superfluous
      // template instantiations
      ComputeStateTransitionVector<ActivePolicyT>(
        sm_count, stv_tile_state, fst_offset_tile_state, d_thread_state_transition);

      // State-transition vector scan computing using the composition operator
      cub::DeviceScan::ExclusiveScan(allocations[MEM_SCAN],
                                     allocation_sizes[MEM_SCAN],
                                     d_thread_state_transition,
                                     d_thread_state_transition,
                                     state_vector_scan_op,
                                     state_identity_vector,
                                     num_threads,
                                     stream);
    }

    //------------------------------------------------------------------------------
    // SIMULATE DFA
    //------------------------------------------------------------------------------
    return SimulateDFA<ActivePolicyT>(
      sm_count, stv_tile_state, fst_offset_tile_state, seed_state, d_thread_state_transition);
  }
};
}  // namespace cudf::io::fst::detail
