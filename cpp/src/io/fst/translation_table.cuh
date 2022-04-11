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

#include "in_reg_array.cuh"

#include <cub/cub.cuh>

#include <cstdint>

namespace cudf {
namespace io {
namespace fst {
namespace detail {

/**
 * @brief Lookup table mapping (old_state, symbol_group_id) transitions to a sequence of symbols to
 * output
 *
 * @tparam OutSymbolT The symbol type being returned
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
struct TransducerLookupTable {
  //------------------------------------------------------------------------------
  // TYPEDEFS
  //------------------------------------------------------------------------------
  struct _TempStorage {
    OutSymbolOffsetT out_offset[MAX_NUM_STATES * MAX_NUM_SYMBOLS + 1];
    OutSymbolT out_symbols[MAX_TABLE_SIZE];
  };

  struct TempStorage : cub::Uninitialized<_TempStorage> {
  };

  struct KernelParameter {
    OutSymbolOffsetT* d_trans_offsets;
    OutSymbolT* d_out_symbols;
  };

  //------------------------------------------------------------------------------
  // HELPER METHODS
  //------------------------------------------------------------------------------
  __host__ static cudaError_t CreateTransitionTable(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const std::vector<std::vector<std::vector<OutSymbolT>>>& trans_table,
    KernelParameter& kernel_param,
    cudaStream_t stream = 0)
  {
    enum { MEM_OFFSETS = 0, MEM_OUT_SYMBOLS, NUM_ALLOCATIONS };

    size_t allocation_sizes[NUM_ALLOCATIONS] = {};
    void* allocations[NUM_ALLOCATIONS]       = {};
    allocation_sizes[MEM_OFFSETS] =
      (MAX_NUM_STATES * MAX_NUM_SYMBOLS + 1) * sizeof(OutSymbolOffsetT);
    allocation_sizes[MEM_OUT_SYMBOLS] = MAX_TABLE_SIZE * sizeof(OutSymbolT);

    // Alias the temporary allocations from the single storage blob (or compute the necessary size
    // of the blob)
    cudaError_t error =
      cub::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes);
    if (error) return error;

    // Return if the caller is simply requesting the size of the storage allocation
    if (d_temp_storage == nullptr) return cudaSuccess;

    std::vector<OutSymbolT> out_symbols;
    out_symbols.reserve(MAX_TABLE_SIZE);
    std::vector<OutSymbolOffsetT> out_symbol_offsets;
    out_symbol_offsets.reserve(MAX_NUM_STATES * MAX_NUM_SYMBOLS + 1);
    out_symbol_offsets.push_back(0);

    int st = 0;
    // Iterate over the states in the transition table
    for (auto const& state_trans : trans_table) {
      uint32_t num_added = 0;
      // Iterate over the symbols in the transition table
      for (auto const& symbol_out : state_trans) {
        // Insert the output symbols for this specific (state, symbol) transition
        out_symbols.insert(std::end(out_symbols), std::begin(symbol_out), std::end(symbol_out));
        out_symbol_offsets.push_back(out_symbols.size());
        num_added++;
      }
      st++;

      // Copy the last offset for all symbols (to guarantee a proper lookup for omitted symbols of
      // this state)
      if (MAX_NUM_SYMBOLS > num_added) {
        int32_t count = MAX_NUM_SYMBOLS - num_added;
        auto begin_it = std::prev(std::end(out_symbol_offsets));
        std::copy(begin_it, begin_it + count, std::back_inserter(out_symbol_offsets));
      }
    }

    // Check whether runtime-provided table size exceeds the compile-time given max. table size
    if (out_symbols.size() > MAX_TABLE_SIZE) { return cudaErrorInvalidValue; }

    kernel_param.d_trans_offsets = static_cast<OutSymbolOffsetT*>(allocations[MEM_OFFSETS]);
    kernel_param.d_out_symbols   = static_cast<OutSymbolT*>(allocations[MEM_OUT_SYMBOLS]);

    // Copy out symbols
    error = cudaMemcpyAsync(kernel_param.d_trans_offsets,
                            out_symbol_offsets.data(),
                            out_symbol_offsets.size() * sizeof(out_symbol_offsets[0]),
                            cudaMemcpyHostToDevice,
                            stream);
    if (error) { return error; }

    // Copy offsets into output symbols
    return cudaMemcpyAsync(kernel_param.d_out_symbols,
                           out_symbols.data(),
                           out_symbols.size() * sizeof(out_symbols[0]),
                           cudaMemcpyHostToDevice,
                           stream);
  }

  //------------------------------------------------------------------------------
  // MEMBER VARIABLES
  //------------------------------------------------------------------------------
  _TempStorage& temp_storage;

  __device__ __forceinline__ _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  //------------------------------------------------------------------------------
  // CONSTRUCTOR
  //------------------------------------------------------------------------------
  __host__ __device__ __forceinline__ TransducerLookupTable(const KernelParameter& kernel_param,
                                                            TempStorage& temp_storage)
    : temp_storage(temp_storage.Alias())
  {
    constexpr uint32_t num_offsets = MAX_NUM_STATES * MAX_NUM_SYMBOLS + 1;
#if CUB_PTX_ARCH > 0
    for (int i = threadIdx.x; i < num_offsets; i += blockDim.x) {
      this->temp_storage.out_offset[i] = kernel_param.d_trans_offsets[i];
    }
    // Make sure all threads in the block can read out_symbol_offsets[num_offsets - 1] from shared
    // memory
    __syncthreads();
    for (int i = threadIdx.x; i < this->temp_storage.out_offset[num_offsets - 1]; i += blockDim.x) {
      this->temp_storage.out_symbols[i] = kernel_param.d_out_symbols[i];
    }
    __syncthreads();
#else
    for (int i = 0; i < num_offsets; i++) {
      this->temp_storage.out_symbol_offsets[i] = kernel_param.d_trans_offsets[i];
    }
    for (int i = 0; i < this->temp_storage.out_symbol_offsets[i]; i++) {
      this->temp_storage.out_symbols[i] = kernel_param.d_out_symbols[i];
    }
#endif
  }

  template <typename StateIndexT, typename SymbolIndexT, typename RelativeOffsetT>
  __host__ __device__ __forceinline__ OutSymbolT operator()(StateIndexT state_id,
                                                            SymbolIndexT match_id,
                                                            RelativeOffsetT relative_offset) const
  {
    auto offset = temp_storage.out_offset[state_id * MAX_NUM_SYMBOLS + match_id] + relative_offset;
    return temp_storage.out_symbols[offset];
  }

  template <typename StateIndexT, typename SymbolIndexT>
  __host__ __device__ __forceinline__ OutSymbolOffsetT operator()(StateIndexT state_id,
                                                                  SymbolIndexT match_id) const
  {
    return temp_storage.out_offset[state_id * MAX_NUM_SYMBOLS + match_id + 1] -
           temp_storage.out_offset[state_id * MAX_NUM_SYMBOLS + match_id];
  }
};

}  // namespace detail
}  // namespace fst
}  // namespace io
}  // namespace cudf
