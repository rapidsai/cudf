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

#include <cub/cub.cuh>

#include <cstdint>

namespace cudf {
namespace io {
namespace fst {
namespace detail {

template <int MAX_NUM_SYMBOLS, int MAX_NUM_STATES>
struct TransitionTable {
  //------------------------------------------------------------------------------
  // DEFAULT TYPEDEFS
  //------------------------------------------------------------------------------
  using ItemT = char;

  struct TransitionVectorWrapper {
    const ItemT* data;

    __host__ __device__ TransitionVectorWrapper(const ItemT* data) : data(data) {}

    __host__ __device__ __forceinline__ uint32_t Get(int32_t index) const { return data[index]; }
  };

  //------------------------------------------------------------------------------
  // TYPEDEFS
  //------------------------------------------------------------------------------
  using TransitionVectorT = TransitionVectorWrapper;

  struct _TempStorage {
    //
    ItemT transitions[MAX_NUM_STATES * MAX_NUM_SYMBOLS];
  };

  struct TempStorage : cub::Uninitialized<_TempStorage> {
  };

  struct KernelParameter {
    ItemT* transitions;
  };

  using LoadAliasT = std::uint32_t;

  static constexpr std::size_t NUM_AUX_MEM_BYTES =
    CUB_QUOTIENT_CEILING(MAX_NUM_STATES * MAX_NUM_SYMBOLS * sizeof(ItemT), sizeof(LoadAliasT)) *
    sizeof(LoadAliasT);

  //------------------------------------------------------------------------------
  // HELPER METHODS
  //------------------------------------------------------------------------------
  __host__ static cudaError_t CreateTransitionTable(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const std::vector<std::vector<int>>& trans_table,
    KernelParameter& kernel_param,
    cudaStream_t stream = 0)
  {
    if (!d_temp_storage) {
      temp_storage_bytes = NUM_AUX_MEM_BYTES;
      return cudaSuccess;
    }

    // trans_vectors[symbol][state] -> new_state
    ItemT trans_vectors[MAX_NUM_STATES * MAX_NUM_SYMBOLS];

    // trans_table[state][symbol] -> new state
    for (std::size_t state = 0; state < trans_table.size(); ++state) {
      for (std::size_t symbol = 0; symbol < trans_table[state].size(); ++symbol) {
        trans_vectors[symbol * MAX_NUM_STATES + state] = trans_table[state][symbol];
      }
    }

    kernel_param.transitions = static_cast<ItemT*>(d_temp_storage);

    // Copy transition table to device
    return cudaMemcpyAsync(
      d_temp_storage, trans_vectors, NUM_AUX_MEM_BYTES, cudaMemcpyHostToDevice, stream);
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
  __host__ __device__ __forceinline__ TransitionTable(const KernelParameter& kernel_param,
                                                      TempStorage& temp_storage)
    : temp_storage(temp_storage.Alias())
  {
#if CUB_PTX_ARCH > 0
    for (int i = threadIdx.x; i < CUB_QUOTIENT_CEILING(NUM_AUX_MEM_BYTES, sizeof(LoadAliasT));
         i += blockDim.x) {
      reinterpret_cast<LoadAliasT*>(this->temp_storage.transitions)[i] =
        reinterpret_cast<LoadAliasT*>(kernel_param.transitions)[i];
    }
    __syncthreads();
#else
    for (int i = 0; i < kernel_param.num_luts; i++) {
      this->temp_storage.transitions[i] = kernel_param.transitions[i];
    }
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
  __host__ __device__ __forceinline__ int32_t operator()(StateIndexT state_id,
                                                              SymbolIndexT match_id) const
  {
    return temp_storage.transitions[match_id * MAX_NUM_STATES + state_id];
   }
};

}  // namespace detail
}  // namespace fst
}  // namespace io
}  // namespace cudf
