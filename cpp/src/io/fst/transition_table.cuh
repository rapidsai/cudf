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

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <io/utilities/hostdevice_vector.hpp>

#include <cub/cub.cuh>

#include <cstdint>

namespace cudf {
namespace io {
namespace fst {
namespace detail {

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

  static void InitDeviceTransitionTable(hostdevice_vector<KernelParameter>& transition_table_init,
                                        const std::vector<std::vector<int>>& trans_table,
                                        rmm::cuda_stream_view stream)
  {
    // trans_table[state][symbol] -> new state
    for (std::size_t state = 0; state < trans_table.size(); ++state) {
      for (std::size_t symbol = 0; symbol < trans_table[state].size(); ++symbol) {
        transition_table_init.host_ptr()->transitions[symbol * MAX_NUM_STATES + state] =
          trans_table[state][symbol];
      }
    }

    // Copy transition table to device
    transition_table_init.host_to_device(stream);
  }

  constexpr CUDF_HOST_DEVICE TransitionTable(const KernelParameter& kernel_param,
                                             TempStorage& temp_storage)
    : temp_storage(temp_storage.Alias())
  {
#if CUB_PTX_ARCH > 0
    for (int i = threadIdx.x; i < MAX_NUM_STATES * MAX_NUM_SYMBOLS; i += blockDim.x) {
      this->temp_storage.transitions[i] = kernel_param.transitions[i];
    }
    __syncthreads();
#else
    for (int i = 0; i < MAX_NUM_STATES * MAX_NUM_SYMBOLS; i++) {
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

}  // namespace detail
}  // namespace fst
}  // namespace io
}  // namespace cudf
