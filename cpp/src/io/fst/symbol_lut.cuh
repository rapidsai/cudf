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
#include <io/utilities/hostdevice_vector.hpp>

#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cudf {
namespace io {
namespace fst {
namespace detail {
/**
 * @brief Class template that can be plugged into the finite-state machine to look up the symbol
 * group index for a given symbol. Class template does not support multi-symbol lookups (i.e., no
 * look-ahead).
 *
 * @tparam SymbolT The symbol type being passed in to lookup the corresponding symbol group id
 */
template <typename SymbolT>
class SingleSymbolSmemLUT {
 private:
  // Type used for representing a symbol group id (i.e., what we return for a given symbol)
  using SymbolGroupIdT = uint8_t;

  /// Number of entries for every lookup (e.g., for 8-bit Symbol this is 256)
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

  //------------------------------------------------------------------------------
  // HELPER METHODS
  //------------------------------------------------------------------------------
  /**
   * @brief
   *
   * @param[out] sgid_init A hostdevice_vector that will be populated
   * @param[in] symbol_strings Array of strings, where the i-th string holds all symbols
   * (characters!) that correspond to the i-th symbol group index
   * @param[in] stream The stream that shall be used to cudaMemcpyAsync the lookup table
   * @return
   */
  template <typename SymbolGroupItT>
  static void InitDeviceSymbolGroupIdLut(hostdevice_vector<KernelParameter>& sgid_init,
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

    // Alias memory / return memory requiremenets
    // TODO I think this could be +1?
    sgid_init.host_ptr()->num_valid_entries = max_base_match_val + 2;

    sgid_init.host_to_device(stream);
  }

  //------------------------------------------------------------------------------
  // MEMBER VARIABLES
  //------------------------------------------------------------------------------
  _TempStorage& temp_storage;
  SymbolGroupIdT num_valid_entries;

  //------------------------------------------------------------------------------
  // CONSTRUCTOR
  //------------------------------------------------------------------------------
  __device__ __forceinline__ _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

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
    for (std::size_t i = 0; i < kernel_param.num_luts; i++) {
      this->temp_storage.sym_to_sgid[i] = kernel_param.sym_to_sgid[i];
    }
#endif
  }

  constexpr CUDF_HOST_DEVICE int32_t operator()(SymbolT const symbol) const
  {
    // Look up the symbol group for given symbol
    return temp_storage.sym_to_sgid[min(symbol, num_valid_entries - 1)];
  }
};

}  // namespace detail
}  // namespace fst
}  // namespace io
}  // namespace cudf
