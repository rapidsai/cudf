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
struct SingleSymbolSmemLUT {
  //------------------------------------------------------------------------------
  // DEFAULT TYPEDEFS
  //------------------------------------------------------------------------------
  // Type used for representing a symbol group id (i.e., what we return for a given symbol)
  using SymbolGroupIdT = uint8_t;

  //------------------------------------------------------------------------------
  // DERIVED CONFIGURATIONS
  //------------------------------------------------------------------------------
  /// Number of entries for every lookup (e.g., for 8-bit Symbol this is 256)
  static constexpr uint32_t NUM_ENTRIES_PER_LUT = 0x01U << (sizeof(SymbolT) * 8U);

  //------------------------------------------------------------------------------
  // TYPEDEFS
  //------------------------------------------------------------------------------

  struct _TempStorage {
    // d_match_meta_data[symbol] -> symbol group index
    SymbolGroupIdT match_meta_data[NUM_ENTRIES_PER_LUT];
  };

  struct KernelParameter {
    // d_match_meta_data[min(symbol,num_valid_entries)] -> symbol group index
    SymbolGroupIdT num_valid_entries;

    // d_match_meta_data[symbol] -> symbol group index
    SymbolGroupIdT* d_match_meta_data;
  };

  struct TempStorage : cub::Uninitialized<_TempStorage> {
  };

  //------------------------------------------------------------------------------
  // HELPER METHODS
  //------------------------------------------------------------------------------
  /**
   * @brief
   *
   * @param[in] d_temp_storage Device-side temporary storage that can be used to store the lookup
   * table. If no storage is provided it will return the temporary storage requirements in \p
   * d_temp_storage_bytes.
   * @param[in,out] d_temp_storage_bytes Amount of device-side temporary storage that can be used in
   * the number of bytes
   * @param[in] symbol_strings Array of strings, where the i-th string holds all symbols
   * (characters!) that correspond to the i-th symbol group index
   * @param[out] kernel_param The kernel parameter object to be initialized with the given mapping
   * of symbols to symbol group ids.
   * @param[in] stream The stream that shall be used to cudaMemcpyAsync the lookup table
   * @return
   */
  template <typename SymbolGroupItT>
  __host__ __forceinline__ static cudaError_t PrepareLUT(void* d_temp_storage,
                                                         size_t& d_temp_storage_bytes,
                                                         SymbolGroupItT const& symbol_strings,
                                                         KernelParameter& kernel_param,
                                                         cudaStream_t stream = 0)
  {
    // The symbol group index to be returned if none of the given symbols match
    SymbolGroupIdT no_match_id = symbol_strings.size();

    std::vector<SymbolGroupIdT> lut(NUM_ENTRIES_PER_LUT);
    SymbolGroupIdT max_base_match_val = 0;

    // Initialize all entries: by default we return the no-match-id
    for (uint32_t i = 0; i < NUM_ENTRIES_PER_LUT; ++i) {
      lut[i] = no_match_id;
    }

    // Set up lookup table
    uint32_t sg_id = 0;
    for (auto const& sg_symbols : symbol_strings) {
      for (auto const& sg_symbol : sg_symbols) {
        max_base_match_val = std::max(max_base_match_val, static_cast<SymbolGroupIdT>(sg_symbol));
        lut[sg_symbol] = sg_id;
      }
      sg_id++;
    }

    // Initialize the out-of-bounds lookup: d_match_meta_data[max_base_match_val+1] -> no_match_id
    lut[max_base_match_val + 1] = no_match_id;

    // Alias memory / return memory requiremenets
    kernel_param.num_valid_entries = max_base_match_val + 2;
    if (d_temp_storage) {
      cudaError_t error = cudaMemcpyAsync(d_temp_storage,
                                          lut.data(),
                                          kernel_param.num_valid_entries * sizeof(SymbolGroupIdT),
                                          cudaMemcpyHostToDevice,
                                          stream);

      kernel_param.d_match_meta_data = reinterpret_cast<SymbolGroupIdT*>(d_temp_storage);
      return error;
    } else {
      d_temp_storage_bytes = kernel_param.num_valid_entries * sizeof(SymbolGroupIdT);
      return cudaSuccess;
    }

    return cudaSuccess;
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

  __host__ __device__ __forceinline__ SingleSymbolSmemLUT(KernelParameter const& kernel_param,
                                                          TempStorage& temp_storage)
    : temp_storage(temp_storage.Alias()), num_valid_entries(kernel_param.num_valid_entries)
  {
    // GPU-side init
#if CUB_PTX_ARCH > 0
    for (int32_t i = threadIdx.x; i < kernel_param.num_valid_entries; i += blockDim.x) {
      this->temp_storage.match_meta_data[i] = kernel_param.d_match_meta_data[i];
    }
    __syncthreads();

#else
    // CPU-side init
    for (std::size_t i = 0; i < kernel_param.num_luts; i++) {
      this->temp_storage.match_meta_data[i] = kernel_param.d_match_meta_data[i];
    }
#endif
  }

  __host__ __device__ __forceinline__ int32_t operator()(SymbolT const symbol) const
  {
    // Look up the symbol group for given symbol
    return temp_storage.match_meta_data[min(symbol, num_valid_entries - 1)];
  }
};

}  // namespace detail
}  // namespace fst
}  // namespace io
}  // namespace cudf
