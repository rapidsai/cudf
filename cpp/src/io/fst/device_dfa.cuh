/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "dispatch_dfa.cuh"
#include "io/utilities/hostdevice_vector.hpp"

#include <cstdint>

namespace cudf::io::fst {

/**
 * @brief Uses a deterministic finite automaton to transduce a sequence of symbols from an input
 * iterator to a sequence of transduced output symbols.
 *
 * @tparam DfaT The DFA specification
 * @tparam SymbolItT Random-access input iterator type to symbols fed into the FST
 * @tparam TransducedOutItT Random-access output iterator to which the transduced output will be
 * written
 * @tparam TransducedIndexOutItT Random-access output iterator type to which the input symbols'
 * indexes are written.
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

}  // namespace cudf::io::fst
