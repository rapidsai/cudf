/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "page_decode.cuh"

namespace cudf::io::parquet::detail {

#define CUDF_PARQUET_PAGE_STATE_ERROR_METHODS                                                  \
  inline __device__ void set_error_code(decode_error err)                                      \
  {                                                                                            \
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{setup.error};     \
    ref.fetch_or(static_cast<kernel_error::value_type>(err), cuda::std::memory_order_relaxed); \
  }                                                                                            \
  inline __device__ void reset_error_code()                                                    \
  {                                                                                            \
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{setup.error};     \
    ref.store(0, cuda::std::memory_order_release);                                             \
  }

// Composition C: level-only preprocess (RLE level decoding only, no output).
// Used by: preprocess_levels.
// Includes setup (page metadata + error) and stream (level byte ranges + RLE state)
// because this pass only advances the level streams and never needs nesting,
// progress, or conversion scratch.
struct level_scan_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

#undef CUDF_PARQUET_PAGE_STATE_ERROR_METHODS

}  // namespace cudf::io::parquet::detail
