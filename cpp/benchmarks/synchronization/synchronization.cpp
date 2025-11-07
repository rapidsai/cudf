/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "synchronization.hpp"

#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

cuda_event_timer::cuda_event_timer(benchmark::State& state,
                                   bool flush_l2_cache,
                                   rmm::cuda_stream_view stream)
  : stream(stream), p_state(&state)
{
  // flush all of L2$
  if (flush_l2_cache) {
    int current_device = 0;
    CUDF_CUDA_TRY(cudaGetDevice(&current_device));

    int l2_cache_bytes = 0;
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&l2_cache_bytes, cudaDevAttrL2CacheSize, current_device));

    if (l2_cache_bytes > 0) {
      int const memset_value = 0;
      rmm::device_buffer l2_cache_buffer(l2_cache_bytes, stream);
      CUDF_CUDA_TRY(
        cudaMemsetAsync(l2_cache_buffer.data(), memset_value, l2_cache_bytes, stream.value()));
    }
  }

  CUDF_CUDA_TRY(cudaEventCreate(&start));
  CUDF_CUDA_TRY(cudaEventCreate(&stop));
  CUDF_CUDA_TRY(cudaEventRecord(start, stream.value()));
}

cuda_event_timer::~cuda_event_timer()
{
  CUDF_CUDA_TRY(cudaEventRecord(stop, stream.value()));
  CUDF_CUDA_TRY(cudaEventSynchronize(stop));

  float milliseconds = 0.0f;
  CUDF_CUDA_TRY(cudaEventElapsedTime(&milliseconds, start, stop));
  p_state->SetIterationTime(milliseconds / (1000.0f));
  CUDF_CUDA_TRY(cudaEventDestroy(start));
  CUDF_CUDA_TRY(cudaEventDestroy(stop));
}
