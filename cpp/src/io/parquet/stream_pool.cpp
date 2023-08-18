/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <mutex>

#include "stream_pool.hpp"

#include <cudf/utilities/error.hpp>

namespace cudf::io::detail::parquet {

namespace {

std::size_t constexpr STREAM_POOL_SIZE = 32;

// doing lazy initialization to avoid allocating this before cuInit is called (was
// causing issues with compute-sanitizer).
// deliberately allowing the pool to leak to avoid deleting streams after cuda shutdown.
rmm::cuda_stream_pool* pool = nullptr;

std::mutex stream_pool_mutex;

void init()
{
  std::lock_guard<std::mutex> lock(stream_pool_mutex);
  if (pool == nullptr) { pool = new rmm::cuda_stream_pool(STREAM_POOL_SIZE); }
}

std::atomic_size_t stream_idx{};

}  // anonymous namespace

rmm::cuda_stream_view get_stream()
{
  init();
  return pool->get_stream();
}

rmm::cuda_stream_view get_stream(std::size_t stream_id)
{
  init();
  return pool->get_stream(stream_id);
}

std::vector<rmm::cuda_stream_view> get_streams(uint32_t count)
{
  init();

  // TODO maybe add mutex to be sure streams don't overlap
  auto streams = std::vector<rmm::cuda_stream_view>();
  for (uint32_t i = 0; i < count; i++) {
    streams.emplace_back(pool->get_stream((stream_idx++)));
  }
  return streams;
}

void fork_streams(std::vector<rmm::cuda_stream_view>& streams, rmm::cuda_stream_view stream)
{
  cudaEvent_t event;
  CUDF_CUDA_TRY(cudaEventCreate(&event));
  CUDF_CUDA_TRY(cudaEventRecord(event, stream));
  std::for_each(streams.begin(), streams.end(), [&](auto& strm) {
    CUDF_CUDA_TRY(cudaStreamWaitEvent(strm, event, 0));
  });
  CUDF_CUDA_TRY(cudaEventDestroy(event));
}

void join_streams(std::vector<rmm::cuda_stream_view>& streams, rmm::cuda_stream_view stream)
{
  cudaEvent_t event;
  CUDF_CUDA_TRY(cudaEventCreate(&event));
  std::for_each(streams.begin(), streams.end(), [&](auto& strm) {
    CUDF_CUDA_TRY(cudaEventRecord(event, strm));
    CUDF_CUDA_TRY(cudaStreamWaitEvent(stream, event, 0));
  });
  CUDF_CUDA_TRY(cudaEventDestroy(event));
}

std::size_t get_stream_pool_size() { return STREAM_POOL_SIZE; }

}  // namespace cudf::io::detail::parquet
