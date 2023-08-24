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

#include <cudf/detail/utilities/logger.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf::io::detail::parquet {

namespace {

// TODO: what is a good number here. what's the penalty for making it larger?
std::size_t constexpr STREAM_POOL_SIZE = 32;

std::mutex stream_pool_mutex;

auto& get_stream_pool()
{
  // TODO: creating this on the heap because there were issues with trying to call the
  // stream pool destructor during cuda shutdown that lead to a segmentation fault in
  // nvbench. this allocation is being deliberately leaked to avoid the above.
#if 1
  static auto pool = new rmm::cuda_stream_pool{STREAM_POOL_SIZE};
  return *pool;
#else
  // FIXME
  // running ./benchmarks/PARQUET_READER_NVBENCH -b parquet_read_decode --axis data_type=STRUCT
  // with this code results in a segmentation fault in the cuda_stream_pool dtor during shutdown.
  // it seems cudaStreamDestroy is called twice on the streams in the pool.
  static rmm::cuda_stream_pool pool{STREAM_POOL_SIZE};
  return pool;
#endif
}

}  // anonymous namespace

rmm::cuda_stream_view get_stream() { return get_stream_pool().get_stream(); }

rmm::cuda_stream_view get_stream(std::size_t stream_id)
{
  return get_stream_pool().get_stream(stream_id);
}

std::vector<rmm::cuda_stream_view> get_streams(uint32_t count)
{
  if (count > STREAM_POOL_SIZE) {
    CUDF_LOG_WARN("get_streams called with count ({}) > pool size ({})", count, STREAM_POOL_SIZE);
  }
  auto streams = std::vector<rmm::cuda_stream_view>();
  std::lock_guard<std::mutex> lock(stream_pool_mutex);
  for (uint32_t i = 0; i < count; i++) {
    streams.emplace_back(get_stream_pool().get_stream());
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
