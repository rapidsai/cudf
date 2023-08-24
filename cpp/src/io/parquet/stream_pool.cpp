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
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf::io::detail::parquet {

namespace {

// TODO: what is a good number here. what's the penalty for making it larger?
std::size_t constexpr STREAM_POOL_SIZE = 32;

class cuda_stream_pool {
 public:
  virtual ~cuda_stream_pool() = default;

  virtual rmm::cuda_stream_view get_stream()                      = 0;
  virtual rmm::cuda_stream_view get_stream(std::size_t stream_id) = 0;
};

class rmm_cuda_stream_pool : public cuda_stream_pool {
  rmm::cuda_stream_pool _pool;

 public:
  rmm_cuda_stream_pool() : _pool{STREAM_POOL_SIZE} {}
  rmm::cuda_stream_view get_stream() override { return _pool.get_stream(); }
  rmm::cuda_stream_view get_stream(std::size_t stream_id) override
  {
    return _pool.get_stream(stream_id);
  }
};

class debug_cuda_stream_pool : public cuda_stream_pool {
 public:
  rmm::cuda_stream_view get_stream() override { return cudf::get_default_stream(); }
  rmm::cuda_stream_view get_stream(std::size_t stream_id) override
  {
    return cudf::get_default_stream();
  }
};

cuda_stream_pool* create_global_cuda_stream_pool()
{
  if (getenv("LIBCUDF_USE_DEBUG_STREAM_POOL")) return new debug_cuda_stream_pool();

  return new rmm_cuda_stream_pool();
}

// TODO: hidden for now...can move out of the anonymous namespace if this needs to be exposed
// to users.
// TODO: move get_streams(uint32_t) into the interface, or leave as is?
cuda_stream_pool& global_cuda_stream_pool()
{
  static cuda_stream_pool* pool = create_global_cuda_stream_pool();
  return *pool;
}

std::mutex stream_pool_mutex;

}  // anonymous namespace

// TODO: these next 2 (3?) can go away if we expose global_cuda_stream_pool()
rmm::cuda_stream_view get_stream() { return global_cuda_stream_pool().get_stream(); }

rmm::cuda_stream_view get_stream(std::size_t stream_id)
{
  return global_cuda_stream_pool().get_stream(stream_id);
}

std::vector<rmm::cuda_stream_view> get_streams(uint32_t count)
{
  if (count > STREAM_POOL_SIZE) {
    CUDF_LOG_WARN("get_streams called with count ({}) > pool size ({})", count, STREAM_POOL_SIZE);
  }
  auto streams = std::vector<rmm::cuda_stream_view>();
  std::lock_guard<std::mutex> lock(stream_pool_mutex);
  for (uint32_t i = 0; i < count; i++) {
    streams.emplace_back(global_cuda_stream_pool().get_stream());
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
