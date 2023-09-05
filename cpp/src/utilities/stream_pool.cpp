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

#include <cudf/detail/utilities/logger.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_pool.hpp>

namespace cudf::detail {

namespace {

// TODO: what is a good number here. what's the penalty for making it larger?
// Dave Baranec rule of thumb was max_streams_needed * num_concurrent_threads,
// where num_concurrent_threads was estimated to be 4. so using 32 will allow
// for 8 streams per thread, which should be plenty (decoding will be up to 4
// kernels when delta_byte_array decoding is added). rmm::cuda_stream_pool
// defaults to 16.
std::size_t constexpr STREAM_POOL_SIZE = 32;

class cuda_stream_pool {
 public:
  virtual ~cuda_stream_pool() = default;

  /**
   * @brief Get a `cuda_stream_view` of a stream in the pool.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @return Stream view.
   */
  virtual rmm::cuda_stream_view get_stream() = 0;

  /**
   * @brief Get a `cuda_stream_view` of the stream associated with `stream_id`.
   *
   * Equivalent values of `stream_id` return a `cuda_stream_view` to the same underlying stream.
   * This function is thread safe with respect to other calls to the same function.
   *
   * @param stream_id Unique identifier for the desired stream
   * @return Requested stream view.
   */
  virtual rmm::cuda_stream_view get_stream(std::size_t stream_id) = 0;

  /**
   * @brief Get a set of `cuda_stream_view` objects from the pool.
   *
   * An attempt is made to ensure that the returned vector does not contain duplicate
   * streams, but this cannot be guaranteed if `count` is greater than the value returned by
   * `get_stream_pool_size()`.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @param count The number of stream views to return.
   * @return Vector containing `count` stream views.
   */
  virtual std::vector<rmm::cuda_stream_view> get_streams(uint32_t count) = 0;

  /**
   * @brief Get the number of stream objects in the pool.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @return the number of stream objects in the pool
   */
  virtual std::size_t get_stream_pool_size() const = 0;
};

/**
 * @brief Implementation of `cuda_stream_pool` that wraps an `rmm::cuda_stram_pool`.
 */
class rmm_cuda_stream_pool : public cuda_stream_pool {
  rmm::cuda_stream_pool _pool;

 public:
  rmm_cuda_stream_pool() : _pool{STREAM_POOL_SIZE} {}
  rmm::cuda_stream_view get_stream() override { return _pool.get_stream(); }
  rmm::cuda_stream_view get_stream(std::size_t stream_id) override
  {
    return _pool.get_stream(stream_id);
  }

  std::vector<rmm::cuda_stream_view> get_streams(uint32_t count) override
  {
    if (count > STREAM_POOL_SIZE) {
      CUDF_LOG_WARN("get_streams called with count ({}) > pool size ({})", count, STREAM_POOL_SIZE);
    }
    auto streams = std::vector<rmm::cuda_stream_view>();
    for (uint32_t i = 0; i < count; i++) {
      streams.emplace_back(_pool.get_stream());
    }
    return streams;
  }

  std::size_t get_stream_pool_size() const override { return STREAM_POOL_SIZE; }
};

/**
 * @brief Implementation of `cuda_stream_pool` that always returns `cudf::get_default_stream()`
 */
class debug_cuda_stream_pool : public cuda_stream_pool {
 public:
  rmm::cuda_stream_view get_stream() override { return cudf::get_default_stream(); }
  rmm::cuda_stream_view get_stream(std::size_t stream_id) override
  {
    return cudf::get_default_stream();
  }

  std::vector<rmm::cuda_stream_view> get_streams(uint32_t count) override
  {
    return std::vector<rmm::cuda_stream_view>(count, cudf::get_default_stream());
  }

  std::size_t get_stream_pool_size() const override { return 1UL; }
};

/**
 * @brief Initialize global stream pool.
 */
cuda_stream_pool* create_global_cuda_stream_pool()
{
  if (getenv("LIBCUDF_USE_DEBUG_STREAM_POOL")) return new debug_cuda_stream_pool();

  return new rmm_cuda_stream_pool();
}

/**
 * @brief RAII struct to wrap a cuda event and ensure it's proper destruction.
 */
struct cuda_event {
  cuda_event()
    : e_{[]() {
        cudaEvent_t event;
        CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        return event;
      }()}
  {
  }

  operator cudaEvent_t() { return e_.get(); }

 private:
  struct deleter {
    using pointer = cudaEvent_t;
    auto operator()(cudaEvent_t e) { cudaEventDestroy(e); }
  };
  std::unique_ptr<cudaEvent_t, deleter> e_;
};

/**
 * @brief Returns a cudaEvent_t for the current thread.
 */
cudaEvent_t event_for_thread()
{
  thread_local cuda_event thread_event;
  return thread_event;
}

/**
 * Returns a reference to the global stream ppol.
 */
cuda_stream_pool& global_cuda_stream_pool()
{
  static cuda_stream_pool* pool = create_global_cuda_stream_pool();
  return *pool;
}

}  // anonymous namespace

std::vector<rmm::cuda_stream_view> fork_stream(rmm::cuda_stream_view stream, uint32_t count)
{
  auto streams      = global_cuda_stream_pool().get_streams(count);
  cudaEvent_t event = event_for_thread();
  CUDF_CUDA_TRY(cudaEventRecord(event, stream));
  std::for_each(streams.begin(), streams.end(), [&](auto& strm) {
    CUDF_CUDA_TRY(cudaStreamWaitEvent(strm, event, 0));
  });
  for (auto& strm : streams) {
    CUDF_CUDA_TRY(cudaStreamWaitEvent(strm, event, 0));
  }
  return streams;
}

void join_streams(host_span<rmm::cuda_stream_view> streams, rmm::cuda_stream_view stream)
{
  cudaEvent_t event = event_for_thread();
  for (auto& strm : streams) {
    CUDF_CUDA_TRY(cudaEventRecord(event, strm));
    CUDF_CUDA_TRY(cudaStreamWaitEvent(stream, event, 0));
  }
}

}  // namespace cudf::detail
