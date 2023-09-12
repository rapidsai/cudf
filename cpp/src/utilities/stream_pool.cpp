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

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace cudf::detail {

namespace {

// TODO: what is a good number here. what's the penalty for making it larger?
// Dave Baranec rule of thumb was max_streams_needed * num_concurrent_threads,
// where num_concurrent_threads was estimated to be 4. so using 32 will allow
// for 8 streams per thread, which should be plenty (decoding will be up to 4
// kernels when delta_byte_array decoding is added). rmm::cuda_stream_pool
// defaults to 16.
std::size_t constexpr STREAM_POOL_SIZE = 32;

// FIXME: "borrowed" from rmm...remove when this stream pool is moved there
#ifdef NDEBUG
#define CUDF_ASSERT_CUDA_SUCCESS(_call) \
  do {                                  \
    (_call);                            \
  } while (0);
#else
#define CUDF_ASSERT_CUDA_SUCCESS(_call)                                         \
  do {                                                                          \
    cudaError_t const status__ = (_call);                                       \
    if (status__ != cudaSuccess) {                                              \
      std::cerr << "CUDA Error detected. " << cudaGetErrorName(status__) << " " \
                << cudaGetErrorString(status__) << std::endl;                   \
    }                                                                           \
    /* NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay) */   \
    assert(status__ == cudaSuccess);                                            \
  } while (0)
#endif

class cuda_stream_pool {
 public:
  // matching type used in rmm::cuda_stream_pool::get_stream(stream_id)
  using stream_id_type = std::size_t;

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
  virtual rmm::cuda_stream_view get_stream(stream_id_type stream_id) = 0;

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
  virtual std::vector<rmm::cuda_stream_view> get_streams(std::size_t count) = 0;

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
  rmm::cuda_stream_view get_stream(stream_id_type stream_id) override
  {
    return _pool.get_stream(stream_id);
  }

  std::vector<rmm::cuda_stream_view> get_streams(std::size_t count) override
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
  rmm::cuda_stream_view get_stream(stream_id_type stream_id) override
  {
    return cudf::get_default_stream();
  }

  std::vector<rmm::cuda_stream_view> get_streams(std::size_t count) override
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

// FIXME: these will be available in rmm soon
inline int get_num_cuda_devices()
{
  rmm::cuda_device_id::value_type num_dev{};
  CUDF_CUDA_TRY(cudaGetDeviceCount(&num_dev));
  return num_dev;
}

rmm::cuda_device_id get_current_cuda_device()
{
  int device_id;
  CUDF_CUDA_TRY(cudaGetDevice(&device_id));
  return rmm::cuda_device_id{device_id};
}

/**
 * @brief RAII struct to wrap a cuda event and ensure its proper destruction.
 */
struct cuda_event {
  cuda_event() { CUDF_CUDA_TRY(cudaEventCreateWithFlags(&e_, cudaEventDisableTiming)); }
  virtual ~cuda_event() { CUDF_ASSERT_CUDA_SUCCESS(cudaEventDestroy(e_)); }

  operator cudaEvent_t() { return e_; }

 private:
  cudaEvent_t e_;
};

/**
 * @brief Returns a cudaEvent_t for the current thread.
 *
 * The returned event is valid for the current device.
 *
 * @return A cudaEvent_t unique to the current thread and valid on the current device.
 */
cudaEvent_t event_for_thread()
{
  thread_local std::vector<std::unique_ptr<cuda_event>> thread_events(get_num_cuda_devices());
  auto const device_id = get_current_cuda_device();
  if (not thread_events[device_id.value()]) {
    thread_events[device_id.value()] = std::make_unique<cuda_event>();
  }
  return *thread_events[device_id.value()];
}

/**
 * @brief Returns a reference to the global stream pool for the current device.
 * @return `cuda_stream_pool` valid on the current device.
 */
cuda_stream_pool& global_cuda_stream_pool()
{
  // using bare pointers here to deliberately allow them to leak. otherwise we wind up with
  // seg faults trying to destroy stream objects after the context has shut down.
  static std::vector<cuda_stream_pool*> pools(get_num_cuda_devices());
  static std::mutex mutex;
  auto const device_id = get_current_cuda_device();

  std::lock_guard<std::mutex> lock(mutex);
  if (pools[device_id.value()] == nullptr) {
    pools[device_id.value()] = create_global_cuda_stream_pool();
  }
  return *pools[device_id.value()];
}

}  // anonymous namespace

std::vector<rmm::cuda_stream_view> fork_streams(rmm::cuda_stream_view stream, std::size_t count)
{
  auto const streams = global_cuda_stream_pool().get_streams(count);
  auto const event   = event_for_thread();
  CUDF_CUDA_TRY(cudaEventRecord(event, stream));
  std::for_each(streams.begin(), streams.end(), [&](auto& strm) {
    CUDF_CUDA_TRY(cudaStreamWaitEvent(strm, event, 0));
  });
  return streams;
}

void join_streams(host_span<rmm::cuda_stream_view const> streams, rmm::cuda_stream_view stream)
{
  auto const event = event_for_thread();
  std::for_each(streams.begin(), streams.end(), [&](auto& strm) {
    CUDF_CUDA_TRY(cudaEventRecord(event, strm));
    CUDF_CUDA_TRY(cudaStreamWaitEvent(stream, event, 0));
  });
}

}  // namespace cudf::detail
