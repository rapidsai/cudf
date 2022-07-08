/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/error.hpp>

#include <thrust/pair.h>

#include <rmm/device_buffer.hpp>

#include <algorithm>

namespace cudf {
namespace io {
namespace {
// When processing the input in chunks, this is the maximum size of each chunk.
// Only one chunk is loaded on the GPU at a time, so this value is chosen to
// be small enough to fit on the GPU in most cases.
constexpr size_t max_chunk_bytes = 256 * 1024 * 1024;  // 256MB

constexpr int bytes_per_find_thread = 64;

using pos_key_pair = thrust::pair<uint64_t, char>;

template <typename T>
constexpr T divCeil(T dividend, T divisor) noexcept
{
  return (dividend + divisor - 1) / divisor;
}

/**
 * @brief Sets the specified element of the array to the passed value
 */
template <class T, class V>
__device__ __forceinline__ void setElement(T* array, cudf::size_type idx, const T& t, const V&)
{
  array[idx] = t;
}

/**
 * @brief Sets the specified element of the array of pairs using the two passed
 * parameters.
 */
template <class T, class V>
__device__ __forceinline__ void setElement(thrust::pair<T, V>* array,
                                           cudf::size_type idx,
                                           const T& t,
                                           const V& v)
{
  array[idx] = {t, v};
}

/**
 * @brief Overloads the setElement() functions for void* arrays.
 * Does not do anything, indexing is not allowed with void* arrays.
 */
template <class T, class V>
__device__ __forceinline__ void setElement(void*, cudf::size_type, const T&, const V&)
{
}

/**
 * @brief CUDA kernel that finds all occurrences of a character in the given
 * character array. If the 'positions' parameter is not void*,
 * positions of all occurrences are stored in the output array.
 *
 * @param[in] data Pointer to the input character array
 * @param[in] size Number of bytes in the input array
 * @param[in] offset Offset to add to the output positions
 * @param[in] key Character to find in the array
 * @param[in,out] count Pointer to the number of found occurrences
 * @param[out] positions Array containing the output positions
 */
template <class T>
__global__ void count_and_set_positions(const char* data,
                                        uint64_t size,
                                        uint64_t offset,
                                        const char key,
                                        cudf::size_type* count,
                                        T* positions)
{
  // thread IDs range per block, so also need the block id
  const uint64_t tid = threadIdx.x + (blockDim.x * blockIdx.x);
  const uint64_t did = tid * bytes_per_find_thread;

  const char* raw = (data + did);

  const long byteToProcess =
    ((did + bytes_per_find_thread) < size) ? bytes_per_find_thread : (size - did);

  // Process the data
  for (long i = 0; i < byteToProcess; i++) {
    if (raw[i] == key) {
      const auto idx = atomicAdd(count, (cudf::size_type)1);
      setElement(positions, idx, did + offset + i, key);
    }
  }
}

}  // namespace

template <class T>
cudf::size_type find_all_from_set(device_span<char const> data,
                                  std::vector<char> const& keys,
                                  uint64_t result_offset,
                                  T* positions,
                                  rmm::cuda_stream_view stream)
{
  int block_size    = 0;  // suggested thread count to use
  int min_grid_size = 0;  // minimum block count required
  CUDF_CUDA_TRY(
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, count_and_set_positions<T>));
  const int grid_size = divCeil(data.size(), (size_t)block_size);

  auto d_count = cudf::detail::make_zeroed_device_uvector_async<cudf::size_type>(1, stream);
  for (char key : keys) {
    count_and_set_positions<T><<<grid_size, block_size, 0, stream.value()>>>(
      data.data(), data.size(), result_offset, key, d_count.data(), positions);
  }

  return cudf::detail::make_std_vector_sync(d_count, stream)[0];
}

template <class T>
cudf::size_type find_all_from_set(host_span<char const> data,
                                  const std::vector<char>& keys,
                                  uint64_t result_offset,
                                  T* positions,
                                  rmm::cuda_stream_view stream)
{
  rmm::device_buffer d_chunk(std::min(max_chunk_bytes, data.size()), stream);
  auto d_count = cudf::detail::make_zeroed_device_uvector_async<cudf::size_type>(1, stream);

  int block_size    = 0;  // suggested thread count to use
  int min_grid_size = 0;  // minimum block count required
  CUDF_CUDA_TRY(
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, count_and_set_positions<T>));

  const size_t chunk_count = divCeil(data.size(), max_chunk_bytes);
  for (size_t ci = 0; ci < chunk_count; ++ci) {
    const auto chunk_offset = ci * max_chunk_bytes;
    const auto h_chunk      = data.data() + chunk_offset;
    const int chunk_bytes = std::min((size_t)(data.size() - ci * max_chunk_bytes), max_chunk_bytes);
    const auto chunk_bits = divCeil(chunk_bytes, bytes_per_find_thread);
    const int grid_size   = divCeil(chunk_bits, block_size);

    // Copy chunk to device
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(d_chunk.data(), h_chunk, chunk_bytes, cudaMemcpyDefault, stream.value()));

    for (char key : keys) {
      count_and_set_positions<T>
        <<<grid_size, block_size, 0, stream.value()>>>(static_cast<char*>(d_chunk.data()),
                                                       chunk_bytes,
                                                       chunk_offset + result_offset,
                                                       key,
                                                       d_count.data(),
                                                       positions);
    }
  }

  return cudf::detail::make_std_vector_sync(d_count, stream)[0];
}

template cudf::size_type find_all_from_set<uint64_t>(device_span<char const> data,
                                                     std::vector<char> const& keys,
                                                     uint64_t result_offset,
                                                     uint64_t* positions,
                                                     rmm::cuda_stream_view stream);

template cudf::size_type find_all_from_set<pos_key_pair>(device_span<char const> data,
                                                         std::vector<char> const& keys,
                                                         uint64_t result_offset,
                                                         pos_key_pair* positions,
                                                         rmm::cuda_stream_view stream);

template cudf::size_type find_all_from_set<uint64_t>(host_span<char const> data,
                                                     std::vector<char> const& keys,
                                                     uint64_t result_offset,
                                                     uint64_t* positions,
                                                     rmm::cuda_stream_view stream);

template cudf::size_type find_all_from_set<pos_key_pair>(host_span<char const> data,
                                                         std::vector<char> const& keys,
                                                         uint64_t result_offset,
                                                         pos_key_pair* positions,
                                                         rmm::cuda_stream_view stream);

cudf::size_type count_all_from_set(device_span<char const> data,
                                   std::vector<char> const& keys,
                                   rmm::cuda_stream_view stream)
{
  return find_all_from_set<void>(data, keys, 0, nullptr, stream);
}

cudf::size_type count_all_from_set(host_span<char const> data,
                                   const std::vector<char>& keys,
                                   rmm::cuda_stream_view stream)
{
  return find_all_from_set<void>(data, keys, 0, nullptr, stream);
}

}  // namespace io
}  // namespace cudf
