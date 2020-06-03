#include <cudf/utilities/error.hpp>

#include <thrust/device_vector.h>
#include <thrust/pair.h>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

#include <cudf/io/types.hpp>

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
 **/
template <class T, class V>
__device__ __forceinline__ void setElement(T* array, cudf::size_type idx, const T& t, const V& v)
{
  array[idx] = t;
}

/**
 * @brief Sets the specified element of the array of pairs using the two passed
 * parameters.
 **/
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
 **/
template <class T, class V>
__device__ __forceinline__ void setElement(void* array, cudf::size_type idx, const T& t, const V& v)
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
 *
 * @return void
 **/
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
cudf::size_type find_all_from_set(const rmm::device_buffer& d_data,
                                  const std::vector<char>& keys,
                                  uint64_t result_offset,
                                  T* positions)
{
  int block_size    = 0;  // suggested thread count to use
  int min_grid_size = 0;  // minimum block count required
  CUDA_TRY(
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, count_and_set_positions<T>));
  const int grid_size = divCeil(d_data.size(), (size_t)block_size);

  rmm::device_vector<cudf::size_type> d_count(1, 0);
  for (char key : keys) {
    count_and_set_positions<T><<<grid_size, block_size>>>(static_cast<const char*>(d_data.data()),
                                                          d_data.size(),
                                                          result_offset,
                                                          key,
                                                          d_count.data().get(),
                                                          positions);
  }

  return d_count[0];
}

template <class T>
cudf::size_type find_all_from_set(const char* h_data,
                                  size_t h_size,
                                  const std::vector<char>& keys,
                                  uint64_t result_offset,
                                  T* positions)
{
  rmm::device_buffer d_chunk(std::min(max_chunk_bytes, h_size));
  rmm::device_vector<cudf::size_type> d_count(1, 0);

  int block_size    = 0;  // suggested thread count to use
  int min_grid_size = 0;  // minimum block count required
  CUDA_TRY(
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, count_and_set_positions<T>));

  const size_t chunk_count = divCeil(h_size, max_chunk_bytes);
  for (size_t ci = 0; ci < chunk_count; ++ci) {
    const auto chunk_offset = ci * max_chunk_bytes;
    const auto h_chunk      = h_data + chunk_offset;
    const int chunk_bytes   = std::min((size_t)(h_size - ci * max_chunk_bytes), max_chunk_bytes);
    const auto chunk_bits   = divCeil(chunk_bytes, bytes_per_find_thread);
    const int grid_size     = divCeil(chunk_bits, block_size);

    // Copy chunk to device
    CUDA_TRY(cudaMemcpyAsync(d_chunk.data(), h_chunk, chunk_bytes, cudaMemcpyDefault));

    for (char key : keys) {
      count_and_set_positions<T><<<grid_size, block_size>>>(static_cast<char*>(d_chunk.data()),
                                                            chunk_bytes,
                                                            chunk_offset + result_offset,
                                                            key,
                                                            d_count.data().get(),
                                                            positions);
    }
  }

  return d_count[0];
}

template cudf::size_type find_all_from_set<uint64_t>(const rmm::device_buffer& d_data,
                                                     const std::vector<char>& keys,
                                                     uint64_t result_offset,
                                                     uint64_t* positions);

template cudf::size_type find_all_from_set<pos_key_pair>(const rmm::device_buffer& d_data,
                                                         const std::vector<char>& keys,
                                                         uint64_t result_offset,
                                                         pos_key_pair* positions);

template cudf::size_type find_all_from_set<uint64_t>(const char* h_data,
                                                     size_t h_size,
                                                     const std::vector<char>& keys,
                                                     uint64_t result_offset,
                                                     uint64_t* positions);

template cudf::size_type find_all_from_set<pos_key_pair>(const char* h_data,
                                                         size_t h_size,
                                                         const std::vector<char>& keys,
                                                         uint64_t result_offset,
                                                         pos_key_pair* positions);

cudf::size_type count_all_from_set(const rmm::device_buffer& d_data, const std::vector<char>& keys)
{
  return find_all_from_set<void>(d_data, keys, 0, nullptr);
}

cudf::size_type count_all_from_set(const char* h_data, size_t h_size, const std::vector<char>& keys)
{
  return find_all_from_set<void>(h_data, h_size, keys, 0, nullptr);
}

std::string infer_compression_type(
  const compression_type& compression_arg,
  const std::string& filename,
  const std::vector<std::pair<std::string, std::string>>& ext_to_comp_map)
{
  auto str_tolower = [](const auto& begin, const auto& end) {
    std::string out;
    std::transform(begin, end, std::back_inserter(out), ::tolower);
    return out;
  };

  // Attempt to infer from user-supplied argument
  if (compression_arg != compression_type::AUTO) {
    switch (compression_arg) {
      case compression_type::GZIP: return "gzip";
      case compression_type::BZIP2: return "bz2";
      case compression_type::ZIP: return "zip";
      case compression_type::XZ: return "xz";
      default: break;
    }
  }

  // Attempt to infer from the file extension
  const auto pos = filename.find_last_of('.');
  if (pos != std::string::npos) {
    const auto ext = str_tolower(filename.begin() + pos + 1, filename.end());
    for (const auto& mapping : ext_to_comp_map) {
      if (mapping.first == ext) { return mapping.second; }
    }
  }

  return "none";
}

}  // namespace io
}  // namespace cudf
