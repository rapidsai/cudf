/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

namespace cuco{
namespace detail {
namespace cg = cooperative_groups;

/**
 * @brief Initializes each slot in the flat `slots` storage to contain `k` and `v`.
 *
 * Each space in `slots` that can hold a key value pair is initialized to a
 * `pair_atomic_type` containing the key `k` and the value `v`.
 * 
 * @tparam atomic_key_type Type of the `Key` atomic container
 * @tparam atomic_mapped_type Type of the `Value` atomic container
 * @tparam Key key type
 * @tparam Value value type
 * @tparam pair_atomic_type key/value pair type
 * @param slots Pointer to flat storage for the map's key/value pairs
 * @param k Key to which all keys in `slots` are initialized
 * @param v Value to which all values in `slots` are initialized
 * @param size Size of the storage pointed to by `slots`
 */
template<std::size_t block_size,
         typename atomic_key_type,
         typename atomic_mapped_type,
         typename Key,
         typename Value,
         typename pair_atomic_type>
__global__ void initialize(
  pair_atomic_type* const slots, Key k,
  Value v, std::size_t size) {
  
  auto tid = block_size * blockIdx.x + threadIdx.x;
  while (tid < size) {
    new (&slots[tid].first) atomic_key_type{k};
    new (&slots[tid].second) atomic_mapped_type{v};
    tid += gridDim.x * block_size;
  }
}

/**
 * @brief Inserts all key/value pairs in the range `[first, last)`. 
 *  
 * If multiple keys in `[first, last)` compare equal, it is unspecified which
 * element is inserted.
 * 
 * @tparam block_size 
 * @tparam InputIt Device accessible input iterator whose `value_type` is
* convertible to the map's `value_type`
 * @tparam atomicT Type of atomic storage
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of key/value pairs
 * @param last End of the sequence of key/value pairs
 * @param num_successes The number of successfully inserted key/value pairs
 * @param view Mutable device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function used to compare two keys for equality
 */
template<std::size_t block_size,
         typename InputIt,
         typename atomicT,
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void insert(InputIt first,
                       InputIt last,
                       atomicT* num_successes,
                       viewT view,
                       Hash hash,
                       KeyEqual key_equal) {
  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_successes = 0;
  
  auto tid = block_size * blockIdx.x + threadIdx.x;
  auto it = first + tid;
  
  while (it < last) {
    typename viewT::value_type const insert_pair{*it};
    if (view.insert(insert_pair, hash, key_equal)) { thread_num_successes++; }
    it += gridDim.x * block_size;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  std::size_t block_num_successes = BlockReduce(temp_storage).Sum(thread_num_successes);
  if(threadIdx.x == 0) {
    *num_successes += block_num_successes;
  }
}

/**
 * @brief Inserts all key/value pairs in the range `[first, last)`. 
 *  
 * If multiple keys in `[first, last)` compare equal, it is unspecified which
 * element is inserted. Uses the CUDA Cooperative Groups API to leverage groups
 * of multiple threads to perform each key/value insertion. This provides a 
 * significant boost in throughput compared to the non Cooperative Group
 * `insert` at moderate to high load factors.
 * 
 * @tparam block_size 
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform
 * inserts
 * @tparam InputIt Device accessible input iterator whose `value_type` is
* convertible to the map's `value_type`
 * @tparam atomicT Type of atomic storage
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of key/value pairs
 * @param last End of the sequence of key/value pairs
 * @param num_successes The number of successfully inserted key/value pairs
 * @param view Mutable device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function used to compare two keys for equality
 */
template<std::size_t block_size,
         uint32_t tile_size,
         typename InputIt,
         typename atomicT,
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void insert(InputIt first,
                       InputIt last,
                       atomicT* num_successes,
                       viewT view,
                       Hash hash,
                       KeyEqual key_equal) {
  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_successes = 0;

  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = block_size * blockIdx.x + threadIdx.x;
  auto it = first + tid / tile_size;
  
  while (it < last) {
    // force conversion to value_type
    typename viewT::value_type const insert_pair{*it};
    if (view.insert(tile, insert_pair, hash, key_equal) && tile.thread_rank() == 0) {
      thread_num_successes++;
    }
    it += (gridDim.x * block_size) / tile_size;
  }
  
  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  std::size_t block_num_successes = BlockReduce(temp_storage).Sum(thread_num_successes);
  if(threadIdx.x == 0) {
    *num_successes += block_num_successes;
  }
}

/**
 * @brief Finds the values corresponding to all keys in the range `[first, last)`.
 * 
 * If the key `*(first + i)` exists in the map, copies its associated value to `(output_begin + i)`. 
 * Else, copies the empty value sentinel. 
 * @tparam block_size The size of the thread block
 * @tparam Value The type of the mapped value for the map
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is 
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of values retrieved for each key
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template<std::size_t block_size,
         typename Value,
         typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void find(InputIt first,
                     InputIt last,
                     OutputIt output_begin,
                     viewT view,
                     Hash hash,
                     KeyEqual key_equal) {
  auto tid = block_size * blockIdx.x + threadIdx.x;
  auto key_idx = tid;
  __shared__ Value writeBuffer[block_size];
  
  while(first + key_idx < last) {
    auto key = *(first + key_idx);
    auto found = view.find(key, hash, key_equal);

    /* 
     * The ld.relaxed.gpu instruction used in view.find causes L1 to 
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    writeBuffer[threadIdx.x] =
      found == view.end()
        ? view.get_empty_value_sentinel()
        : found->second.load(cuda::std::memory_order_relaxed);
    __syncthreads();
    *(output_begin + key_idx) = writeBuffer[threadIdx.x];
    key_idx += gridDim.x * block_size;
  }
}

/**
 * @brief Finds the values corresponding to all keys in the range `[first, last)`.
 * 
 * If the key `*(first + i)` exists in the map, copies its associated value to `(output_begin + i)`. 
 * Else, copies the empty value sentinel. Uses the CUDA Cooperative Groups API to leverage groups
 * of multiple threads to find each key. This provides a significant boost in throughput compared 
 * to the non Cooperative Group `find` at moderate to high load factors.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform
 * inserts
 * @tparam Value The type of the mapped value for the map
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is 
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of values retrieved for each key
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template<std::size_t block_size, uint32_t tile_size,
         typename Value,
         typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void find(InputIt first,
                     InputIt last,
                     OutputIt output_begin,
                     viewT view,
                     Hash hash,
                     KeyEqual key_equal) {
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = block_size * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;
  __shared__ Value writeBuffer[block_size];
  
  while(first + key_idx < last) {
    auto key = *(first + key_idx);
    auto found = view.find(tile, key, hash, key_equal);
    
    /* 
     * The ld.relaxed.gpu instruction used in view.find causes L1 to 
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    if(tile.thread_rank() == 0) {
      writeBuffer[threadIdx.x / tile_size] =
        found == view.end()
          ? view.get_empty_value_sentinel()
          : found->second.load(cuda::std::memory_order_relaxed);
    }
    __syncthreads();
    if(tile.thread_rank() == 0) {
      *(output_begin + key_idx) = writeBuffer[threadIdx.x / tile_size];
    }
    key_idx += (gridDim.x * block_size) / tile_size;
  }
}

/**
 * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
 * 
 * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
 *
 * @tparam block_size The size of the thread block
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type 
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of booleans for the presence of each key
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template<std::size_t block_size,
         typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void contains(InputIt first,
                         InputIt last,
                         OutputIt output_begin,
                         viewT view,
                         Hash hash,
                         KeyEqual key_equal) {
  auto tid = block_size * blockIdx.x + threadIdx.x;
  auto key_idx = tid;
  __shared__ bool writeBuffer[block_size];
  
  while(first + key_idx < last) {
    auto key = *(first + key_idx);
    
    /* 
     * The ld.relaxed.gpu instruction used in view.find causes L1 to 
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    writeBuffer[threadIdx.x] = view.contains(key, hash, key_equal);
    __syncthreads();
    *(output_begin + key_idx) = writeBuffer[threadIdx.x];
    key_idx += gridDim.x * block_size;
  }
}

/**
 * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
 * 
 * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
 * Uses the CUDA Cooperative Groups API to leverage groups of multiple threads to perform the 
 * contains operation for each key. This provides a significant boost in throughput compared 
 * to the non Cooperative Group `contains` at moderate to high load factors.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform
 * inserts
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type 
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of booleans for the presence of each key
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template<std::size_t block_size, uint32_t tile_size,
         typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void contains(InputIt first,
                         InputIt last,
                         OutputIt output_begin,
                         viewT view,
                         Hash hash,
                         KeyEqual key_equal) {
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = block_size * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;
  __shared__ bool writeBuffer[block_size];
  
  while(first + key_idx < last) {
    auto key = *(first + key_idx);
    auto found = view.contains(tile, key, hash, key_equal);
    
    /* 
     * The ld.relaxed.gpu instruction used in view.find causes L1 to 
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    if(tile.thread_rank() == 0) {
      writeBuffer[threadIdx.x / tile_size] = found;
    }
    __syncthreads();
    if(tile.thread_rank() == 0) {
      *(output_begin + key_idx) = writeBuffer[threadIdx.x / tile_size];
    }
    key_idx += (gridDim.x * block_size) / tile_size;
  }
}

} // namespace detail
} // namespace cuco
