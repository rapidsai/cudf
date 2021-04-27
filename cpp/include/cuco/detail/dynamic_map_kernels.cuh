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

namespace cuco {
namespace detail {
namespace cg = cooperative_groups;

/**
 * @brief Inserts all key/value pairs in the range `[first, last)`. 
 *  
 * If multiple keys in `[first, last)` compare equal, it is unspecified which
 * element is inserted.
 * 
 * @tparam block_size 
 * @tparam pair_type Type of the pairs contained in the map
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `value_type`
 * @tparam viewT Type of the `static_map` device views
 * @tparam mutableViewT Type of the `static_map` device mutable views 
 * @tparam atomicT Type of atomic storage
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of key/value pairs
 * @param last End of the sequence of key/value pairs
 * @param submap_views Array of `static_map::device_view` objects used to
 * perform `contains` operations on each underlying `static_map`
 * @param submap_mutable_views Array of `static_map::device_mutable_view` objects 
 * used to perform an `insert` into the target `static_map` submap
 * @param num_successes The number of successfully inserted key/value pairs
 * @param insert_idx The index of the submap we are inserting into
 * @param num_submaps The total number of submaps in the map
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function used to compare two keys for equality
 */
template<uint32_t block_size,
         typename pair_type,
         typename InputIt,
         typename viewT,
         typename mutableViewT,
         typename atomicT,
         typename Hash, 
         typename KeyEqual>
__global__ void insert(InputIt first,
                       InputIt last,
                       viewT* submap_views,
                       mutableViewT* submap_mutable_views,
                       atomicT* num_successes,
                       uint32_t insert_idx,
                       uint32_t num_submaps,
                       Hash hash,
                       KeyEqual key_equal) {
  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_successes = 0;
  
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;

  while(first + tid < last) {
    pair_type insert_pair = *(first + tid);
    auto exists = false;
    
    // manually check for duplicates in those submaps we are not inserting into
    for(auto i = 0; i < num_submaps; ++i) {
      if(i != insert_idx) {
        exists = submap_views[i].contains(insert_pair.first, hash, key_equal);
        if(exists) {
          break;
        }
      }
    }
    if(!exists) {
      if(submap_mutable_views[insert_idx].insert(insert_pair, hash, key_equal)) {
        thread_num_successes++;
      }
    }

    tid += gridDim.x * blockDim.x;
  }
  
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
 * @tparam pair_type Type of the pairs contained in the map
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `value_type`
 * @tparam viewT Type of the `static_map` device views
 * @tparam mutableViewT Type of the `static_map` device mutable views 
 * @tparam atomicT Type of atomic storage
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of key/value pairs
 * @param last End of the sequence of key/value pairs
 * @param submap_views Array of `static_map::device_view` objects used to
 * perform `contains` operations on each underlying `static_map`
 * @param submap_mutable_views Array of `static_map::device_mutable_view` objects 
 * used to perform an `insert` into the target `static_map` submap
 * @param num_successes The number of successfully inserted key/value pairs
 * @param insert_idx The index of the submap we are inserting into
 * @param num_submaps The total number of submaps in the map
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function used to compare two keys for equality
 */
template<uint32_t block_size, uint32_t tile_size,
         typename pair_type,
         typename InputIt,
         typename viewT,
         typename mutableViewT,
         typename atomicT,
         typename Hash, 
         typename KeyEqual>
__global__ void insert(InputIt first,
                       InputIt last,
                       viewT* submap_views,
                       mutableViewT* submap_mutable_views,
                       atomicT* num_successes,
                       uint32_t insert_idx,
                       uint32_t num_submaps,
                       Hash hash,
                       KeyEqual key_equal) {
  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_successes = 0;
  
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto it = first + tid / tile_size;

  while(it < last) {
    pair_type insert_pair = *it;
    auto exists = false;
    
    // manually check for duplicates in those submaps we are not inserting into
    for(auto i = 0; i < num_submaps; ++i) {
      if(i != insert_idx) {
        exists = submap_views[i].contains(tile, insert_pair.first, hash, key_equal);
        if(exists) {
          break;
        }
      }
    }
    if(!exists) {
      if(submap_mutable_views[insert_idx].insert(tile, insert_pair, hash, key_equal) && 
         tile.thread_rank() == 0) {
        thread_num_successes++;
      }
    }

    it += (gridDim.x * blockDim.x) / tile_size;
  }
  
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
 * @tparam block_size The number of threads in the thread block
 * @tparam Value The mapped value type for the map 
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is 
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of `static_map` device view
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of values retrieved for each key
 * @param submap_views Array of `static_map::device_view` objects used to
 * perform `find` operations on each underlying `static_map`
 * @param num_submaps The number of submaps in the map
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template<uint32_t block_size,
         typename Value,
         typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void find(InputIt first,
                     InputIt last,
                     OutputIt output_begin,
                     viewT* submap_views,
                     uint32_t num_submaps,
                     Hash hash,
                     KeyEqual key_equal) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto empty_value_sentinel = submap_views[0].get_empty_value_sentinel();
  __shared__ Value writeBuffer[block_size];

  while(first + tid < last) {
    auto key = *(first + tid);
    auto found_value = empty_value_sentinel;
    for(auto i = 0; i < num_submaps; ++i) {
      auto submap_view = submap_views[i];
      auto found = submap_view.find(key, hash, key_equal);
      if(found != submap_view.end()) {
        found_value = found->second;
        break;
      }
    }

    /* 
     * The ld.relaxed.gpu instruction used in view.find causes L1 to 
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    writeBuffer[threadIdx.x] = found_value;
    __syncthreads();
    *(output_begin + tid) = writeBuffer[threadIdx.x];
    tid += gridDim.x * blockDim.x;
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
 * @tparam block_size The number of threads in the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to
 * perform find operations
 * @tparam Value The mapped value type for the map
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is 
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of `static_map` device view
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of values retrieved for each key
 * @param submap_views Array of `static_map::device_view` objects used to
 * perform `find` operations on each underlying `static_map`
 * @param num_submaps The number of submaps in the map
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template<uint32_t block_size, uint32_t tile_size,
         typename Value,
         typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void find(InputIt first,
                     InputIt last,
                     OutputIt output_begin,
                     viewT* submap_views,
                     uint32_t num_submaps,
                     Hash hash,
                     KeyEqual key_equal) {
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;
  auto empty_value_sentinel = submap_views[0].get_empty_value_sentinel();
  __shared__ Value writeBuffer[block_size];

  while(first + key_idx < last) {
    auto key = *(first + key_idx);
    auto found_value = empty_value_sentinel;
    for(auto i = 0; i < num_submaps; ++i) {
      auto submap_view = submap_views[i];
      auto found = submap_view.find(tile, key, hash, key_equal);
      if(found != submap_view.end()) {
        found_value = found->second;
        break;
      }
    }

    /* 
     * The ld.relaxed.gpu instruction used in view.find causes L1 to 
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    if(tile.thread_rank() == 0) {
      writeBuffer[threadIdx.x / tile_size] = found_value;
    }
    __syncthreads();
    if(tile.thread_rank() == 0) {
      *(output_begin + key_idx) = writeBuffer[threadIdx.x / tile_size];
    }
    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }
}

/**
 * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
 * 
 * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
 *
 * @tparam block_size The number of threads in the thread block
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is 
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of `static_map` device view
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of booleans for the presence of each key
 * @param submap_views Array of `static_map::device_view` objects used to
 * perform `contains` operations on each underlying `static_map`
 * @param num_submaps The number of submaps in the map
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template<uint32_t block_size,
         typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void contains(InputIt first,
                         InputIt last,
                         OutputIt output_begin,
                         viewT* submap_views,
                         uint32_t num_submaps,
                         Hash hash,
                         KeyEqual key_equal) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ bool writeBuffer[block_size];

  while(first + tid < last) {
    auto key = *(first + tid);
    auto found = false;
    for(auto i = 0; i < num_submaps; ++i) {
      found = submap_views[i].contains(key, hash, key_equal);
      if(found) {
        break;
      }
    }
    
    /* 
     * The ld.relaxed.gpu instruction used in view.find causes L1 to 
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    writeBuffer[threadIdx.x] = found;
    __syncthreads();
    *(output_begin + tid) = writeBuffer[threadIdx.x];
    tid += gridDim.x * blockDim.x;
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
 * @tparam block_size The number of threads in the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to
 * perform find operations
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is 
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of `static_map` device view
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of booleans for the presence of each key
 * @param submap_views Array of `static_map::device_view` objects used to
 * perform `contains` operations on each underlying `static_map`
 * @param num_submaps The number of submaps in the map
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template<uint32_t block_size, uint32_t tile_size,
         typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void contains(InputIt first,
                         InputIt last,
                         OutputIt output_begin,
                         viewT* submap_views,
                         uint32_t num_submaps,
                         Hash hash,
                         KeyEqual key_equal) {
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;
  __shared__ bool writeBuffer[block_size];

  while(first + key_idx < last) {
    auto key = *(first + key_idx);
    auto found = false;
    for(auto i = 0; i < num_submaps; ++i) {
      found = submap_views[i].contains(tile, key, hash, key_equal);
      if(found) {
        break;
      }
    }

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
    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }
}
} // namespace detail
} // namespace cuco