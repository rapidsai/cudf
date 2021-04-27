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

#pragma once

#include <cooperative_groups.h>
#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <cuco/detail/dynamic_map_kernels.cuh>
#include <cuco/detail/error.hpp>
#include <cuco/static_map.cuh>
#include <cuda/std/atomic>

namespace cuco {

/**
 * @brief A GPU-accelerated, unordered, associative container of key-value
 * pairs with unique keys
 *
 * Automatically grows capacity as necessary until device memory runs out.
 *
 * Allows constant time concurrent inserts or concurrent find operations (not
 * concurrent insert and find) from threads in device code.
 *
 * Current limitations:
 * - Requires keys that are Arithmetic
 * - Does not support erasing keys
 * - Capacity does not shrink automatically
 * - Requires the user to specify sentinel values for both key and mapped value
 *   to indicate empty slots
 * - Does not support concurrent insert and find operations
 *
 * The `dynamic_map` supports host-side "bulk" operations which include `insert`, `find`
 * and `contains`. These are to be used when there are a large number of keys to insert
 * or lookup in the map. For example, given a range of keys specified by device-accessible
 * iterators, the bulk `insert` function will insert all keys into the map.
 *
 * Example:
 * \code{.cpp}
 * int empty_key_sentinel = -1;
 * int empty_value_sentine = -1;
 *
 * // Constructs a map with 100,000 initial slots using -1 and -1 as the empty key/value
 * // sentinels. Performs one bulk insert of 50,000 keys and a second bulk insert of
 * // 100,000 keys. The map automatically increases capacity to accomodate the excess keys
 * // within the second insert.
 *
 * dynamic_map<int, int> m{100'000, empty_key_sentinel, empty_value_sentinel};
 *
 * // Create a sequence of pairs {{0,0}, {1,1}, ... {i,i}}
 * thrust::device_vector<thrust::pair<int,int>> pairs_0(50'000);
 * thrust::transform(thrust::make_counting_iterator(0),
 *                   thrust::make_counting_iterator(pairs_0.size()),
 *                   pairs_0.begin(),
 *                   []__device__(auto i){ return thrust::make_pair(i,i); };
 *
 * thrust::device_vector<thrust::pair<int,int>> pairs_1(100'000);
 * thrust::transform(thrust::make_counting_iterator(50'000),
 *                   thrust::make_counting_iterator(pairs_1.size()),
 *                   pairs_1.begin(),
 *                   []__device__(auto i){ return thrust::make_pair(i,i); };
 *
 * // Inserts all pairs into the map
 * m.insert(pairs_0.begin(), pairs_0.end());
 * m.insert(pairs_1.begin(), pairs_1.end());
 * \endcode
 *
 * @tparam Key Arithmetic type used for key
 * @tparam Value Type of the mapped values
 * @tparam Scope The scope in which insert/find/contains will be performed by
 * individual threads.
 * @tparam Allocator Type of allocator used to allocate submap device storage
 */
template <typename Key,
          typename Value,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          typename Allocator       = cuco::cuda_allocator<char>>
class dynamic_map {
  static_assert(std::is_arithmetic<Key>::value, "Unsupported, non-arithmetic key type.");

 public:
  using key_type                  = Key;
  using mapped_type               = Value;
  using atomic_ctr_type           = cuda::atomic<std::size_t, Scope>;
  using view_type                 = typename static_map<Key, Value, Scope>::device_view;
  using mutable_view_type         = typename static_map<Key, Value, Scope>::device_mutable_view;
  dynamic_map(dynamic_map const&) = delete;
  dynamic_map(dynamic_map&&)      = delete;
  dynamic_map& operator=(dynamic_map const&) = delete;
  dynamic_map& operator=(dynamic_map&&) = delete;

  /**
   * @brief Construct a dynamically-sized map with the specified initial capacity, growth factor and
   * sentinel values.
   *
   * The capacity of the map will automatically increase as the user adds key/value pairs using
   * `insert`.
   *
   * Capacity increases by a factor of growth_factor each time the size of the map exceeds a
   * threshold occupancy. The performance of `find` and `contains` decreases somewhat each time the
   * map's capacity grows.
   *
   * The `empty_key_sentinel` and `empty_value_sentinel` values are reserved and
   * undefined behavior results from attempting to insert any key/value pair
   * that contains either.
   *
   * @param initial_capacity The initial number of slots in the map
   * @param growth_factor The factor by which the capacity increases when resizing
   * @param empty_key_sentinel The reserved key value for empty slots
   * @param empty_value_sentinel The reserved mapped value for empty slots
   * @param alloc Allocator used to allocate submap device storage
   */
  dynamic_map(std::size_t initial_capacity,
              Key empty_key_sentinel,
              Value empty_value_sentinel,
              Allocator const& alloc = Allocator{});

  /**
   * @brief Destroy the map and frees its contents
   *
   */
  ~dynamic_map();

  /**
   * @brief Grows the capacity of the map so there is enough space for `n` key/value pairs.
   *
   * If there is already enough space for `n` key/value pairs, the capacity remains the same.
   *
   * @param n The number of key value pairs for which there must be space
   */
  void reserve(std::size_t n);

  /**
   * @brief Inserts all key/value pairs in the range `[first, last)`.
   *
   * If multiple keys in `[first, last)` compare equal, it is unspecified which
   * element is inserted.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `value_type`
   * @tparam Hash Unary callable type
   * @tparam KeyEqual Binary callable type
   * @param first Beginning of the sequence of key/value pairs
   * @param last End of the sequence of key/value pairs
   * @param hash The unary function to apply to hash each key
   * @param key_equal The binary function to compare two keys for equality
   */
  template <typename InputIt,
            typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void insert(InputIt first, InputIt last, Hash hash = Hash{}, KeyEqual key_equal = KeyEqual{});

  /**
   * @brief Finds the values corresponding to all keys in the range `[first, last)`.
   *
   * If the key `*(first + i)` exists in the map, copies its associated value to `(output_begin +
   * i)`. Else, copies the empty value sentinel.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is
   * convertible to the map's `mapped_type`
   * @tparam Hash Unary callable type
   * @tparam KeyEqual Binary callable type
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of values retrieved for each key
   * @param hash The unary function to apply to hash each key
   * @param key_equal The binary function to compare two keys for equality
   */
  template <typename InputIt,
            typename OutputIt,
            typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void find(InputIt first,
            InputIt last,
            OutputIt output_begin,
            Hash hash          = Hash{},
            KeyEqual key_equal = KeyEqual{});

  /**
   * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
   *
   * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is
   * convertible to the map's `mapped_type`
   * @tparam Hash Unary callable type
   * @tparam KeyEqual Binary callable type
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
   * @param hash The unary function to apply to hash each key
   * @param key_equal The binary function to compare two keys for equality
   */
  template <typename InputIt,
            typename OutputIt,
            typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void contains(InputIt first,
                InputIt last,
                OutputIt output_begin,
                Hash hash          = Hash{},
                KeyEqual key_equal = KeyEqual{});

  /**
   * @brief Gets the current number of elements in the map
   *
   * @return The current number of elements in the map
   */
  std::size_t get_size() const noexcept { return size_; }

  /**
   * @brief Gets the maximum number of elements the hash map can hold.
   *
   * @return The maximum number of elements the hash map can hold
   */
  std::size_t get_capacity() const noexcept { return capacity_; }

  /**
   * @brief Gets the load factor of the hash map.
   *
   * @return The load factor of the hash map
   */
  float get_load_factor() const noexcept { return static_cast<float>(size_) / capacity_; }

 private:
  key_type empty_key_sentinel_{};       ///< Key value that represents an empty slot
  mapped_type empty_value_sentinel_{};  ///< Initial value of empty slot
  std::size_t size_{};                  ///< Number of keys in the map
  std::size_t capacity_{};              ///< Maximum number of keys that can be inserted
  float max_load_factor_{};             ///< Max load factor before capacity growth

  std::vector<std::unique_ptr<static_map<key_type, mapped_type, Scope>>>
    submaps_;                                      ///< vector of pointers to each submap
  thrust::device_vector<view_type> submap_views_;  ///< vector of device views for each submap
  thrust::device_vector<mutable_view_type>
    submap_mutable_views_;          ///< vector of mutable device views for each submap
  std::size_t min_insert_size_{};   ///< min remaining capacity of submap for insert
  atomic_ctr_type* num_successes_;  ///< number of successfully inserted keys on insert
  Allocator alloc_{};  ///< Allocator passed to submaps to allocate their device storage
};
}  // namespace cuco

#include <cuco/detail/dynamic_map.inl>