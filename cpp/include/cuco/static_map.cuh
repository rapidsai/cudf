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
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <cub/cub.cuh>
#include <cuda/std/atomic>
#include <memory>

#include <cuco/allocator.hpp>

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11000) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
#define CUCO_HAS_CUDA_BARRIER
#endif

#if defined(CUCO_HAS_CUDA_BARRIER)
#include <cuda/barrier>
#endif

#include <cuco/detail/error.hpp>
#include <cuco/detail/hash_functions.cuh>
#include <cuco/detail/pair.cuh>
#include <cuco/detail/static_map_kernels.cuh>

namespace cuco {

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
class dynamic_map;

/**
 * @brief A GPU-accelerated, unordered, associative container of key-value
 * pairs with unique keys.
 *
 * Allows constant time concurrent inserts or concurrent find operations (not
 * concurrent insert and find) from threads in device code.
 *
 * Current limitations:
 * - Requires keys and values that are trivially copyable and have unique object representations
 *    - Comparisons against the "sentinel" values will always be done with bitwise comparisons.
 *      Therefore, the objects must have unique, bitwise object representations (e.g., no padding
 *      bits).
 * - Does not support erasing keys
 * - Capacity is fixed and will not grow automatically
 * - Requires the user to specify sentinel values for both key and mapped value
 * to indicate empty slots
 * - Does not support concurrent insert and find operations
 *
 * The `static_map` supports two types of operations:
 * - Host-side "bulk" operations
 * - Device-side "singular" operations
 *
 * The host-side bulk operations include `insert`, `find`, and `contains`. These
 * APIs should be used when there are a large number of keys to insert or lookup
 * in the map. For example, given a range of keys specified by device-accessible
 * iterators, the bulk `insert` function will insert all keys into the map.
 *
 * The singular device-side operations allow individual threads to perform
 * independent insert or find/contains operations from device code. These
 * operations are accessed through non-owning, trivially copyable "view" types:
 * `device_view` and `mutable_device_view`. The `device_view` class is an
 * immutable view that allows only non-modifying operations such as `find` or
 * `contains`. The `mutable_device_view` class only allows `insert` operations.
 * The two types are separate to prevent erroneous concurrent insert/find
 * operations.
 *
 * Example:
 * \code{.cpp}
 * int empty_key_sentinel = -1;
 * int empty_value_sentine = -1;
 *
 * // Constructs a map with 100,000 slots using -1 and -1 as the empty key/value
 * // sentinels. Note the capacity is chosen knowing we will insert 50,000 keys,
 * // for an load factor of 50%.
 * static_map<int, int> m{100'000, empty_key_sentinel, empty_value_sentinel};
 *
 * // Create a sequence of pairs {{0,0}, {1,1}, ... {i,i}}
 * thrust::device_vector<thrust::pair<int,int>> pairs(50,000);
 * thrust::transform(thrust::make_counting_iterator(0),
 *                   thrust::make_counting_iterator(pairs.size()),
 *                   pairs.begin(),
 *                   []__device__(auto i){ return thrust::make_pair(i,i); };
 *
 *
 * // Inserts all pairs into the map
 * m.insert(pairs.begin(), pairs.end());
 *
 * // Get a `device_view` and passes it to a kernel where threads may perform
 * // `find/contains` lookups
 * kernel<<<...>>>(m.get_device_view());
 * \endcode
 *
 *
 * @tparam Key Arithmetic type used for key
 * @tparam Value Type of the mapped values
 * @tparam Scope The scope in which insert/find operations will be performed by
 * individual threads.
 * @tparam Allocator Type of allocator used for device storage
 */
template <typename Key,
          typename Value,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          typename Allocator       = cuco::cuda_allocator<char>>
class static_map {
  template <typename T>
  static constexpr bool is_CAS_safe =
    std::is_trivially_copyable_v<T>and std::has_unique_object_representations_v<T>;

  static_assert(is_CAS_safe<Key>,
                "Key type must be trivially copyable and have unique object representation.");
  static_assert(is_CAS_safe<Value>,
                "Value type must be trivially copyable and have unique object representation.");

  friend class dynamic_map<Key, Value, Scope, Allocator>;

 public:
  using value_type         = cuco::pair_type<Key, Value>;
  using key_type           = Key;
  using mapped_type        = Value;
  using atomic_key_type    = cuda::atomic<key_type, Scope>;
  using atomic_mapped_type = cuda::atomic<mapped_type, Scope>;
  using pair_atomic_type   = cuco::pair_type<atomic_key_type, atomic_mapped_type>;
  using atomic_ctr_type    = cuda::atomic<std::size_t, Scope>;
  using allocator_type     = Allocator;
  using slot_allocator_type =
    typename std::allocator_traits<Allocator>::rebind_alloc<pair_atomic_type>;

  static_map(static_map const&) = delete;
  static_map(static_map&&)      = delete;
  static_map& operator=(static_map const&) = delete;
  static_map& operator=(static_map&&) = delete;

  /**
   * @brief Construct a fixed-size map with the specified capacity and sentinel values.
   * @brief Construct a statically sized map with the specified number of slots
   * and sentinel values.
   *
   * The capacity of the map is fixed. Insert operations will not automatically
   * grow the map. Attempting to insert equal to or more unique keys than the capacity
   * of the map results in undefined behavior (there should be at least one empty slot).
   *
   * Performance begins to degrade significantly beyond a load factor of ~70%.
   * For best performance, choose a capacity that will keep the load factor
   * below 70%. E.g., if inserting `N` unique keys, choose a capacity of
   * `N * (1/0.7)`.
   *
   * The `empty_key_sentinel` and `empty_value_sentinel` values are reserved and
   * undefined behavior results from attempting to insert any key/value pair
   * that contains either.
   *
   * @param capacity The total number of slots in the map
   * @param empty_key_sentinel The reserved key value for empty slots
   * @param empty_value_sentinel The reserved mapped value for empty slots
   * @param alloc Allocator used for allocating device storage
   */
  static_map(std::size_t capacity,
             Key empty_key_sentinel,
             Value empty_value_sentinel,
             Allocator const& alloc = Allocator{});

  /**
   * @brief Destroys the map and frees its contents.
   *
   */
  ~static_map();

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

 private:
  class device_view_base {
   protected:
    // Import member type definitions from `static_map`
    using value_type     = value_type;
    using key_type       = Key;
    using mapped_type    = Value;
    using iterator       = pair_atomic_type*;
    using const_iterator = pair_atomic_type const*;

   private:
    pair_atomic_type* slots_{};     ///< Pointer to flat slots storage
    std::size_t capacity_{};        ///< Total number of slots
    Key empty_key_sentinel_{};      ///< Key value that represents an empty slot
    Value empty_value_sentinel_{};  ///< Initial Value of empty slot

   protected:
    __host__ __device__ device_view_base(pair_atomic_type* slots,
                                         std::size_t capacity,
                                         Key empty_key_sentinel,
                                         Value empty_value_sentinel) noexcept
      : slots_{slots},
        capacity_{capacity},
        empty_key_sentinel_{empty_key_sentinel},
        empty_value_sentinel_{empty_value_sentinel}
    {
    }

    /**
     * @brief Gets slots array.
     *
     * @return Slots array
     */
    __device__ pair_atomic_type* get_slots() noexcept { return slots_; }

    /**
     * @brief Gets slots array.
     *
     * @return Slots array
     */
    __device__ pair_atomic_type const* get_slots() const noexcept { return slots_; }

    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * @tparam Hash Unary callable type
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the initial slot for `k`
     */
    template <typename Hash>
    __device__ iterator initial_slot(Key const& k, Hash hash) noexcept
    {
      return &slots_[hash(k) % capacity_];
    }

    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * @tparam Hash Unary callable type
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the initial slot for `k`
     */
    template <typename Hash>
    __device__ const_iterator initial_slot(Key const& k, Hash hash) const noexcept
    {
      return &slots_[hash(k) % capacity_];
    }

    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * To be used for Cooperative Group based probing.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @param g the Cooperative Group for which the initial slot is needed
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the initial slot for `k`
     */
    template <typename CG, typename Hash>
    __device__ iterator initial_slot(CG g, Key const& k, Hash hash) noexcept
    {
      return &slots_[(hash(k) + g.thread_rank()) % capacity_];
    }

    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * To be used for Cooperative Group based probing.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @param g the Cooperative Group for which the initial slot is needed
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the initial slot for `k`
     */
    template <typename CG, typename Hash>
    __device__ const_iterator initial_slot(CG g, Key const& k, Hash hash) const noexcept
    {
      return &slots_[(hash(k) + g.thread_rank()) % capacity_];
    }

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot.
     *
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    __device__ iterator next_slot(iterator s) noexcept { return (++s < end()) ? s : begin_slot(); }

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot.
     *
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    __device__ const_iterator next_slot(const_iterator s) const noexcept
    {
      return (++s < end()) ? s : begin_slot();
    }

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot. To
     * be used for Cooperative Group based probing.
     *
     * @tparam CG The Cooperative Group type
     * @param g The Cooperative Group for which the next slot is needed
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    template <typename CG>
    __device__ iterator next_slot(CG g, iterator s) noexcept
    {
      uint32_t index = s - slots_;
      return &slots_[(index + g.size()) % capacity_];
    }

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot. To
     * be used for Cooperative Group based probing.
     *
     * @tparam CG The Cooperative Group type
     * @param g The Cooperative Group for which the next slot is needed
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    template <typename CG>
    __device__ const_iterator next_slot(CG g, const_iterator s) const noexcept
    {
      uint32_t index = s - slots_;
      return &slots_[(index + g.size()) % capacity_];
    }

   public:
    /**
     * @brief Gets the maximum number of elements the hash map can hold.
     *
     * @return The maximum number of elements the hash map can hold
     */
    __host__ __device__ std::size_t get_capacity() const noexcept { return capacity_; }

    /**
     * @brief Gets the sentinel value used to represent an empty key slot.
     *
     * @return The sentinel value used to represent an empty key slot
     */
    __host__ __device__ Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

    /**
     * @brief Gets the sentinel value used to represent an empty value slot.
     *
     * @return The sentinel value used to represent an empty value slot
     */
    __host__ __device__ Value get_empty_value_sentinel() const noexcept
    {
      return empty_value_sentinel_;
    }

    /**
     * @brief Returns iterator to the first slot.
     *
     * @note Unlike `std::map::begin()`, the `begin_slot()` iterator does _not_ point to the first
     * occupied slot. Instead, it refers to the first slot in the array of contiguous slot storage.
     * Iterating from `begin_slot()` to `end_slot()` will iterate over all slots, including those
     * both empty and filled.
     *
     * There is no `begin()` iterator to avoid confusion as it is not possible to provide an
     * iterator over only the filled slots.
     *
     * @return Iterator to the first slot
     */
    __device__ iterator begin_slot() noexcept { return slots_; }

    /**
     * @brief Returns iterator to the first slot.
     *
     * @note Unlike `std::map::begin()`, the `begin_slot()` iterator does _not_ point to the first
     * occupied slot. Instead, it refers to the first slot in the array of contiguous slot storage.
     * Iterating from `begin_slot()` to `end_slot()` will iterate over all slots, including those
     * both empty and filled.
     *
     * There is no `begin()` iterator to avoid confusion as it is not possible to provide an
     * iterator over only the filled slots.
     *
     * @return Iterator to the first slot
     */
    __device__ const_iterator begin_slot() const noexcept { return slots_; }

    /**
     * @brief Returns a const_iterator to one past the last slot.
     *
     * @return A const_iterator to one past the last slot
     */
    __host__ __device__ const_iterator end_slot() const noexcept { return slots_ + capacity_; }

    /**
     * @brief Returns an iterator to one past the last slot.
     *
     * @return An iterator to one past the last slot
     */
    __host__ __device__ iterator end_slot() noexcept { return slots_ + capacity_; }

    /**
     * @brief Returns a const_iterator to one past the last slot.
     *
     * `end()` calls `end_slot()` and is provided for convenience for those familiar with checking
     * an iterator returned from `find()` against the `end()` iterator.
     *
     * @return A const_iterator to one past the last slot
     */
    __host__ __device__ const_iterator end() const noexcept { return end_slot(); }

    /**
     * @brief Returns an iterator to one past the last slot.
     *
     * `end()` calls `end_slot()` and is provided for convenience for those familiar with checking
     * an iterator returned from `find()` against the `end()` iterator.
     *
     * @return An iterator to one past the last slot
     */
    __host__ __device__ iterator end() noexcept { return end_slot(); }
  };

 public:
  /**
   * @brief Mutable, non-owning view-type that may be used in device code to
   * perform singular inserts into the map.
   *
   * `device_mutable_view` is trivially-copyable and is intended to be passed by
   * value.
   *
   * Example:
   * \code{.cpp}
   * cuco::static_map<int,int> m{100'000, -1, -1};
   *
   * // Inserts a sequence of pairs {{0,0}, {1,1}, ... {i,i}}
   * thrust::for_each(thrust::make_counting_iterator(0),
   *                  thrust::make_counting_iterator(50'000),
   *                  [map = m.get_mutable_device_view()]
   *                  __device__ (auto i) mutable {
   *                     map.insert(thrust::make_pair(i,i));
   *                  });
   * \endcode
   */
  class device_mutable_view : public device_view_base {
   public:
    using value_type     = typename device_view_base::value_type;
    using key_type       = typename device_view_base::key_type;
    using mapped_type    = typename device_view_base::mapped_type;
    using iterator       = typename device_view_base::iterator;
    using const_iterator = typename device_view_base::const_iterator;
    /**
     * @brief Construct a mutable view of the first `capacity` slots of the
     * slots array pointed to by `slots`.
     *
     * @param slots Pointer to beginning of initialized slots array
     * @param capacity The number of slots viewed by this object
     * @param empty_key_sentinel The reserved value for keys to represent empty
     * slots
     * @param empty_value_sentinel The reserved value for mapped values to
     * represent empty slots
     */
     __host__ __device__ device_mutable_view(pair_atomic_type* slots,
                                             std::size_t capacity,
                                             Key empty_key_sentinel,
                                             Value empty_value_sentinel) noexcept
      : device_view_base{slots, capacity, empty_key_sentinel, empty_value_sentinel}
    {
    }

    /**
     * @brief Inserts the specified key/value pair into the map.
     *
     * Returns a pair consisting of an iterator to the inserted element (or to
     * the element that prevented the insertion) and a `bool` denoting whether
     * the insertion took place.
     *
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     * @param insert_pair The pair to insert
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys for
     * equality
     * @return `true` if the insert was successful, `false` otherwise.
     */
    template <typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ bool insert(value_type const& insert_pair,
                           Hash hash          = Hash{},
                           KeyEqual key_equal = KeyEqual{}) noexcept;
    /**
     * @brief Inserts the specified key/value pair into the map.
     *
     * Returns a pair consisting of an iterator to the inserted element (or to
     * the element that prevented the insertion) and a `bool` denoting whether
     * the insertion took place. Uses the CUDA Cooperative Groups API to
     * to leverage multiple threads to perform a single insert. This provides a
     * significant boost in throughput compared to the non Cooperative Group
     * `insert` at moderate to high load factors.
     *
     * @tparam Cooperative Group type
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     *
     * @param g The Cooperative Group that performs the insert
     * @param insert_pair The pair to insert
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys for
     * equality
     * @return `true` if the insert was successful, `false` otherwise.
     */
    template <typename CG,
              typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ bool insert(CG g,
                           value_type const& insert_pair,
                           Hash hash          = Hash{},
                           KeyEqual key_equal = KeyEqual{}) noexcept;

  };  // class device mutable view

  /**
   * @brief Non-owning view-type that may be used in device code to
   * perform singular find and contains operations for the map.
   *
   * `device_view` is trivially-copyable and is intended to be passed by
   * value.
   *
   */
  class device_view : public device_view_base {
   public:
    using value_type     = typename device_view_base::value_type;
    using key_type       = typename device_view_base::key_type;
    using mapped_type    = typename device_view_base::mapped_type;
    using iterator       = typename device_view_base::iterator;
    using const_iterator = typename device_view_base::const_iterator;
    /**
     * @brief Construct a view of the first `capacity` slots of the
     * slots array pointed to by `slots`.
     *
     * @param slots Pointer to beginning of initialized slots array
     * @param capacity The number of slots viewed by this object
     * @param empty_key_sentinel The reserved value for keys to represent empty
     * slots
     * @param empty_value_sentinel The reserved value for mapped values to
     * represent empty slots
     */
     __host__ __device__ device_view(pair_atomic_type* slots,
                                     std::size_t capacity,
                                     Key empty_key_sentinel,
                                     Value empty_value_sentinel) noexcept
      : device_view_base{slots, capacity, empty_key_sentinel, empty_value_sentinel}
    {
    }

    /**
     * @brief Makes a copy of given `device_view` using non-owned memory.
     *
     * This function is intended to be used to create shared memory copies of small static maps, although global memory can be used as well.
     *
     * Example:
     * @code{.cpp}
     * template <typename MapType, int CAPACITY>
     * __global__ void use_device_view(const typename MapType::device_view device_view,
     *                                 map_key_t const* const keys_to_search,
     *                                 map_value_t* const values_found,
     *                                 const size_t number_of_elements)
     * {
     *     const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
     *
     *     __shared__ typename MapType::pair_atomic_type sm_buffer[CAPACITY];
     *
     *     auto g = cg::this_thread_block();
     *
     *     const map_t::device_view sm_static_map = device_view.make_copy(g,
     *                                                                    sm_buffer);
     *
     *     for (size_t i = g.thread_rank(); i < number_of_elements; i += g.size())
     *     {
     *         values_found[i] = sm_static_map.find(keys_to_search[i])->second;
     *     }
     * }
     * @endcode
     *
     * @tparam CG The type of the cooperative thread group
     * @param g The ooperative thread group used to copy the slots
     * @param source_device_view `device_view` to copy from
     * @param memory_to_use Array large enough to support `capacity` elements. Object does not take the ownership of the memory
     * @return Copy of passed `device_view`
     */
    template <typename CG>
    __device__ static device_view make_copy(CG g,
                                            pair_atomic_type* const memory_to_use,
                                            device_view source_device_view) noexcept
    {
#if defined(CUDA_HAS_CUDA_BARRIER)
      __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
      if (g.thread_rank() == 0) {
        init(&barrier, g.size());
      }
      g.sync();

      cuda::memcpy_async(g,
                         memory_to_use,
                         source_device_view.get_slots(),
                         sizeof(pair_atomic_type) * source_device_view.get_capacity(),
                         barrier);

      barrier.arrive_and_wait();
#else
      pair_atomic_type const* const slots_ptr = source_device_view.get_slots();
      for (std::size_t i = g.thread_rank(); i < source_device_view.get_capacity(); i += g.size())
      {
        new (&memory_to_use[i].first) atomic_key_type{slots_ptr[i].first.load(cuda::memory_order_relaxed)};
        new (&memory_to_use[i].second) atomic_mapped_type{slots_ptr[i].second.load(cuda::memory_order_relaxed)};
      }
      g.sync();
#endif

      return device_view(memory_to_use,
                         source_device_view.get_capacity(),
                         source_device_view.get_empty_key_sentinel(),
                         source_device_view.get_empty_value_sentinel());
    }

    /**
     * @brief Finds the value corresponding to the key `k`.
     *
     * Returns an iterator to the pair whose key is equivalent to `k`.
     * If no such pair exists, returns `end()`.
     *
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return An iterator to the position at which the key/value pair
     * containing `k` was inserted
     */
    template <typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ iterator find(Key const& k,
                             Hash hash          = Hash{},
                             KeyEqual key_equal = KeyEqual{}) noexcept;

    /** @brief Finds the value corresponding to the key `k`.
     *
     * Returns a const_iterator to the pair whose key is equivalent to `k`.
     * If no such pair exists, returns `end()`.
     *
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return An iterator to the position at which the key/value pair
     * containing `k` was inserted
     */
    template <typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ const_iterator find(Key const& k,
                                   Hash hash          = Hash{},
                                   KeyEqual key_equal = KeyEqual{}) const noexcept;

    /**
     * @brief Finds the value corresponding to the key `k`.
     *
     * Returns an iterator to the pair whose key is equivalent to `k`.
     * If no such pair exists, returns `end()`. Uses the CUDA Cooperative Groups API to
     * to leverage multiple threads to perform a single find. This provides a
     * significant boost in throughput compared to the non Cooperative Group
     * `find` at moderate to high load factors.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the find
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return An iterator to the position at which the key/value pair
     * containing `k` was inserted
     */
    template <typename CG,
              typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ iterator
    find(CG g, Key const& k, Hash hash = Hash{}, KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Finds the value corresponding to the key `k`.
     *
     * Returns a const_iterator to the pair whose key is equivalent to `k`.
     * If no such pair exists, returns `end()`. Uses the CUDA Cooperative Groups API to
     * to leverage multiple threads to perform a single find. This provides a
     * significant boost in throughput compared to the non Cooperative Group
     * `find` at moderate to high load factors.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the find
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return An iterator to the position at which the key/value pair
     * containing `k` was inserted
     */
    template <typename CG,
              typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ const_iterator
    find(CG g, Key const& k, Hash hash = Hash{}, KeyEqual key_equal = KeyEqual{}) const noexcept;

    /**
     * @brief Indicates whether the key `k` was inserted into the map.
     *
     * If the key `k` was inserted into the map, find returns
     * true. Otherwise, it returns false.
     *
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return A boolean indicating whether the key/value pair
     * containing `k` was inserted
     */
    template <typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ bool contains(Key const& k,
                             Hash hash          = Hash{},
                             KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Indicates whether the key `k` was inserted into the map.
     *
     * If the key `k` was inserted into the map, find returns
     * true. Otherwise, it returns false. Uses the CUDA Cooperative Groups API to
     * to leverage multiple threads to perform a single contains operation. This provides a
     * significant boost in throughput compared to the non Cooperative Group
     * `contains` at moderate to high load factors.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the contains operation
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return A boolean indicating whether the key/value pair
     * containing `k` was inserted
     */
    template <typename CG,
              typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ bool contains(CG g,
                             Key const& k,
                             Hash hash          = Hash{},
                             KeyEqual key_equal = KeyEqual{}) noexcept;
  };  // class device_view

  /**
   * @brief Gets the maximum number of elements the hash map can hold.
   *
   * @return The maximum number of elements the hash map can hold
   */
  std::size_t get_capacity() const noexcept { return capacity_; }

  /**
   * @brief Gets the number of elements in the hash map.
   *
   * @return The number of elements in the map
   */
  std::size_t get_size() const noexcept { return size_; }

  /**
   * @brief Gets the load factor of the hash map.
   *
   * @return The load factor of the hash map
   */
  float get_load_factor() const noexcept { return static_cast<float>(size_) / capacity_; }

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

  /**
   * @brief Gets the sentinel value used to represent an empty value slot.
   *
   * @return The sentinel value used to represent an empty value slot
   */
  Value get_empty_value_sentinel() const noexcept { return empty_value_sentinel_; }

  /**
   * @brief Constructs a device_view object based on the members of the `static_map` object.
   *
   * @return A device_view object based on the members of the `static_map` object
   */
  device_view get_device_view() const noexcept
  {
    return device_view(slots_, capacity_, empty_key_sentinel_, empty_value_sentinel_);
  }

  /**
   * @brief Constructs a device_mutable_view object based on the members of the `static_map` object
   *
   * @return A device_mutable_view object based on the members of the `static_map` object
   */
  device_mutable_view get_device_mutable_view() const noexcept
  {
    return device_mutable_view(slots_, capacity_, empty_key_sentinel_, empty_value_sentinel_);
  }

 private:
  pair_atomic_type* slots_{nullptr};      ///< Pointer to flat slots storage
  std::size_t capacity_{};                ///< Total number of slots
  std::size_t size_{};                    ///< Number of keys in map
  Key empty_key_sentinel_{};              ///< Key value that represents an empty slot
  Value empty_value_sentinel_{};          ///< Initial value of empty slot
  atomic_ctr_type* num_successes_{};      ///< Number of successfully inserted keys on insert
  slot_allocator_type slot_allocator_{};  ///< Allocator used to allocate slots
};
}  // namespace cuco

#include <cuco/detail/static_map.inl>
