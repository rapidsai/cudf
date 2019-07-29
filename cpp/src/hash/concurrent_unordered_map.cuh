/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef CONCURRENT_UNORDERED_MAP_CUH
#define CONCURRENT_UNORDERED_MAP_CUH

#include "managed_allocator.cuh"
#include "managed.cuh"
#include "hash_functions.cuh"
#include "helper_functions.cuh"
#include <utilities/device_atomics.cuh>

#include <iterator>
#include <type_traits>
#include <cassert>
#include <iostream>
#include <thrust/pair.h>
#include <limits>

namespace {
    template <std::size_t N> struct packed { using type = void; };
    template <> struct packed<sizeof(uint64_t)> { using type = uint64_t; };
    template <> struct packed<sizeof(uint32_t)> { using type = uint32_t; };
    template <typename pair_type>
    using packed_t = typename packed<sizeof(pair_type)>::type;

    /**---------------------------------------------------------------------------*
     * @brief Indicates if a pair type can be packed.
     *
     * When the size of the key,value pair being inserted into the hash table is
     * equal in size to a type where atomicCAS is natively supported, it is more
     * efficient to "pack" the pair and insert it with a single atomicCAS.
     *
     * @note Only integral key and value types may be packed because we use
     * bitwise equality comparison, which may not be valid for non-integral
     * types.
     *
     * @tparam pair_type The pair type in question
     * @return true If the pair type can be packed
     * @return false  If the pair type cannot be packed
     *---------------------------------------------------------------------------**/
    template <typename pair_type,
              typename key_type = typename pair_type::first_type,
              typename value_type = typename pair_type::second_type>
    constexpr bool is_packable() {
      return std::is_integral<key_type>::value and
             std::is_integral<value_type>::value and
             not std::is_void<packed_t<pair_type>>::value;
    }

    /**---------------------------------------------------------------------------*
     * @brief Allows viewing a pair in a packed representation
     *
     * Used as an optimization for inserting when a pair can be inserted with a
     * single atomicCAS
     *---------------------------------------------------------------------------**/
    template <typename pair_type, typename Enable = void>
    union pair_packer;

    template <typename pair_type>
    union pair_packer<pair_type, std::enable_if_t<is_packable<pair_type>()>> {
      using packed_type = packed_t<pair_type>;
      packed_type const packed;
      pair_type const pair;

      __device__ pair_packer(pair_type _pair) : pair{_pair} {}

      __device__ pair_packer(packed_type _packed) : packed{_packed} {}
    };
}

/**
 * Supports concurrent insert, but not concurrent insert and find.
 *
 * TODO:
 *  - add constructor that takes pointer to hash_table to avoid allocations
 *  - extend interface to accept streams
 */
template <typename Key,
          typename Element,
          typename Hasher = default_hash<Key>,
          typename Equality = equal_to<Key>,
          typename Allocator = legacy_allocator<thrust::pair<Key, Element> >>
class concurrent_unordered_map : public managed
{

public:
    using size_type = size_t;
    using hasher = Hasher;
    using key_equal = Equality;
    using allocator_type = Allocator;
    using key_type = Key;
    using mapped_type = Element;
    using value_type = thrust::pair<Key, Element>;
    using iterator = cycle_iterator_adapter<value_type*>;
    using const_iterator = const cycle_iterator_adapter<value_type*>;


public:
 /**---------------------------------------------------------------------------*
  * @brief Construct new concurrent unordered map with of a specified m_capacity.
  *
  * @note The implementation of this unordered_map uses sentinel values to
  * indicate an entry in the hash table that is empty, i.e., if a hash bucket is
  * empty, the pair residing there will be equal to (unused_key, unused_element).
  * As a result, attempting to insert a key equal to `unused_key` results in
  * undefined behavior.
  *
  * @param _capacity The desired m_capacity of the hash table
  * @param unused_element The sentinel value to use for an empty value
  * @param unused_key The sentinel value to use for an empty key
  * @param hf The hash function to use for hashing keys
  * @param eql The equality comparison function for comparing if two keys are
  * equal
  * @param allocator The allocator to use for allocation the hash table's
  * storage
  *---------------------------------------------------------------------------**/
 explicit concurrent_unordered_map(
     size_type _capacity,
     const mapped_type unused_element = std::numeric_limits<key_type>::max(),
     const key_type unused_key = std::numeric_limits<key_type>::max(),
     const Hasher& hf = hasher(), const Equality& eql = key_equal(),
     const allocator_type& allocator = allocator_type())
     : m_hf(hf),
       m_equal(eql),
       m_allocator(allocator),
       m_capacity(_capacity),
       m_unused_element(unused_element),
       m_unused_key(unused_key) {
   m_hashtbl_values = m_allocator.allocate(m_capacity);
   constexpr int block_size = 128;
   {
     cudaPointerAttributes hashtbl_values_ptr_attributes;
     cudaError_t status = cudaPointerGetAttributes(
         &hashtbl_values_ptr_attributes, m_hashtbl_values);

     if (cudaSuccess == status && isPtrManaged(hashtbl_values_ptr_attributes)) {
       int dev_id = 0;
       CUDA_RT_CALL(cudaGetDevice(&dev_id));
       CUDA_RT_CALL(cudaMemPrefetchAsync(
           m_hashtbl_values, m_capacity * sizeof(value_type), dev_id, 0));
     }
   }

   init_hashtbl<<<((m_capacity - 1) / block_size) + 1, block_size>>>(
       m_hashtbl_values, m_capacity, m_unused_key, m_unused_element);
   CUDA_RT_CALL(cudaGetLastError());
   CUDA_RT_CALL(cudaStreamSynchronize(0));
    }
    
    ~concurrent_unordered_map()
    {
        m_allocator.deallocate( m_hashtbl_values, m_capacity );
    }

    __device__ iterator begin() {
      return iterator(m_hashtbl_values, m_hashtbl_values + m_capacity,
                      m_hashtbl_values);
    }
    __device__ const_iterator begin() const {
      return const_iterator(m_hashtbl_values, m_hashtbl_values + m_capacity,
                            m_hashtbl_values);
    }
    __device__ iterator end() {
      return iterator(m_hashtbl_values, m_hashtbl_values + m_capacity,
                      m_hashtbl_values + m_capacity);
    }
    __device__ const_iterator end() const {
      return const_iterator(m_hashtbl_values, m_hashtbl_values + m_capacity,
                            m_hashtbl_values + m_capacity);
    }
    __device__ value_type* data() const { return m_hashtbl_values; }

    __host__ __device__ key_type get_unused_key() const { return m_unused_key; }

    __host__ __device__ mapped_type get_unused_element() const {
      return m_unused_element;
    }

    __host__ __device__ size_type capacity() const { return m_capacity; }

   private:
    /**---------------------------------------------------------------------------*
     * @brief Enumeration of the possible results of attempting to insert into a 
     * hash bucket
     *---------------------------------------------------------------------------**/
    enum class insert_result {
      CONTINUE,  ///< Insert did not succeed, continue trying to insert (collision)
      SUCCESS,   ///< New pair inserted successfully
      DUPLICATE  ///< Insert did not succeed, key is already present
    };

    /**---------------------------------------------------------------------------*
     * @brief Specialization for value types that can be packed.
     *
     * When the size of the key,value pair being inserted is equal in size to a 
     * type where atomicCAS is natively supported, this optimization path will
     * insert the pair in a single atomicCAS operation.
     *---------------------------------------------------------------------------**/
    template <typename pair_type = value_type>
    __device__ std::enable_if_t<is_packable<pair_type>(), insert_result>
    attempt_insert(value_type* insert_location, value_type const& insert_pair) {
      pair_packer<pair_type> const unused{
          thrust::make_pair(m_unused_key, m_unused_element)};
      pair_packer<pair_type> const new_pair{insert_pair};
      pair_packer<pair_type> const old{atomicCAS(
          reinterpret_cast<typename pair_packer<pair_type>::packed_type*>(
              insert_location),
          unused.packed, new_pair.packed)};

      if (old.packed == unused.packed) {
        return insert_result::SUCCESS;
      }

      if (m_equal(old.pair.first, insert_pair.first)) {
        return insert_result::DUPLICATE;
      }
      return insert_result::CONTINUE;
    }

    /**---------------------------------------------------------------------------*
     * @brief Atempts to insert a key,value pair at the specified hash bucket.
     *
     * @param[in] insert_location Pointer to hash bucket to attempt insert
     * @param[in] insert_pair The pair to insert
     * @return Enum indicating result of insert attempt.
     *---------------------------------------------------------------------------**/
    template <typename pair_type = value_type>
    __device__ std::enable_if_t<not is_packable<pair_type>(), insert_result>
    attempt_insert(value_type* const __restrict__ insert_location,
                   value_type const& insert_pair) {
      key_type const old_key{atomicCAS(&(insert_location->first), m_unused_key,
                                       insert_pair.first)};

      // Hash bucket empty
      if (m_equal(m_unused_key, old_key)) {
        insert_location->second = insert_pair.second;
        return insert_result::SUCCESS;
      }

      // Key already exists
      if (m_equal(old_key, insert_pair.first)) {
        return insert_result::DUPLICATE;
      }

      return insert_result::CONTINUE;
    }

   public:
    /**---------------------------------------------------------------------------*
     * @brief Attempts to insert a key, value pair into the map.
     *
     * Returns an iterator, boolean pair.
     *
     * If the new key already present in the map, the iterator points to
     * the location of the existing key and the boolean is `false` indicating
     * that the insert did not succeed.
     *
     * If the new key was not present, the iterator points to the location where
     * the insert occured and the boolean is `true` indicating that the insert
     * succeeded.
     *
     * @param insert_pair The key and value pair to insert
     * @return Iterator, Boolean pair. Iterator is to the location of the newly
     * inserted pair, or the existing pair that prevented the insert. Boolean
     * indicates insert success.
     *---------------------------------------------------------------------------**/
    __device__ thrust::pair<iterator, bool> insert(
        value_type const& insert_pair) {
      const size_type key_hash{m_hf(insert_pair.first)};
      size_type index{key_hash % m_capacity};

      insert_result status{insert_result::CONTINUE};

      value_type* current_bucket{nullptr};

      while (status == insert_result::CONTINUE) {
        current_bucket = &m_hashtbl_values[index];
        status = attempt_insert(current_bucket, insert_pair);
        index = (index + 1) % m_capacity;
      }

      bool const insert_success =
          (status == insert_result::SUCCESS) ? true : false;

      return thrust::make_pair(
          iterator(m_hashtbl_values, m_hashtbl_values + m_capacity,
                   current_bucket),
          insert_success);
    }

    /**---------------------------------------------------------------------------*
     * @brief Searches the map for the specified key.
     *
     * @note `find` is not threadsafe with `insert`. I.e., it is not safe to
     *do concurrent `insert` and `find` operations.
     *
     * @param k The key to search for
     * @return An iterator to the key if it exists, else map.end()
     *---------------------------------------------------------------------------**/
    __device__ const_iterator find(key_type const& k) const {
      size_type const key_hash = m_hf(k);
      size_type index = key_hash % m_capacity;

      value_type* current_bucket = &m_hashtbl_values[index];

      while (true) {
        key_type const existing_key = current_bucket->first;

        if (m_equal(k, existing_key)) {
          return const_iterator(m_hashtbl_values, m_hashtbl_values + m_capacity,
                                current_bucket);
        }
        if (m_equal(m_unused_key, existing_key)) {
          return this->end();
        }
        index = (index + 1) % m_capacity;
        current_bucket = &m_hashtbl_values[index];
      }
    }

    gdf_error assign_async( const concurrent_unordered_map& other, cudaStream_t stream = 0 )
    {
        if ( other.m_capacity <= m_capacity ) {
            m_capacity = other.m_capacity;
        } else {
            m_allocator.deallocate( m_hashtbl_values, m_capacity );
            m_capacity = other.m_capacity;
            m_capacity = other.m_capacity;
            
            m_hashtbl_values = m_allocator.allocate( m_capacity );
        }
        CUDA_TRY( cudaMemcpyAsync( m_hashtbl_values, other.m_hashtbl_values, m_capacity*sizeof(value_type), cudaMemcpyDefault, stream ) );
        return GDF_SUCCESS;
    }
    
    void clear_async( cudaStream_t stream = 0 ) 
    {
        constexpr int block_size = 128;
        init_hashtbl<<<((m_capacity-1)/block_size)+1,block_size,0,stream>>>( m_hashtbl_values, m_capacity, m_unused_key, m_unused_element );
    }
    
    void print()
    {
        for (size_type i = 0; i < m_capacity; ++i) 
        {
            std::cout<<i<<": "<<m_hashtbl_values[i].first<<","<<m_hashtbl_values[i].second<<std::endl;
        }
    }
    
    gdf_error prefetch( const int dev_id, cudaStream_t stream = 0 )
    {
        cudaPointerAttributes hashtbl_values_ptr_attributes;
        cudaError_t status = cudaPointerGetAttributes( &hashtbl_values_ptr_attributes, m_hashtbl_values );
        
        if ( cudaSuccess == status && isPtrManaged(hashtbl_values_ptr_attributes)) {
            CUDA_TRY( cudaMemPrefetchAsync(m_hashtbl_values, m_capacity*sizeof(value_type), dev_id, stream) );
        }
        CUDA_TRY( cudaMemPrefetchAsync(this, sizeof(*this), dev_id, stream) );

        return GDF_SUCCESS;
    }

   private:
    hasher const m_hf;
    key_equal const m_equal;

    mapped_type const m_unused_element;
    key_type const m_unused_key;

    allocator_type m_allocator;

    size_type m_capacity;
    value_type* m_hashtbl_values;
};

#endif //CONCURRENT_UNORDERED_MAP_CUH
