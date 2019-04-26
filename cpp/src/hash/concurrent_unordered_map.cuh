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

#include <iterator>
#include <type_traits>
#include <cassert>
#include <iostream>
#include <thrust/pair.h>

#include "cudf.h"
#include "groupby/aggregation_operations.hpp"

#include "managed_allocator.cuh"
#include "managed.cuh"
#include "hash_functions.cuh"

#include "helper_functions.cuh"

#include "utilities/device_atomics.cuh"

/**
 * Does support concurrent insert, but not concurrent insert and probping.
 *
 * TODO:
 *  - add constructor that takes pointer to hash_table to avoid allocations
 *  - extend interface to accept streams
 */
template <typename Key,
          typename Element,
          Key unused_key,
          typename Hasher = default_hash<Key>,
          typename Equality = equal_to<Key>,
          typename Allocator = managed_allocator<thrust::pair<Key, Element> >,
          bool count_collisions = false>
class concurrent_unordered_map : public managed
{

public:
    using size_type = size_t;
    using hasher = Hasher;
    using key_equal = Equality;
    using allocator_type = Allocator;
    using key_type = Key;
    using value_type = thrust::pair<Key, Element>;
    using mapped_type = Element;
    using iterator = cycle_iterator_adapter<value_type*>;
    using const_iterator = const cycle_iterator_adapter<value_type*>;

private:
    union pair2longlong
    {
        unsigned long long int  longlong;
        value_type              pair;
    };
    
public:

    explicit concurrent_unordered_map(size_type n,
                                      const mapped_type unused_element,
                                      const Hasher& hf = hasher(),
                                      const Equality& eql = key_equal(),
                                      const allocator_type& a = allocator_type())
        : m_hf(hf), m_equal(eql), m_allocator(a), m_hashtbl_size(n), m_hashtbl_capacity(n), m_collisions(0), m_unused_element(unused_element)
    {
        m_hashtbl_values = m_allocator.allocate( m_hashtbl_capacity );
        constexpr int block_size = 128;
        {
            cudaPointerAttributes hashtbl_values_ptr_attributes;
            cudaError_t status = cudaPointerGetAttributes( &hashtbl_values_ptr_attributes, m_hashtbl_values );
            
            if ( cudaSuccess == status && isPtrManaged(hashtbl_values_ptr_attributes)) {
                int dev_id = 0;
                CUDA_RT_CALL( cudaGetDevice( &dev_id ) );
                CUDA_RT_CALL( cudaMemPrefetchAsync(m_hashtbl_values, m_hashtbl_size*sizeof(value_type), dev_id, 0) );
            }
        }
        
        init_hashtbl<<<((m_hashtbl_size-1)/block_size)+1,block_size>>>( m_hashtbl_values, m_hashtbl_size, unused_key, m_unused_element );
        CUDA_RT_CALL( cudaGetLastError() );
        CUDA_RT_CALL( cudaStreamSynchronize(0) );
    }
    
    ~concurrent_unordered_map()
    {
        m_allocator.deallocate( m_hashtbl_values, m_hashtbl_capacity );
    }
    
    __host__ __device__ iterator begin()
    {
        return iterator( m_hashtbl_values,m_hashtbl_values+m_hashtbl_size,m_hashtbl_values );
    }
    __host__ __device__ const_iterator begin() const
    {
        return const_iterator( m_hashtbl_values,m_hashtbl_values+m_hashtbl_size,m_hashtbl_values );
    }
    __host__ __device__ iterator end()
    {
        return iterator( m_hashtbl_values,m_hashtbl_values+m_hashtbl_size,m_hashtbl_values+m_hashtbl_size );
    }
    __host__ __device__ const_iterator end() const
    {
        return const_iterator( m_hashtbl_values,m_hashtbl_values+m_hashtbl_size,m_hashtbl_values+m_hashtbl_size );
    }
    __host__ __device__ size_type size() const
    {
        return m_hashtbl_size;
    }
    __host__ __device__ value_type* data() const
    {
      return m_hashtbl_values;
    }
    
    __forceinline__
    static constexpr __host__ __device__ key_type get_unused_key()
    {
        return unused_key;
    }

    // Generic update of a hash table value for any aggregator
    template <typename aggregation_type>
    __forceinline__  __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, aggregation_type op)
    {
      const mapped_type insert_value = insert_pair.second;

      mapped_type old_value = existing_value;

      mapped_type expected{old_value};

      // Attempt to perform the aggregation with existing_value and
      // store the result atomically
      do 
      {
        expected = old_value;

        const mapped_type new_value = op(insert_value, old_value);

        old_value = atomicCAS(&existing_value, expected, new_value);
      }
      // Guard against another thread's update to existing_value
      while( expected != old_value );
    }

    // TODO Overload atomicAdd for 1 byte and 2 byte types, until then, overload specifically for the types
    // where atomicAdd already has an overload. Otherwise the generic update_existing_value will be used.
    // Specialization for COUNT aggregator
    __forceinline__ __host__ __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, count_op<int32_t> op)
    {
      atomicAdd(&existing_value, static_cast<mapped_type>(1));
    }
    // Specialization for COUNT aggregator
    __forceinline__ __host__ __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, count_op<int64_t> op)
    {
      atomicAdd(&existing_value, static_cast<mapped_type>(1));
    }
    // Specialization for COUNT aggregator
    __forceinline__ __host__ __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, count_op<float> op)
    {
      atomicAdd(&existing_value, static_cast<mapped_type>(1));
    }
    // Specialization for COUNT aggregator
    __forceinline__ __host__ __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, count_op<double> op)
    {
      atomicAdd(&existing_value, static_cast<mapped_type>(1));
    }

    // Specialization for SUM aggregator (int32)
    __forceinline__  __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, sum_op<int32_t> op)
    {
      atomicAdd(&existing_value, insert_pair.second);
    }

    // Specialization for SUM aggregator (int64)
    __forceinline__  __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, sum_op<int64_t> op)
    {
      atomicAdd(&existing_value, insert_pair.second);
    }

    // Specialization for SUM aggregator (fp32)
    __forceinline__  __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, sum_op<float> op)
    {
      atomicAdd(&existing_value, insert_pair.second);
    }

    // Specialization for SUM aggregator (fp64)
    __forceinline__  __device__
    void update_existing_value(mapped_type & existing_value, value_type const & insert_pair, sum_op<double> op)
    {
      atomicAdd(&existing_value, insert_pair.second);
    }

    /* --------------------------------------------------------------------------*/
    /** 
     * @brief  Inserts a new (key, value) pair. If the key already exists in the map
                  an aggregation operation is performed with the new value and existing value.
                  E.g., if the aggregation operation is 'max', then the maximum is computed
                  between the new value and existing value and the result is stored in the map.
     * 
     * @param[in] x The new (key, value) pair to insert
     * @param[in] op The aggregation operation to perform
     * @param[in] keys_equal An optional functor for comparing two keys 
     * @param[in] precomputed_hash Indicates if a precomputed hash value is being passed in to use
     * to determine the write location of the new key
     * @param[in] precomputed_hash_value The precomputed hash value
     * @tparam aggregation_type A functor for a binary operation that performs the aggregation
     * @tparam comparison_type A functor for comparing two keys
     * 
     * @returns An iterator to the newly inserted key,value pair
     */
    /* ----------------------------------------------------------------------------*/
    template<typename aggregation_type,
             class comparison_type = key_equal,
             typename hash_value_type = typename Hasher::result_type>
    __forceinline__
    __device__ iterator insert(const value_type& x, 
                               aggregation_type op,
                               comparison_type keys_equal = key_equal(),
                               bool precomputed_hash = false,
                               hash_value_type precomputed_hash_value = 0)
    {
        const size_type hashtbl_size    = m_hashtbl_size;
        value_type* hashtbl_values      = m_hashtbl_values;

        hash_value_type hash_value{0};

        // If a precomputed hash value has been passed in, then use it to determine
        // the write location of the new key
        if(true == precomputed_hash)
        {
          hash_value = precomputed_hash_value;
        }
        // Otherwise, compute the hash value from the new key
        else
        {
          hash_value = m_hf(x.first);
        }

        size_type current_index         = hash_value % hashtbl_size;
        value_type *current_hash_bucket = &(hashtbl_values[current_index]);

        const key_type insert_key = x.first;
        
        bool insert_success = false;
        
        while (false == insert_success) {

          key_type& existing_key = current_hash_bucket->first;
          mapped_type& existing_value = current_hash_bucket->second;

          // Try and set the existing_key for the current hash bucket to insert_key
          const key_type old_key = atomicCAS( &existing_key, unused_key, insert_key);

          // If old_key == unused_key, the current hash bucket was empty
          // and existing_key was updated to insert_key by the atomicCAS. 
          // If old_key == insert_key, this key has already been inserted. 
          // In either case, perform the atomic aggregation of existing_value and insert_value
          // Because the hash table is initialized with the identity value of the aggregation
          // operation, it is safe to perform the operation when the existing_value still 
          // has its initial value
          // TODO: Use template specialization to make use of native atomic functions
          // TODO: How to handle data types less than 32 bits?
          if ( keys_equal( unused_key, old_key ) || keys_equal(insert_key, old_key) ) {

            update_existing_value(existing_value, x, op);

            insert_success = true;
          }

          current_index = (current_index+1)%hashtbl_size;
          current_hash_bucket = &(hashtbl_values[current_index]);
        }
        
        return iterator( m_hashtbl_values,m_hashtbl_values+hashtbl_size, current_hash_bucket);
    }
    
    /* This function is not currently implemented
    __forceinline__
    __host__ __device__ iterator insert(const value_type& x)
    {
        const size_type hashtbl_size    = m_hashtbl_size;
        value_type* hashtbl_values      = m_hashtbl_values;
        const size_type key_hash        = m_hf( x.first );
        size_type hash_tbl_idx          = key_hash%hashtbl_size;
        
        value_type* it = 0;
        
        while (0 == it) {
            value_type* tmp_it = hashtbl_values + hash_tbl_idx;
#ifdef __CUDA_ARCH__
            if ( std::numeric_limits<key_type>::is_integer && std::numeric_limits<mapped_type>::is_integer &&
                 sizeof(unsigned long long int) == sizeof(value_type) )
            {
                pair2longlong converter = {0ull};
                converter.pair = thrust::make_pair( unused_key, m_unused_element );
                const unsigned long long int unused = converter.longlong;
                converter.pair = x;
                const unsigned long long int value = converter.longlong;
                const unsigned long long int old_val = atomicCAS( reinterpret_cast<unsigned long long int*>(tmp_it), unused, value );
                if ( old_val == unused ) {
                    it = tmp_it;
                }
                else if ( count_collisions )
                {
                    atomicAdd( &m_collisions, 1 );
                }
            } else {
                const key_type old_key = atomicCAS( &(tmp_it->first), unused_key, x.first );
                if ( m_equal( unused_key, old_key ) ) {
                    (m_hashtbl_values+hash_tbl_idx)->second = x.second;
                    it = tmp_it;
                }
                else if ( count_collisions )
                {
                    atomicAdd( &m_collisions, 1 );
                }
            }
#else
            
            #pragma omp critical
            {
                if ( m_equal( unused_key, tmp_it->first ) ) {
                    hashtbl_values[hash_tbl_idx] = thrust::make_pair( x.first, x.second );
                    it = tmp_it;
                }
            }
#endif
            hash_tbl_idx = (hash_tbl_idx+1)%hashtbl_size;
        }
        
        return iterator( m_hashtbl_values,m_hashtbl_values+hashtbl_size,it);
    }
    */
    
    __forceinline__
    __host__ __device__ const_iterator find(const key_type& k ) const
    {
        size_type key_hash = m_hf( k );
        size_type hash_tbl_idx = key_hash%m_hashtbl_size;
        
        value_type* begin_ptr = 0;
        
        size_type counter = 0;
        while ( 0 == begin_ptr ) {
            value_type* tmp_ptr = m_hashtbl_values + hash_tbl_idx;
            const key_type tmp_val = tmp_ptr->first;
            if ( m_equal( k, tmp_val ) ) {
                begin_ptr = tmp_ptr;
                break;
            }
            if ( m_equal( unused_key , tmp_val ) || counter > m_hashtbl_size ) {
                begin_ptr = m_hashtbl_values + m_hashtbl_size;
                break;
            }
            hash_tbl_idx = (hash_tbl_idx+1)%m_hashtbl_size;
            ++counter;
        }
        
        return const_iterator( m_hashtbl_values,m_hashtbl_values+m_hashtbl_size,begin_ptr);
    }
    
    gdf_error assign_async( const concurrent_unordered_map& other, cudaStream_t stream = 0 )
    {
        m_collisions = other.m_collisions;
        if ( other.m_hashtbl_size <= m_hashtbl_capacity ) {
            m_hashtbl_size = other.m_hashtbl_size;
        } else {
            m_allocator.deallocate( m_hashtbl_values, m_hashtbl_capacity );
            m_hashtbl_capacity = other.m_hashtbl_size;
            m_hashtbl_size = other.m_hashtbl_size;
            
            m_hashtbl_values = m_allocator.allocate( m_hashtbl_capacity );
        }
        CUDA_TRY( cudaMemcpyAsync( m_hashtbl_values, other.m_hashtbl_values, m_hashtbl_size*sizeof(value_type), cudaMemcpyDefault, stream ) );
        return GDF_SUCCESS;
    }
    
    void clear_async( cudaStream_t stream = 0 ) 
    {
        constexpr int block_size = 128;
        init_hashtbl<<<((m_hashtbl_size-1)/block_size)+1,block_size,0,stream>>>( m_hashtbl_values, m_hashtbl_size, unused_key, m_unused_element );
        if ( count_collisions )
            m_collisions = 0;
    }
    
    unsigned long long get_num_collisions() const
    {
        return m_collisions;
    }
    
    void print()
    {
        for (size_type i = 0; i < m_hashtbl_size; ++i) 
        {
            std::cout<<i<<": "<<m_hashtbl_values[i].first<<","<<m_hashtbl_values[i].second<<std::endl;
        }
    }
    
    gdf_error prefetch( const int dev_id, cudaStream_t stream = 0 )
    {
        cudaPointerAttributes hashtbl_values_ptr_attributes;
        cudaError_t status = cudaPointerGetAttributes( &hashtbl_values_ptr_attributes, m_hashtbl_values );
        
        if ( cudaSuccess == status && isPtrManaged(hashtbl_values_ptr_attributes)) {
            CUDA_TRY( cudaMemPrefetchAsync(m_hashtbl_values, m_hashtbl_size*sizeof(value_type), dev_id, stream) );
        }
        CUDA_TRY( cudaMemPrefetchAsync(this, sizeof(*this), dev_id, stream) );

        return GDF_SUCCESS;
    }
    
private:
    const hasher            m_hf;
    const key_equal         m_equal;

    const mapped_type       m_unused_element;
    
    allocator_type              m_allocator;
    
    size_type   m_hashtbl_size;
    size_type   m_hashtbl_capacity;
    value_type* m_hashtbl_values;
    
    unsigned long long m_collisions;
};

#endif //CONCURRENT_UNORDERED_MAP_CUH
