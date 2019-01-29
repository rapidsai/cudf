/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION.
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

#ifndef CONCURRENT_UNORDERED_MULTIMAP_CUH
#define CONCURRENT_UNORDERED_MULTIMAP_CUH

#include <iostream>
#include <iterator>
#include <type_traits>
#include <cassert>
#include <cudf.h>

#include <thrust/pair.h>

#include "managed_allocator.cuh"
#include "managed.cuh"
#include "hash_functions.cuh"

// TODO: replace this with CUDA_TRY and propagate the error
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call ) 									   \
{                                                                                                  \
    cudaError_t cudaStatus = call;                                                                 \
    if ( cudaSuccess != cudaStatus ) {                                                             \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);    \
        exit(1);										   \
    }												   \
}
#endif

// TODO: can we do this more efficiently?
__inline__ __device__ int8_t atomicCAS(int8_t* address, int8_t compare, int8_t val)
{
  int32_t *base_address = (int32_t*)((char*)address - ((size_t)address & 3));
  int32_t int_val = (int32_t)val << (((size_t)address & 3) * 8);
  int32_t int_comp = (int32_t)compare << (((size_t)address & 3) * 8);
  return (int8_t)atomicCAS(base_address, int_comp, int_val);
}

// TODO: can we do this more efficiently?
__inline__ __device__ int16_t atomicCAS(int16_t* address, int16_t compare, int16_t val)
{
  int32_t *base_address = (int32_t*)((char*)address - ((size_t)address & 2));
  int32_t int_val = (int32_t)val << (((size_t)address & 2) * 8);
  int32_t int_comp = (int32_t)compare << (((size_t)address & 2) * 8);
  return (int16_t)atomicCAS(base_address, int_comp, int_val);
}

__inline__ __device__ int64_t atomicCAS(int64_t* address, int64_t compare, int64_t val)
{
  return (int64_t)atomicCAS((unsigned long long*)address, (unsigned long long)compare, (unsigned long long)val);
}

__inline__ __device__ int64_t atomicAdd(int64_t* address, int64_t val)
{
  return (int64_t)atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

template<typename pair_type>
__forceinline__
__device__ pair_type load_pair_vectorized( const pair_type* __restrict__ const ptr )
{
    if ( sizeof(uint4) == sizeof(pair_type) ) {
        union pair_type2vec_type
        {
            uint4       vec_val;
            pair_type   pair_val;
        };
        pair_type2vec_type converter = {0,0,0,0};
        converter.vec_val = *reinterpret_cast<const uint4*>(ptr);
        return converter.pair_val;
    } else if ( sizeof(uint2) == sizeof(pair_type) ) {
        union pair_type2vec_type
        {
            uint2       vec_val;
            pair_type   pair_val;
        };
        pair_type2vec_type converter = {0,0};
        converter.vec_val = *reinterpret_cast<const uint2*>(ptr);
        return converter.pair_val;
    } else if ( sizeof(int) == sizeof(pair_type) ) {
        union pair_type2vec_type
        {
            int         vec_val;
            pair_type   pair_val;
        };
        pair_type2vec_type converter = {0};
        converter.vec_val = *reinterpret_cast<const int*>(ptr);
        return converter.pair_val;
    } else if ( sizeof(short) == sizeof(pair_type) ) {
        union pair_type2vec_type
        {
            short       vec_val;
            pair_type   pair_val;
        };
        pair_type2vec_type converter = {0};
        converter.vec_val = *reinterpret_cast<const short*>(ptr);
        return converter.pair_val;
    } else {
        return *ptr;
    }
}

template<typename pair_type>
__forceinline__
__device__ void store_pair_vectorized( pair_type* __restrict__ const ptr, const pair_type val )
{
    if ( sizeof(uint4) == sizeof(pair_type) ) {
        union pair_type2vec_type
        {
            uint4       vec_val;
            pair_type   pair_val;
        };
        pair_type2vec_type converter = {0,0,0,0};
        converter.pair_val = val;
        *reinterpret_cast<uint4*>(ptr) = converter.vec_val;
    } else if ( sizeof(uint2) == sizeof(pair_type) ) {
        union pair_type2vec_type
        {
            uint2       vec_val;
            pair_type   pair_val;
        };
        pair_type2vec_type converter = {0,0};
        converter.pair_val = val;
        *reinterpret_cast<uint2*>(ptr) = converter.vec_val;
    } else if ( sizeof(int) == sizeof(pair_type) ) {
        union pair_type2vec_type
        {
            int         vec_val;
            pair_type   pair_val;
        };
        pair_type2vec_type converter = {0};
        converter.pair_val = val;
        *reinterpret_cast<int*>(ptr) = converter.vec_val;
    } else if ( sizeof(short) == sizeof(pair_type) ) {
        union pair_type2vec_type
        {
            short       vec_val;
            pair_type   pair_val;
        };
        pair_type2vec_type converter = {0};
        converter.pair_val = val;
        *reinterpret_cast<short*>(ptr) = converter.vec_val;
    } else {
        *ptr = val;
    }
}

template<typename value_type, typename size_type, typename key_type, typename elem_type>
__global__ void init_hashtbl(
    value_type* __restrict__ const hashtbl_values,
    const size_type n,
    const key_type key_val,
    const elem_type elem_val)
{
    const size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < n )
    {
        store_pair_vectorized( hashtbl_values + idx, thrust::make_pair( key_val, elem_val ) );
    }
}

template <typename T>
struct equal_to
{
    using result_type = bool;
    using first_argument_type = T;
    using second_argument_type = T;
    __forceinline__
    __host__ __device__ constexpr bool operator()(const first_argument_type &lhs, const second_argument_type &rhs) const 
    {
        return lhs == rhs;
    }
};

template<typename Iterator>
class cycle_iterator_adapter {
public:
    using value_type = typename std::iterator_traits<Iterator>::value_type; 
    using difference_type = typename std::iterator_traits<Iterator>::difference_type;
    using pointer = typename std::iterator_traits<Iterator>::pointer;
    using reference = typename std::iterator_traits<Iterator>::reference;
    using iterator_type = Iterator;
    
    cycle_iterator_adapter() = delete;
    
    __host__ __device__ explicit cycle_iterator_adapter( const iterator_type& begin, const iterator_type& end, const iterator_type& current )
        : m_begin( begin ), m_end( end ), m_current( current )
    {}
    
    __host__ __device__ cycle_iterator_adapter& operator++()
    {
        if ( m_end == (m_current+1) )
            m_current = m_begin;
        else
            ++m_current;
        return *this;
    }
    
    __host__ __device__ const cycle_iterator_adapter& operator++() const
    {
        if ( m_end == (m_current+1) )
            m_current = m_begin;
        else
            ++m_current;
        return *this;
    }
    
    __host__ __device__ cycle_iterator_adapter& operator++(int)
    {
        cycle_iterator_adapter<iterator_type> old( m_begin, m_end, m_current);
        if ( m_end == (m_current+1) )
            m_current = m_begin;
        else
            ++m_current;
        return old;
    }
    
    __host__ __device__ const cycle_iterator_adapter& operator++(int) const
    {
        cycle_iterator_adapter<iterator_type> old( m_begin, m_end, m_current);
        if ( m_end == (m_current+1) )
            m_current = m_begin;
        else
            ++m_current;
        return old;
    }
    
    __host__ __device__ bool equal(const cycle_iterator_adapter<iterator_type>& other) const
    {
        return m_current == other.m_current && m_begin == other.m_begin && m_end == other.m_end;
    }
    
    __host__ __device__ reference& operator*()
    {
        return *m_current;
    }
    
    __host__ __device__ const reference& operator*() const
    {
        return *m_current;
    }

    __host__ __device__ const pointer operator->() const
    {
        return m_current.operator->();
    }
    
    __host__ __device__ pointer operator->()
    {
        return m_current;
    }
    
private:
    iterator_type m_current;
    iterator_type m_begin;
    iterator_type m_end;
};

template <class T>
__host__ __device__ bool operator==(const cycle_iterator_adapter<T>& lhs, const cycle_iterator_adapter<T>& rhs)
{
    return lhs.equal(rhs);
}

template <class T>
__host__ __device__ bool operator!=(const cycle_iterator_adapter<T>& lhs, const cycle_iterator_adapter<T>& rhs)
{
    return !lhs.equal(rhs);
}

/**
 * Does support concurrent insert, but not concurrent insert and probping.
 *
 * TODO:
 *  - add constructor that takes pointer to hash_table to avoid allocations
 *  - extend interface to accept streams
 */
template <typename Key,
          typename Element,
          typename size_type,
          Key unused_key,
          Element unused_element,
          typename Hasher = default_hash<Key>,
          typename Equality = equal_to<Key>,
          typename Allocator = managed_allocator<thrust::pair<Key, Element> >,
          bool count_collisions = false>
class concurrent_unordered_multimap : public managed
{

public:
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

    /* --------------------------------------------------------------------------*/
    /**
     * @Synopsis Allocates memory and optionally fills the hash map with unused keys/values
     *
     * @Param[in] n The size of the hash table (the number of key-value pairs)
     * @Param[in] init Initialize the hash table with the unused keys/values
     * @Param[in] hf An optional hashing function
     * @Param[in] eql An optional functor for comparing if two keys are equal
     * @Param[in] a An optional functor for allocating the hash table memory
     */
    /* ----------------------------------------------------------------------------*/
    explicit concurrent_unordered_multimap(size_type n,
                                           const bool init = true,
                                           const Hasher& hf = hasher(),
                                           const Equality& eql = key_equal(),
                                           const allocator_type& a = allocator_type())
        : m_hf(hf), m_equal(eql), m_allocator(a), m_hashtbl_size(n), m_hashtbl_capacity(n), m_collisions(0)
    {
        m_hashtbl_values = m_allocator.allocate( m_hashtbl_capacity );
        constexpr int block_size = 128;
        {
            cudaPointerAttributes hashtbl_values_ptr_attributes;
            cudaError_t status = cudaPointerGetAttributes( &hashtbl_values_ptr_attributes, m_hashtbl_values );
            
            if ( cudaSuccess == status && hashtbl_values_ptr_attributes.isManaged ) {
                int dev_id = 0;
                CUDA_RT_CALL( cudaGetDevice( &dev_id ) );
                CUDA_RT_CALL( cudaMemPrefetchAsync(m_hashtbl_values, m_hashtbl_size*sizeof(value_type), dev_id, 0) );
            }
        }

        if( init )
        {
            init_hashtbl<<<((m_hashtbl_size-1)/block_size)+1,block_size>>>( m_hashtbl_values, m_hashtbl_size, unused_key, unused_element );
            CUDA_RT_CALL( cudaGetLastError() );
            CUDA_RT_CALL( cudaStreamSynchronize(0) );
        }
    }
    
    ~concurrent_unordered_multimap()
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
    
    __forceinline__
    static constexpr __host__ __device__ key_type get_unused_key()
    {
        return unused_key;
    }
   
    /* --------------------------------------------------------------------------*/
    /**
     * @Synopsis Computes a hash value for a key
     *
     * @Param[in] the_key The key to compute a hash for
     * @tparam hash_value_type The datatype of the hash value
     *
     * @Returns   The hash value for the key
     */
    /* ----------------------------------------------------------------------------*/
    template <typename hash_value_type = typename Hasher::result_type>
    __forceinline__
    __host__ __device__ hash_value_type get_hash(const key_type& the_key) const
    {
        return m_hf(the_key);
    }

    /* --------------------------------------------------------------------------*/
    /**
     * @Synopsis Computes the destination hash map partition for a key
     *
     * @Param[in] the_key The key to search for
     * @Param[in] num_parts The total number of partitions in the partitioned
     * hash table
     * @Param[in] precomputed_hash A flag indicating whether or not a precomputed
     * hash value is passed in
     * @Param[in] precomputed_hash_value A precomputed hash value to use for determing
     * the write location of the key into the hash map instead of computing the
     * the hash value directly from the key
     * @tparam hash_value_type The datatype of the hash value
     *
     * @Returns   The destination hash table partition for the specified key
     */
    /* ----------------------------------------------------------------------------*/
    template <typename hash_value_type = typename Hasher::result_type>
    __forceinline__
    __host__ __device__ int get_partition(const key_type& the_key,
                                          const int num_parts = 1,
                                          bool precomputed_hash = false,
                                          hash_value_type precomputed_hash_value = 0) const
    {
        hash_value_type hash_value{0};

        // If a precomputed hash value has been passed in, then use it to determine
        // the location of the key
        if(true == precomputed_hash) {
          hash_value = precomputed_hash_value;
        }
        // Otherwise, compute the hash value from the key
        else {
          hash_value = m_hf(the_key);
        }

        size_type hash_tbl_idx = hash_value % m_hashtbl_size;

        const size_type partition_size  = m_hashtbl_size/num_parts;

        int dest_part = hash_tbl_idx/partition_size;
        // Note that if m_hashtbl_size % num_parts != 0 then dest_part can be
        // num_parts for the last few elements and we remap that to the
        // num_parts-1 partition
        if (dest_part == num_parts) dest_part = num_parts-1;

        return dest_part;
    }

    /* --------------------------------------------------------------------------*/
    /** 
     * @Synopsis  Inserts a (key, value) pair into the hash map
     * 
     * @Param[in] x The (key, value) pair to insert
     * @Param[in] precomputed_hash A flag indicating whether or not a precomputed 
     * hash value is passed in
     * @Param[in] precomputed_hash_value A precomputed hash value to use for determing
     * the write location of the key into the hash map instead of computing the
     * the hash value directly from the key
     * @Param[in] keys_are_equal An optional functor for comparing if two keys are equal
     * @tparam hash_value_type The datatype of the hash value
     * @tparam comparison_type The type of the key comparison functor
     * 
     * @Returns An iterator to the newly inserted (key, value) pair
     */
    /* ----------------------------------------------------------------------------*/
    template < typename hash_value_type = typename Hasher::result_type,
               typename comparison_type = key_equal>
    __forceinline__
    __device__ iterator insert(const value_type& x,
                               bool precomputed_hash = false,
                               hash_value_type precomputed_hash_value = 0,
                               comparison_type keys_are_equal = key_equal())
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

        size_type hash_tbl_idx = hash_value % hashtbl_size;

        value_type* it = 0;

        size_type attempt_counter{0};
        
        while (0 == it) {
            value_type* tmp_it = hashtbl_values + hash_tbl_idx;

            if ( std::numeric_limits<key_type>::is_integer && std::numeric_limits<mapped_type>::is_integer &&
                 sizeof(unsigned long long int) == sizeof(value_type) )
            {
                pair2longlong converter = {0ull};
                converter.pair = thrust::make_pair( unused_key, unused_element );
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
            } 
            else 
            {
                const key_type old_key = atomicCAS( &(tmp_it->first), unused_key, x.first );

                if ( keys_are_equal( unused_key, old_key ) ) 
                {
                    (m_hashtbl_values+hash_tbl_idx)->second = x.second;
                    it = tmp_it;
                }
                else if ( count_collisions )
                {
                    atomicAdd( &m_collisions, 1 );
                }
            }

            hash_tbl_idx = (hash_tbl_idx+1)%hashtbl_size;

            attempt_counter++;
            if( attempt_counter > hashtbl_size)
            {
              printf("Attempted to insert to multimap but the map is full!\n");
              return this->end();
            }
        }
        
        return iterator( m_hashtbl_values,m_hashtbl_values+hashtbl_size,it);
    }

    /* --------------------------------------------------------------------------*/
    /**
     * @Synopsis  Inserts a (key, value) pair into the hash map partition. This
     * is useful when building the hash table in multiple passes, one
     * contiguous partition at a time, or when building the hash table
     * distributed between multiple devices.
     *
     * @Param[in] x The (key, value) pair to insert
     * @Param[in] part The partition number for the partitioned hash table build
     * @Param[in] num_parts The total number of partitions in the partitioned
     * hash table
     * @Param[in] precomputed_hash A flag indicating whether or not a precomputed
     * hash value is passed in
     * @Param[in] precomputed_hash_value A precomputed hash value to use for determing
     * the write location of the key into the hash map instead of computing the
     * the hash value directly from the key
     * @Param[in] keys_are_equal An optional functor for comparing if two keys are equal
     * @tparam hash_value_type The datatype of the hash value
     * @tparam comparison_type The type of the key comparison functor
     *
     * @Returns An iterator to the newly inserted (key, value) pair
     */
    /* ----------------------------------------------------------------------------*/
    template < typename hash_value_type = typename Hasher::result_type,
               typename comparison_type = key_equal>
    __forceinline__
    __device__ iterator insert_part(const value_type& x,
                                    const int part = 0,
                                    const int num_parts = 1,
                                    bool precomputed_hash = false,
                                    hash_value_type precomputed_hash_value = 0,
                                    comparison_type keys_are_equal = key_equal())
    {
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

	// Find the destination partition index 
	int dest_part = get_partition(x.first, num_parts, true, hash_value);

        // Only insert if the key belongs to the specified partition
        if ( dest_part != part )
          return end();
        else
          return insert(x, true, hash_value, keys_are_equal);
    }
    
    /* --------------------------------------------------------------------------*/
    /** 
     * @Synopsis Searches for a key in the hash map and returns an iterator to the first
     * instance of the key in the map.
     * 
     * @Param[in] the_key The key to search for
     * @Param[in] precomputed_hash A flag indicating whether or not a precomputed 
     * hash value is passed in
     * @Param[in] precomputed_hash_value A precomputed hash value to use for determing
     * the write location of the key into the hash map instead of computing the
     * the hash value directly from the key
     * @Param[in] keys_are_equal An optional functor for comparing if two keys are equal
     * @tparam hash_value_type The datatype of the hash value
     * @tparam comparison_type The type of the key comparison functor
     * 
     * @Returns   An iterator to the first instance of the key in the map
     */
    /* ----------------------------------------------------------------------------*/
    template < typename hash_value_type = typename Hasher::result_type,
               typename comparison_type = key_equal>
    __forceinline__
    __host__ __device__ const_iterator find(const key_type& the_key,
                                            bool precomputed_hash = false,
                                            hash_value_type precomputed_hash_value = 0,
                                            comparison_type keys_are_equal = key_equal()) const
    {
        hash_value_type hash_value{0};

        // If a precomputed hash value has been passed in, then use it to determine
        // the location of the key 
        if(true == precomputed_hash) {
          hash_value = precomputed_hash_value;
        }
        // Otherwise, compute the hash value from the key
        else {
          hash_value = m_hf(the_key);
        }

        size_type hash_tbl_idx = hash_value % m_hashtbl_size;

        value_type* begin_ptr = 0;
        
        size_type counter = 0;
        while ( 0 == begin_ptr ) 
        {
            value_type* tmp_ptr = m_hashtbl_values + hash_tbl_idx;
            const key_type tmp_val = tmp_ptr->first;
            if ( keys_are_equal( the_key, tmp_val ) ) {
                begin_ptr = tmp_ptr;
                break;
            }
            if ( keys_are_equal( unused_key , tmp_val ) || (counter > m_hashtbl_size) ) {
                begin_ptr = m_hashtbl_values + m_hashtbl_size;
                break;
            }
            hash_tbl_idx = (hash_tbl_idx+1)%m_hashtbl_size;
            ++counter;
        }
        
        return const_iterator( m_hashtbl_values,m_hashtbl_values+m_hashtbl_size,begin_ptr);
    }

    gdf_error assign_async( const concurrent_unordered_multimap& other, cudaStream_t stream = 0 )
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
        init_hashtbl<<<((m_hashtbl_size-1)/block_size)+1,block_size,0,stream>>>( m_hashtbl_values, m_hashtbl_size, unused_key, unused_element );
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
        
        if ( cudaSuccess == status && hashtbl_values_ptr_attributes.isManaged ) {
            CUDA_TRY( cudaMemPrefetchAsync(m_hashtbl_values, m_hashtbl_size*sizeof(value_type), dev_id, stream) );
        }
        CUDA_TRY( cudaMemPrefetchAsync(this, sizeof(*this), dev_id, stream) );
        return GDF_SUCCESS;
    }
    
private:
    const hasher            m_hf;
    const key_equal         m_equal;
    
    allocator_type              m_allocator;
    
    size_type   m_hashtbl_size;
    size_type   m_hashtbl_capacity;
    value_type* m_hashtbl_values;
    
    unsigned long long m_collisions;
};

#endif //CONCURRENT_UNORDERED_MULTIMAP_CUH
