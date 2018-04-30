/* Copyright 2017-2018 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#ifndef CONCURRENT_UNORDERED_MULTIMAP_CUH
#define CONCURRENT_UNORDERED_MULTIMAP_CUH

#include <iterator>
#include <type_traits>

#include <thrust/pair.h>

#include "managed_allocator.cuh"
#include "managed.cuh"
#include "hash_functions.cuh"

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
        hashtbl_values[idx] = thrust::make_pair( key_val, elem_val );
    }
}

template <typename T>
struct equal_to
{
    typedef bool result_type;
    typedef T first_argument_type;
    typedef T second_argument_type;
    __forceinline__
    __host__ __device__ constexpr bool operator()(const first_argument_type &lhs, const second_argument_type &rhs) const 
    {
        return lhs == rhs;
    }
};

template<typename Iterator>
class cycle_iterator_adapter {
public:
    typedef typename std::iterator_traits<Iterator>::value_type         value_type; 
    typedef typename std::iterator_traits<Iterator>::difference_type    difference_type;
    typedef typename std::iterator_traits<Iterator>::pointer            pointer;
    typedef typename std::iterator_traits<Iterator>::reference          reference;
    typedef Iterator                                                    iterator_type;
    
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
          Key unused_key,
          Element unused_element,
          typename Hasher = default_hash<Key>,
          typename Equality = equal_to<Key>,
          typename Allocator = managed_allocator<thrust::pair<Key, Element> >,
          bool count_collisions = false>
class concurrent_unordered_multimap : public managed
{

public:
    typedef size_t                                          size_type;
    typedef Hasher                                          hasher;
    typedef Equality                                        key_equal;
    typedef Allocator                                       allocator_type;
    typedef Key                                             key_type;
    typedef thrust::pair<Key, Element>                      value_type;
    typedef Element                                         mapped_type;
    typedef cycle_iterator_adapter<value_type*>             iterator;
    typedef const cycle_iterator_adapter<value_type*>       const_iterator;

private:
    union pair2longlong
    {
        unsigned long long int  longlong;
        value_type              pair;
    };
public:

    explicit concurrent_unordered_multimap(size_type n,
                                           const Hasher& hf = hasher(),
                                           const Equality& eql = key_equal(),
                                           const allocator_type& a = allocator_type())
        : m_hf(hf), m_equal(eql), m_allocator(a) , m_hashtbl_size(n), m_optimized(false), m_collisions(0)
    {
        m_hashtbl_values = m_allocator.allocate( m_hashtbl_size );
        constexpr int block_size = 128;
        init_hashtbl<<<((m_hashtbl_size-1)/block_size)+1,block_size>>>( m_hashtbl_values, m_hashtbl_size, unused_key, unused_element );
        CUDA_RT_CALL( cudaGetLastError() );
        CUDA_RT_CALL( cudaStreamSynchronize(0) );
    }
    
    ~concurrent_unordered_multimap()
    {
        m_allocator.deallocate( m_hashtbl_values, m_hashtbl_size );
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
    __host__ __device__ key_type get_unused_key() const
    {
        return unused_key;
    }
    
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
    
    __forceinline__
    __host__ __device__ const_iterator find(const key_type& k) const
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
    
    void prefetch( const int dev_id )
    {
        CUDA_RT_CALL( cudaMemPrefetchAsync(this, sizeof(*this), dev_id, 0) );
        CUDA_RT_CALL( cudaMemPrefetchAsync(m_hashtbl_values, m_hashtbl_size*sizeof(value_type), dev_id, 0) );
    }
    
private:
    const hasher            m_hf;
    const key_equal         m_equal;
    
    allocator_type  m_allocator;
    
    const size_type m_hashtbl_size;
    value_type*     m_hashtbl_values;
    
    bool            m_optimized;
    
    unsigned long long m_collisions;
};

#endif //CONCURRENT_UNORDERED_MULTIMAP_CUH
