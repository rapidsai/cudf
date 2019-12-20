/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.
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

#ifndef HELPER_FUNCTIONS_CUH
#define HELPER_FUNCTIONS_CUH

constexpr int64_t DEFAULT_HASH_TABLE_OCCUPANCY = 50;

/**---------------------------------------------------------------------------*
 * @brief  Compute requisite size of hash table.
 * 
 * Computes the number of entries required in a hash table to satisfy
 * inserting a specified number of keys to achieve the specified hash table
 * occupancy.
 *
 * @param num_keys_to_insert The number of keys that will be inserted
 * @param desired_occupancy The desired occupancy percentage, e.g., 50 implies a
 * 50% occupancy
 * @return size_t The size of the hash table that will satisfy the desired
 * occupancy for the specified number of insertions
 *---------------------------------------------------------------------------**/
inline size_t compute_hash_table_size(
    cudf::size_type num_keys_to_insert,
    uint32_t desired_occupancy = DEFAULT_HASH_TABLE_OCCUPANCY) {
  assert(desired_occupancy != 0);
  assert(desired_occupancy <= 100);
  double const grow_factor{100.0 / desired_occupancy};

  // Calculate size of hash map based on the desired occupancy
  size_t hash_table_size{
      static_cast<size_t>(std::ceil(num_keys_to_insert * grow_factor))};

  return hash_table_size;
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

#endif // HELPER_FUNCTIONS_CUH
