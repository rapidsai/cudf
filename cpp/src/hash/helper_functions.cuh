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

// TODO: replace this with CUDA_TRY and propagate the error
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                       \
{                                                                                                  \
    cudaError_t cudaStatus = call;                                                                 \
    if ( cudaSuccess != cudaStatus ) {                                                             \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);    \
        exit(1);                                                                                   \
    }                                                                                              \
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

__inline__ __device__ uint64_t atomicCAS(uint64_t* address, uint64_t compare, uint64_t val)
{
  return (uint64_t)atomicCAS((unsigned long long*)address, (unsigned long long)compare, (unsigned long long)val);
}

__inline__ __device__ long long int atomicCAS(long long int* address, long long int compare, long long int val)
{
  return (long long int)atomicCAS((unsigned long long*)address, (unsigned long long)compare, (unsigned long long)val);
}

__inline__ __device__ double atomicCAS(double* address, double compare, double val)
{
  return __longlong_as_double(atomicCAS((unsigned long long int*)address, __double_as_longlong(compare), __double_as_longlong(val)));
}

__inline__ __device__ float atomicCAS(float* address, float compare, float val)
{
  return __int_as_float(atomicCAS((int*)address, __float_as_int(compare), __float_as_int(val)));
}

__inline__ __device__ int64_t atomicAdd(int64_t* address, int64_t val)
{
  return (int64_t) atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

__inline__ __device__ uint64_t atomicAdd(uint64_t* address, uint64_t val)
{
  return (uint64_t) atomicAdd((unsigned long long*)address, (unsigned long long)val);
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

#endif HELPER_FUNCTIONS_CUH
