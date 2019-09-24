/*
* Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include <thrust/transform_scan.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include "nvstrings/NVStrings.h"

#include "./NVStringsImpl.h"
#include "../custring_view.cuh"
#include "../unicode/is_flags.h"
#include "../util.h"

// this will return the number of characters for each string
unsigned int NVStrings::len(int* lengths, bool todevice)
{
    unsigned int count = size();
    if( lengths==0 || count==0 )
        return count;

    auto execpol = rmm::exec_policy(0);
    int* d_rtn = lengths;
    if( !todevice )
        d_rtn = device_alloc<int>(count,0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->chars_count();
            else
                d_rtn[idx] = -1;
        });
    //
    //printCudaError(cudaDeviceSynchronize(),"nvs-len");
    size_t size = thrust::reduce(execpol->on(0), d_rtn, d_rtn+count, (size_t)0,
         []__device__(int lhs, int rhs) {
            if( lhs < 0 )
                lhs = 0;
            if( rhs < 0 )
                rhs = 0;
            return lhs + rhs;
         });

    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(lengths,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (unsigned int)size;
}

// this will return the number of bytes for each string
size_t NVStrings::byte_count(int* lengths, bool todevice)
{
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    int* d_rtn = lengths;
    if( !lengths )
        todevice = false; // makes sure we free correctly
    if( !todevice )
        d_rtn = device_alloc<int>(count,0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->size();
            else
                d_rtn[idx] = -1;
        });
    //
    //printCudaError(cudaDeviceSynchronize(),"nvs-bytes");
    size_t size = thrust::reduce(execpol->on(0), d_rtn, d_rtn+count, (size_t)0,
         []__device__(int lhs, int rhs) {
            if( lhs < 0 )
                lhs = 0;
            if( rhs < 0 )
                rhs = 0;
            return lhs + rhs;
         });
    if( !todevice )
    {   // copy result back to host
        if( lengths )
            CUDA_TRY( cudaMemcpyAsync(lengths,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (unsigned int)size;
}


//
unsigned int NVStrings::isalnum( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<bool>(count,0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // alnum requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_ALPHANUM(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    // count the number of trues
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true );
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (unsigned int)matches;
}

unsigned int NVStrings::isalpha( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<bool>(count,0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // alpha requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_ALPHA(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    // count the number of trues
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (unsigned int)matches;
}

//
unsigned int NVStrings::isdigit( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<bool>(count,0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // digit requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_DIGIT(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    // count the number of trues
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (unsigned int)matches;
}

unsigned int NVStrings::isspace( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<bool>(count,0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // space requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_SPACE(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    // count the number of trues
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (unsigned int)matches;
}

unsigned int NVStrings::isdecimal( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<bool>(count,0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // decimal requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_DECIMAL(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    // count the number of trues
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (unsigned int)matches;
}

unsigned int NVStrings::isnumeric( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<bool>(count,0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // numeric requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    brc = IS_NUMERIC(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    // count the number of trues
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (unsigned int)matches;
}

unsigned int NVStrings::islower( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<bool>(count,0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                    brc = !IS_ALPHA(flg) || IS_LOWER(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    // count the number of trues
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (unsigned int)matches;
}

unsigned int NVStrings::isupper( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned char* d_flags = get_unicode_flags();
    bool* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<bool>(count,0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_flags, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = false;
            if( dstr )
            {
                brc = !dstr->empty(); // requires at least one character
                for( auto itr = dstr->begin(); brc && (itr != dstr->end()); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                    brc = !IS_ALPHA(flg) || IS_UPPER(flg);
                }
            }
            d_rtn[idx] = brc;
        });
    // count the number of trues
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (unsigned int)matches;
}

unsigned int NVStrings::is_empty( bool* results, bool todevice )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    bool* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<bool>(count,0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            bool brc = true; // null is empty
            if( dstr )
                brc = dstr->empty(); // requires at least one character
            d_rtn[idx] = brc;
        });
    // count the number of trues
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (unsigned int)matches;
}

//
// s = ["a","xyz", "éee"]
// s.code_points(results)
// results is [   97   120   121   122 50089   101   101]
//
unsigned int NVStrings::code_points( unsigned int* d_results )
{
    auto count = size();
    auto execpol = rmm::exec_policy(0);
    custring_view_array d_strings = pImpl->getStringsPtr();

    // offsets point to each individual integer range
    rmm::device_vector<size_t> offsets(count);
    size_t* d_offsets = offsets.data().get();
    thrust::transform_exclusive_scan(execpol->on(0),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(count),
        d_offsets,
        [d_strings] __device__(size_t idx){
            custring_view* d_str = d_strings[idx];
            size_t length = 0;
            if( d_str )
                length = d_str->chars_count();
            return length;
        },
        0, thrust::plus<unsigned int>());

    // now set the ranges from each strings' character values
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* d_str = d_strings[idx];
            if( !d_str )
                return;
            auto result = d_results + d_offsets[idx];
            for( auto itr = d_str->begin(); itr != d_str->end(); ++itr )
                *result++ = (unsigned int)*itr;
        });
    //
    return offsets.back();
}
