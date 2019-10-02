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
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/error_utils.hpp>

#include "nvstrings/NVStrings.h"

#include "./NVStringsImpl.h"
#include "../custring_view.cuh"
#include "../util.h"

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

//
unsigned int NVStrings::compare( const char* str, int* results, bool todevice )
{
    unsigned int count = size();
    if( str==0 || results==0 || count==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(str);
    if( bytes==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    char* d_str = device_alloc<char>(bytes,0);
    CUDA_TRY( cudaMemcpyAsync(d_str,str,bytes,cudaMemcpyHostToDevice))

    int* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<int>(count,0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, bytes, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->compare(d_str,bytes);
            else
                d_rtn[idx] = (d_str ? -1: 0);
        });
    //
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, 0);
    //
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    RMM_FREE(d_str,0);
    return (unsigned int)matches;
}

// searches from the beginning of each string
unsigned int NVStrings::find( const char* str, int start, int end, int* results, bool todevice )
{
    unsigned int count = size();
    if( str==0 || count==0 || results==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(str)+1; // the +1 allows searching for empty string
    if( start < 0 )
        start = 0;

    auto execpol = rmm::exec_policy(0);
    char* d_str = device_alloc<char>(bytes,0);
    CUDA_TRY( cudaMemcpyAsync(d_str,str,bytes,cudaMemcpyHostToDevice))

    int* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<int>(count,0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, bytes, start, end, d_rtn] __device__(unsigned int idx){
            //__shared__ char tgt[24];
            char* dtgt = d_str;
            //if( bytes<24  )
            //{
            //    dtgt = tgt;
            //    if( threadIdx.x==0 )
            //        memcpy(dtgt,d_str,bytes);
            //}
            //__syncthreads();
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->find(dtgt,bytes-1,start,end-start);
            else
                d_rtn[idx] = -2; // indicate null to caller
        });
    //
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(int val){ return val!=-1; });
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    RMM_FREE(d_str,0);
    return rtn;
}

// searches from the beginning of each string and specified individual starting positions
unsigned int NVStrings::find_from( const char* str, int* starts, int* ends, int* results, bool todevice )
{
    unsigned int count = size();
    if( str==0 || count==0 || results==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(str)+1; // the +1 allows searching for empty string

    auto execpol = rmm::exec_policy(0);
    char* d_str = device_alloc<char>(bytes,0);
    CUDA_TRY( cudaMemcpyAsync(d_str,str,bytes,cudaMemcpyHostToDevice))

    int* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<int>(count,0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, bytes, starts, ends, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {
                int pos = (starts ? starts[idx] : 0);
                int len = (ends ? (ends[idx]-pos) : -1);
                d_rtn[idx] = dstr->find(d_str,bytes-1,pos,len);
            }
            else
                d_rtn[idx] = -2; // indicate null to caller
        });
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(int val){ return val!=-1; });
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    RMM_FREE(d_str,0);
    return rtn;
}

// searches from the end of each string
unsigned int NVStrings::rfind( const char* str, int start, int end, int* results, bool todevice )
{
    unsigned int count = size();
    if( str==0 || count==0 || results==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(str)+1;
    if( start < 0 )
        start = 0;

    auto execpol = rmm::exec_policy(0);
    char* d_str = device_alloc<char>(bytes,0);
    CUDA_TRY( cudaMemcpyAsync(d_str,str,bytes,cudaMemcpyHostToDevice))

    int* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<int>(count,0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, bytes, start, end, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->rfind(d_str,bytes-1,start,end-start);
            else
                d_rtn[idx] = -2; // indicate null to caller
        });
    //
    // count the number of successful finds
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(int val){ return val!=-1; });
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    RMM_FREE(d_str,0);
    return rtn;
}

//
unsigned int NVStrings::find_multiple( NVStrings& strs, int* results, bool todevice )
{
    unsigned int count = size();
    unsigned int tcount = strs.size();
    if( results==0 || count==0 || tcount==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    int* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<int>(tcount*count,0);
    //
    custring_view_array d_strings = pImpl->getStringsPtr();
    custring_view_array d_targets = strs.pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_targets, tcount, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            for( int jdx=0; jdx < tcount; ++jdx )
            {
                custring_view* dtgt = d_targets[jdx];
                d_rtn[(idx*tcount)+jdx] = ( (dstr && dtgt) ? dstr->find(*dtgt) : -2 );
            }
        });
    //
    unsigned int rtn = thrust::count_if(execpol->on(0), d_rtn, d_rtn+count, [] __device__(int val){ return val!=-1; });
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(int)*count*tcount,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return rtn;
}


// does specified string occur in each string
int NVStrings::contains( const char* str, bool* results, bool todevice )
{
    if( str==0 || results==0 )
        return -1;
    unsigned int count = size();
    if( count==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(str)+1; // the +1 allows searching for empty string

    auto execpol = rmm::exec_policy(0);
    char* d_str = device_alloc<char>(bytes,0);
    CUDA_TRY( cudaMemcpyAsync(d_str,str,bytes,cudaMemcpyHostToDevice))

    bool* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<bool>(count,0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, bytes, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->find(d_str,bytes-1)>=0;
            else
                d_rtn[idx] = false;
        });
    // count the number of successful finds
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    RMM_FREE(d_str,0);
    return matches;
}


//
int NVStrings::match_strings( NVStrings& strs, bool* results, bool bdevmem )
{
    if( results==0 )
        return -1;
    unsigned int count = size();
    if( count==0 )
        return 0;
    if( count != strs.size() )
        throw std::invalid_argument("sizes must match");

    auto execpol = rmm::exec_policy(0);
    custring_view_array d_strings1 = pImpl->getStringsPtr();
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings2 = strings.data().get();
    strs.create_custring_index(d_strings2);

    bool* d_rtn = results;
    if( !bdevmem )
        d_rtn = device_alloc<bool>(count,0);

    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings1, d_strings2, d_rtn] __device__(unsigned int idx){
            custring_view* dstr1 = d_strings1[idx];
            custring_view* dstr2 = d_strings2[idx];
            if( dstr1 && dstr2 )
                d_rtn[idx] = dstr1->compare(*dstr2)==0;
            else
                d_rtn[idx] = dstr1==dstr2;
        });
    // count the number of successful finds
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !bdevmem )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return matches;
}

//
unsigned int NVStrings::startswith( const char* str, bool* results, bool todevice )
{
    unsigned int count = size();
    if( str==0 || count==0 || results==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(str)+1; // the +1 allows searching for empty string

    auto execpol = rmm::exec_policy(0);
    char* d_str = device_alloc<char>(bytes,0);
    CUDA_TRY( cudaMemcpyAsync(d_str,str,bytes,cudaMemcpyHostToDevice))

    bool* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<bool>(count,0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, bytes, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->starts_with(d_str,bytes-1);
            else
                d_rtn[idx] = false;
        });
    //
    // count the number of successful finds
    unsigned int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true );
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    RMM_FREE(d_str,0);
    return matches;
}

//
unsigned int NVStrings::endswith( const char* str, bool* results, bool todevice )
{
    unsigned int count = size();
    if( str==0 || count==0 || results==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(str)+1; // the +1 allows searching for empty string

    auto execpol = rmm::exec_policy(0);
    char* d_str = device_alloc<char>(bytes,0);
    CUDA_TRY( cudaMemcpyAsync(d_str,str,bytes,cudaMemcpyHostToDevice))

    bool* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<bool>(count,0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_str, bytes, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->ends_with(d_str,bytes-1);
            else
                d_rtn[idx] = false;
        });
    //
    // count the number of successful finds
    unsigned int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true );
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    RMM_FREE(d_str,0);
    return matches;
}
