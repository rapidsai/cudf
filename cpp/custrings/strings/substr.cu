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

#include <exception>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include "nvstrings/NVStrings.h"

#include "./NVStringsImpl.h"
#include "../custring_view.cuh"


// Extract character from each component at specified position
NVStrings* NVStrings::get(unsigned int pos)
{
    return slice(pos,pos+1,1);
}


// All strings are substr'd with the same (start,stop) position values.
NVStrings* NVStrings::slice( int start, int stop, int step )
{
    if( (stop > 0) && (start > stop) )
        throw std::invalid_argument("nvstrings::slice start cannot be greater than stop");

    auto execpol = rmm::exec_policy(0);
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, start, stop, step, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int len = ( stop < 0 ? dstr->chars_count() : stop ) - start;
            unsigned int size = dstr->substr_size((unsigned)start,(unsigned)len,(unsigned)step);
            size = ALIGN_SIZE(size);
            d_lengths[idx] = (size_t)size;
        });
    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
        return rtn;
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // slice it and dice it
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, start, stop, step, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            char* buffer = d_buffer + d_offsets[idx];
            int len = ( stop < 0 ? dstr->chars_count() : stop ) - start;
            d_results[idx] = dstr->substr((unsigned)start,(unsigned)len,(unsigned)step,buffer);
        });
    //
    return rtn;
}

// Each string is substr'd according to the individual (start,stop) position values
NVStrings* NVStrings::slice_from( const int* starts, const int* stops )
{
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    auto execpol = rmm::exec_policy(0);
    // compute size of output buffer
    rmm::device_vector<size_t> lengths(count,0);
    size_t* d_lengths = lengths.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, starts, stops, d_lengths] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int start = (starts ? starts[idx]:0);
            int stop = (stops ? stops[idx]: -1);
            int len = ( stop < 0 ? dstr->chars_count() : stop ) - start;
            unsigned int size = dstr->substr_size((unsigned)start,(unsigned)len);
            size = ALIGN_SIZE(size);
            d_lengths[idx] = (size_t)size;
        });
    // create output object
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
    if( d_buffer==0 )
        return rtn;
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    // slice, slice, baby
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, starts, stops, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int start = (starts ? starts[idx]:0);
            int stop = (stops ? stops[idx]: -1);
            char* buffer = d_buffer + d_offsets[idx];
            int len = ( stop < 0 ? dstr->chars_count() : stop ) - start;
            d_results[idx] = dstr->substr((unsigned)start,(unsigned)len,1,buffer);
        });
    //
    return rtn;
}

