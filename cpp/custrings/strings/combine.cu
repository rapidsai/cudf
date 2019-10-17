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
#include <thrust/for_each.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/error_utils.hpp>

#include "nvstrings/NVStrings.h"

#include "./NVStringsImpl.h"
#include "../custring_view.cuh"
#include "../util.h"

//
NVStrings* NVStrings::cat( NVStrings* others, const char* separator, const char* narep )
{
    if( others==0 )
        return nullptr; // return a copy of ourselves?
    unsigned int count = size();
    if( others->size() != count )
        throw std::invalid_argument("nvstrings::cat sizes do not match");

    auto execpol = rmm::exec_policy(0);
    unsigned int seplen = 0;
    if( separator )
        seplen = (unsigned int)strlen(separator);
    char* d_sep = nullptr;
    if( seplen )
    {
        d_sep = device_alloc<char>(seplen,0);
        CUDA_TRY( cudaMemcpyAsync(d_sep,separator,seplen,cudaMemcpyHostToDevice))
    }
    unsigned int narlen = 0;
    char* d_narep = nullptr;
    if( narep )
    {
        narlen = (unsigned int)strlen(narep);
        d_narep = device_alloc<char>(narlen+1,0);
        CUDA_TRY( cudaMemcpyAsync(d_narep,narep,narlen+1,cudaMemcpyHostToDevice))
    }

    custring_view_array d_strings = pImpl->getStringsPtr();
    custring_view_array d_others = others->pImpl->getStringsPtr();

    // first compute the size of the output
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_others, d_sep, seplen, d_narep, narlen, d_sizes] __device__(unsigned int idx){
            custring_view* dstr1 = d_strings[idx];
            custring_view* dstr2 = d_others[idx];
            if( (!dstr1 || !dstr2) && !d_narep )
                return; // null case
            int nchars = 0;
            int bytes = 0;
            // left side
            if( dstr1 )
            {
                nchars = dstr1->chars_count();
                bytes = dstr1->size();
            }
            else if( d_narep )
            {
                nchars = custring_view::chars_in_string(d_narep,narlen);
                bytes = narlen;
            }
            // separator
            if( d_sep )
            {
                nchars += custring_view::chars_in_string(d_sep,seplen);
                bytes += seplen;
            }
            // right side
            if( dstr2 )
            {
                nchars += dstr2->chars_count();
                bytes += dstr2->size();
            }
            else if( d_narep )
            {
                nchars += custring_view::chars_in_string(d_narep,narlen);
                bytes += narlen;
            }
            int size = custring_view::alloc_size(bytes,nchars);
            //printf("cat:%lu:size=%d\n",idx,size);
            size = ALIGN_SIZE(size);
            d_sizes[idx] = size;
        });

    // allocate the memory for the output
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    if( d_buffer==0 )
    {
        if( d_sep )
            RMM_FREE(d_sep,0);
        if( d_narep )
            RMM_FREE(d_narep,0);
        return rtn;
    }
    cudaMemset(d_buffer,0,rtn->pImpl->getMemorySize());
    // compute the offset
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // do the thing
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_others, d_sep, seplen, d_narep, narlen, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            char* buffer = d_buffer + d_offsets[idx];
            custring_view* dstr1 = d_strings[idx];
            custring_view* dstr2 = d_others[idx];
            if( (!dstr1 || !dstr2) && !d_narep )
                return; // if null, an no null rep, done
            custring_view* dout = custring_view::create_from(buffer,0,0); // init empty string
            if( dstr1 )
                dout->append(*dstr1);        // add left side
            else if( d_narep )               // (or null rep)
                dout->append(d_narep,narlen);
            if( d_sep )
                dout->append(d_sep,seplen);  // add separator
            if( dstr2 )
                dout->append(*dstr2);        // add right side
            else if( d_narep )               // (or null rep)
                dout->append(d_narep,narlen);
            //printf("cat:%lu:[]=%d\n",idx,dout->size());
            d_results[idx] = dout;
    });
    //printCudaError(cudaDeviceSynchronize(),"nvs-cat: combining strings");

    if( d_sep )
        RMM_FREE(d_sep,0);
    if( d_narep )
        RMM_FREE(d_narep,0);
    return rtn;
}

//
NVStrings* NVStrings::cat( std::vector<NVStrings*>& others, const char* separator, const char* narep )
{
    if( others.size()==0 )
        return nullptr; // return a copy of ourselves?
    unsigned int count = size();
    for( auto itr=others.begin(); itr!=others.end(); itr++ )
        if( (*itr)->size() != count )
            throw std::invalid_argument("nvstrings::cat sizes do not match");

    auto execpol = rmm::exec_policy(0);
    custring_view* d_separator = nullptr;
    if( separator )
    {
        unsigned int seplen = (unsigned int)strlen(separator);
        unsigned int sep_size = custring_view::alloc_size(separator,seplen);
        d_separator = reinterpret_cast<custring_view*>(device_alloc<char>(sep_size,0));
        custring_view::create_from_host(d_separator,separator,seplen);
    }
    custring_view* d_narep = nullptr;
    if( narep )
    {
        unsigned int narlen = (unsigned int)strlen(narep);
        unsigned int nar_size = custring_view::alloc_size(narep,narlen);
        d_narep = reinterpret_cast<custring_view*>(device_alloc<char>(nar_size,0));
        custring_view::create_from_host(d_narep,narep,narlen);
    }

    custring_view_array d_strings = pImpl->getStringsPtr();
    rmm::device_vector<custring_view_array> dothers;
    for( auto itr=others.begin(); itr!=others.end(); itr++ )
        dothers.push_back((*itr)->pImpl->getStringsPtr());
    custring_view_array* d_others = dothers.data().get();
    unsigned int others_count = (unsigned int)others.size();

    // first compute the size of the output
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_others, others_count, d_separator, d_narep, d_sizes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            int nchars = 0;
            int bytes = 0;
            bool allnulls = !dstr && !d_narep;
            if( dstr )
            {
                nchars += dstr->chars_count();
                bytes += dstr->size();
            }
            else if( d_narep )
            {
                nchars += d_narep->chars_count();
                bytes += d_narep->size();
            }
            for( unsigned int jdx=0; !allnulls && (jdx < others_count); ++jdx )
            {
                custring_view_array dcat2 = d_others[jdx];
                dstr = dcat2[idx];
                allnulls = !dstr && !d_narep;
                if( d_separator )
                {
                    nchars += d_separator->chars_count();
                    bytes += d_separator->size();
                }
                if( dstr )
                {
                    nchars += dstr->chars_count();
                    bytes += dstr->size();
                }
                else if( d_narep )
                {
                    nchars += d_narep->chars_count();
                    bytes += d_narep->size();
                }
            }
            int size = custring_view::alloc_size(bytes,nchars);
            size = ALIGN_SIZE(size);
            if( allnulls )
                size = 0;
            //printf("cat:%lu:size=%d\n",idx,size);
            d_sizes[idx] = size;
        });

    // allocate the memory for the output
    NVStrings* rtn = new NVStrings(count);
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    if( d_buffer==0 )
    {
        if( d_separator )
            RMM_FREE(d_separator,0);
        if( d_narep )
            RMM_FREE(d_narep,0);
        return rtn;
    }
    cudaMemset(d_buffer,0,rtn->pImpl->getMemorySize());
    // compute the offset
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // do the thing
    custring_view_array d_results = rtn->pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_others, others_count, d_separator, d_narep, d_buffer, d_sizes, d_offsets, d_results] __device__(unsigned int idx){
            if( d_sizes[idx]==0 )
                return; // null string
            char* buffer = d_buffer + d_offsets[idx];
            custring_view* dstr = d_strings[idx];
            custring_view* dout = custring_view::create_from(buffer,0,0); // init empty string
            if( dstr )
                dout->append(*dstr);
            else if( d_narep )
                dout->append(*d_narep);
            for( unsigned int jdx=0; jdx < others_count; ++jdx )
            {
                custring_view_array dcat2 = d_others[jdx];
                dstr = dcat2[idx];
                if( d_separator )
                    dout->append(*d_separator);
                if( dstr )
                    dout->append(*dstr);
                else if( d_narep )
                    dout->append(*d_narep);
            }
            //printf("cat:%lu:[]=%d\n",idx,dout->size());
            d_results[idx] = dout;
    });
    //printCudaError(cudaDeviceSynchronize(),"nvs-cat: combining strings");

    if( d_separator )
        RMM_FREE(d_separator,0);
    if( d_narep )
        RMM_FREE(d_narep,0);
    return rtn;
}

// this returns one giant string joining all the strings
// in the list with the delimiter string between each one
NVStrings* NVStrings::join( const char* delimiter, const char* narep )
{
    if( delimiter==0 )
        throw std::invalid_argument("nvstrings::join delimiter cannot be null");
    auto execpol = rmm::exec_policy(0);
    unsigned int dellen = (unsigned int)strlen(delimiter);
    char* d_delim = nullptr;
    if( dellen > 0 )
    {
        d_delim = device_alloc<char>(dellen,0);
        CUDA_TRY( cudaMemcpyAsync(d_delim,delimiter,dellen,cudaMemcpyHostToDevice))
    }
    unsigned int narlen = 0;
    char* d_narep = nullptr;
    if( narep )
    {
        narlen = (unsigned int)strlen(narep);
        d_narep = device_alloc<char>(narlen+1,0);
        CUDA_TRY( cudaMemcpyAsync(d_narep,narep,narlen+1,cudaMemcpyHostToDevice))
    }

    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();

    // need to compute the giant buffer size
    rmm::device_vector<size_t> lens(count,0);
    size_t* d_lens = lens.data().get();
    rmm::device_vector<size_t> chars(count,0);
    size_t* d_chars = chars.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delim, dellen, d_narep, narlen, count, d_lens, d_chars] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            int dlen = dellen;
            int nchars = 0;
            int bytes = 0;
            if( idx+1 >= count )
                dlen = 0; // no trailing delimiter
            if( dstr )
            {
                nchars = dstr->chars_count();
                bytes = dstr->size();
            }
            else if( d_narep )
            {
                nchars = custring_view::chars_in_string(d_narep,narlen);
                bytes = narlen;
            }
            else
                dlen = 0; // for null, no delimiter
            if( dlen )
            {
                nchars += custring_view::chars_in_string(d_delim,dellen);
                bytes += dellen;
            }
            d_lens[idx] = bytes;
            d_chars[idx] = nchars;
        });

    //cudaDeviceSynchronize();
    // compute how much space is required for the giant string
    size_t totalBytes = thrust::reduce(execpol->on(0), lens.begin(), lens.end());
    size_t totalChars = thrust::reduce(execpol->on(0), chars.begin(), chars.end());
    //printf("totalBytes=%ld, totalChars=%ld\n",totalBytes,totalChars);
    size_t allocSize = custring_view::alloc_size((unsigned int)totalBytes,(unsigned int)totalChars);
    //printf("allocSize=%ld\n",allocSize);

    // convert the lens values into offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(lens.begin(),lens.end(),offsets.begin());
    size_t* d_offsets = offsets.data().get();
    // create one big buffer to hold the strings
    char* d_buffer = nullptr;
    rmmError_t rmmerr = RMM_ALLOC(&d_buffer,allocSize,0);
    if( rmmerr != RMM_SUCCESS )
    {
        std::ostringstream message;
        message << "allocate error " << rmmerr;
        throw std::runtime_error(message.str());
    }
    NVStrings* rtn = new NVStrings(1);
    custring_view_array d_result = rtn->pImpl->getStringsPtr();
    rtn->pImpl->setMemoryBuffer(d_buffer,allocSize);
    // copy the strings into it
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_buffer, d_offsets, count, d_delim, dellen, d_narep, narlen] __device__(unsigned int idx){
            char* sptr = d_buffer + 8 + d_offsets[idx];
            char* dlim = d_delim;
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {   // copy string to output
                int ssz = dstr->size();
                memcpy(sptr,dstr->data(),ssz);
                sptr += ssz;
            }
            else if( d_narep )
            {   // or copy null-replacement to output
                memcpy(sptr,d_narep,narlen);
                sptr += narlen;
            }
            else // or copy nothing to output
                dlim = 0; // prevent delimiter copy below
            // copy delimiter to output
            if( (idx+1 < count) && dlim )
                memcpy(sptr,dlim,dellen);
        });

    // assign to resulting custring_view
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), 1,
        [d_buffer, totalBytes, d_result] __device__ (unsigned int idx){
            char* sptr = d_buffer + 8;
            d_result[0] = custring_view::create_from(d_buffer,sptr,totalBytes);
        });
    //printCudaError(cudaDeviceSynchronize(),"nvs-join");

    if( d_delim )
        RMM_FREE(d_delim,0);
    if( d_narep )
        RMM_FREE(d_narep,0);
    return rtn;
}
