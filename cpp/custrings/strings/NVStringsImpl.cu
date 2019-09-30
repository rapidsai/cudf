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
#include <locale.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/error_utils.hpp>

#include "nvstrings/NVStrings.h"

#include "./NVStringsImpl.h"
#include "../custring_view.cuh"
#include "../custring.cuh"
#include "../unicode/unicode_flags.h"
#include "../unicode/charcases.h"
#include "../util.h"

//
void printCudaError( cudaError_t err, const char* prefix )
{
    if( err == cudaSuccess )
        return;
    fprintf(stderr,"%s: %s(%d):%s\n",prefix,cudaGetErrorName(err),(int)err,cudaGetErrorString(err));
    //cudaError_t err2 = cudaGetLastError(); // clears the error too
    //if( err != err2 )
    //    fprintf(stderr,"  %s:(%d):%s\n",cudaGetErrorName(err2),(int)err2,cudaGetErrorString(err2));
}

//
char32_t* to_char32( const char* ca )
{
    unsigned int size = (unsigned int)strlen(ca);
    unsigned int count = custring_view::chars_in_string(ca,size);
    char32_t* rtn = new char32_t[count+1];
    char32_t* optr = rtn;
    const char* iptr = ca;
    for( unsigned int i=0; i < size; ++i )
    {
        Char oc = 0;
        unsigned int cw = custring_view::char_to_Char(iptr,oc);
        iptr += cw;
        i += cw - 1;
        *optr++ = oc;
    }
    rtn[count] = 0;
    return rtn;
}

//
static unsigned char* d_unicode_flags = nullptr;
unsigned char* get_unicode_flags()
{
    if( !d_unicode_flags )
    {
        // leave this out of RMM since it is never freed
        cudaMalloc(&d_unicode_flags,65536);
        cudaMemcpy(d_unicode_flags,unicode_flags,65536,cudaMemcpyHostToDevice);
    }
    return d_unicode_flags;
}

static unsigned short* d_charcases = nullptr;
unsigned short* get_charcases()
{
    if( !d_charcases )
    {
        // leave this out of RMM since it is never freed
        cudaMalloc(&d_charcases,65536*sizeof(unsigned short));
        cudaMemcpy(d_charcases,charcases,65536*sizeof(unsigned short),cudaMemcpyHostToDevice);
    }
    return d_charcases;
}

//
NVStringsImpl::NVStringsImpl(unsigned int count)
              : bufferSize(0), memoryBuffer(nullptr), bIpcHandle(false), stream_id(0)
{
    pList = new rmm::device_vector<custring_view*>(count,nullptr);
}

NVStringsImpl::~NVStringsImpl()
{
    if( memoryBuffer && !bIpcHandle )
        RMM_FREE(memoryBuffer,0);
    if( bIpcHandle )
        cudaIpcCloseMemHandle(memoryBuffer);

    memoryBuffer = nullptr;
    delete pList;
    pList = nullptr;
    bufferSize = 0;
}


char* NVStringsImpl::createMemoryFor( size_t* d_lengths )
{
    unsigned int count = (unsigned int)pList->size();
    auto execpol = rmm::exec_policy(stream_id);
    bufferSize = thrust::reduce(execpol->on(stream_id), d_lengths, d_lengths+count);
    if( bufferSize==0 )
        return 0; // this is valid; all sizes are zero
    memoryBuffer = device_alloc<char>(bufferSize,stream_id);
    return memoryBuffer;
}

//
int NVStrings_init_from_strings(NVStringsImpl* pImpl, const char** strs, unsigned int count )
{
    cudaError_t err = cudaSuccess;
    auto execpol = rmm::exec_policy(0);
    // first compute the size of each string
    size_t nbytes = 0;
    thrust::host_vector<size_t> hoffsets(count+1,0);
    //hoffsets[0] = 0; --already set by this ----^
    thrust::host_vector<size_t> hlengths(count,0);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        const char* str = strs[idx];
        size_t len = ( str ? (strlen(str)+1) : 0 );
        size_t nsz = len; // include null-terminator
        if( len > 0 )     // len=0 is null, len=1 is empty string
        {
            hlengths[idx] = len; // just the string length
            int nchars = custring_view::chars_in_string(str,(int)len-1);
            nsz = custring_view::alloc_size((int)len-1,nchars);
        }
        nsz = ALIGN_SIZE(nsz);
        nbytes += nsz;
        hoffsets[idx+1] = nbytes;
    }
    // check if they are all null
    if( nbytes==0 )
        return (int)err;

    // Host serialization
    size_t cheat = 0;//sizeof(custring_view);
    char* h_flatstrs = (char*)malloc(nbytes);
    if( !h_flatstrs )
    {
        fprintf(stderr,"init_from_strings: not enough CPU memory for intermediate buffer of size %ld bytes\n", nbytes);
        return -1;
    }
    for( unsigned int idx = 0; idx < count; ++idx )
        memcpy(h_flatstrs + hoffsets[idx] + cheat, strs[idx], hlengths[idx]);

    // copy to device memory
    char* d_flatstrs = nullptr;
    rmmError_t rerr = RMM_ALLOC(&d_flatstrs,nbytes,0);
    if( rerr == RMM_SUCCESS )
        err = cudaMemcpyAsync(d_flatstrs, h_flatstrs, nbytes, cudaMemcpyHostToDevice);
    free(h_flatstrs); // no longer needed
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-sts: alloc/copy %'lu bytes\n",nbytes);
        printCudaError(err);
        return (int)err;
    }

    // copy offsets and lengths to device memory
    rmm::device_vector<size_t> offsets(hoffsets);
    rmm::device_vector<size_t> lengths(hlengths);
    size_t* d_offsets = offsets.data().get();
    size_t* d_lengths = lengths.data().get();

    // initialize custring objects in device memory
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_flatstrs, d_offsets, d_lengths, cheat, d_strings] __device__(unsigned int idx){
            size_t len = d_lengths[idx];
            if( len < 1 )
                return; // null string
            size_t offset = d_offsets[idx];
            char* ptr = d_flatstrs + offset;
            char* str = ptr + cheat;
            d_strings[idx] = custring_view::create_from(ptr,str,(int)len-1);
        });
    //
    //err = cudaDeviceSynchronize();
    //if( err!=cudaSuccess )
    //{
    //    fprintf(stderr,"nvs-sts: sync=%d copy %'u strings\n",(int)err,count);
    //    printCudaError(err);
    //}

    pImpl->setMemoryBuffer(d_flatstrs,nbytes);
    return (int)err;
}

// build strings from array of device pointers and sizes
int NVStrings_init_from_indexes( NVStringsImpl* pImpl, std::pair<const char*,size_t>* indexes, unsigned int count, bool bdevmem, NVStrings::sorttype stype )
{
    cudaError_t err = cudaSuccess;
    rmmError_t rerr = RMM_SUCCESS;
    auto execpol = rmm::exec_policy(0);
    thrust::pair<const char*,size_t>* d_indexes = (thrust::pair<const char*,size_t>*)indexes;
    if( !bdevmem )
    {
        rerr = RMM_ALLOC(&d_indexes,sizeof(std::pair<const char*,size_t>)*count,0);
        if( rerr == RMM_SUCCESS )
            err = cudaMemcpyAsync(d_indexes,indexes,sizeof(std::pair<const char*,size_t>)*count,cudaMemcpyHostToDevice);
    }
    else
    {
        // Lets check what we got from the caller by reading all the memory once.
        // This is wasteful but I cannot keep people from passing bad data:
        //   https://github.com/rapidsai/custrings/issues/191
        // This check cannot be done inline below because libraries like thrust may terminate the process
        // when illegal pointers are passed in. Here we do a pre-check, handle the error and return it.
        // Do not put any other thrust calls before this line in this method.
        try
        {
            thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
                [d_indexes] __device__ (unsigned int idx) {
                    const char* str = d_indexes[idx].first;
                    size_t bytes = d_indexes[idx].second;
                    if( str )
                        custring_view::chars_in_string(str,(unsigned int)bytes);
            });
            err = cudaDeviceSynchronize(); // do not remove this
        }
        catch( thrust::system_error& exc )
        {
            err = (cudaError_t)exc.code().value();
            //printf("exception: %d: %s\n", (int)err, e.what());
        }
    }
    if( err != cudaSuccess || rerr != RMM_SUCCESS )
    {
        printCudaError(err,"nvs-idx: checking parms");
        if( !bdevmem )
            RMM_FREE(d_indexes,0);
        return (int)err;
    }

    // sort the list - helps reduce divergence
    if( stype )
    {
        thrust::sort(execpol->on(0), d_indexes, d_indexes + count,
            [stype] __device__( thrust::pair<const char*,size_t>& lhs, thrust::pair<const char*,size_t>& rhs ) {
                if( lhs.first==0 || rhs.first==0 )
                    return rhs.first!=0; // null < non-null
                int diff = 0;
                if( stype & NVStrings::length )
                    diff = (unsigned int)(lhs.second - rhs.second);
                if( diff==0 && (stype & NVStrings::name) )
                    diff = custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second);
                return (diff < 0);
            });
    }

    // first get the size we need to store these strings
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_indexes, d_sizes] __device__ (unsigned int idx) {
            const char* str = d_indexes[idx].first;
            size_t bytes = d_indexes[idx].second;
            if( str )
                d_sizes[idx] = ALIGN_SIZE(custring_view::alloc_size((char*)str,(int)bytes));
        });

    // allocate device memory
    size_t nbytes = thrust::reduce(execpol->on(0),sizes.begin(),sizes.end());
    //printf("nvs-idx: %'lu bytes\n",nbytes);
    if( nbytes==0 ) {
        if( !bdevmem )
            RMM_FREE(d_indexes,0);
        return 0;  // done, all the strings were null
    }
    char* d_flatdstrs = nullptr;
    rerr = RMM_ALLOC(&d_flatdstrs,nbytes,0);
    if( rerr != RMM_SUCCESS )
    {
        fprintf(stderr,"nvs-idx: RMM_ALLOC(%p,%lu)=%d\n", d_flatdstrs,nbytes,(int)rerr);
        //printCudaError(err);
        if( !bdevmem )
            RMM_FREE(d_indexes,0);
        return (int)err;
    }

    // build offsets array
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());

    // now build the strings vector
    custring_view_array d_strings = pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_indexes, d_flatdstrs, d_offsets, d_sizes, d_strings] __device__(unsigned int idx){
            // add string to internal vector array
            const char* str = d_indexes[idx].first;
            size_t bytes = d_indexes[idx].second;
            size_t offset = d_offsets[idx];
            char* ptr = d_flatdstrs + offset;
            custring_view* dstr = 0;
            if( str )
                dstr = custring_view::create_from(ptr,(char*)str,(int)bytes);
            d_strings[idx] = dstr;
            d_sizes[idx] = bytes;
        });
    //
    pImpl->setMemoryBuffer(d_flatdstrs,nbytes);
    if( !bdevmem )
        RMM_FREE(d_indexes,0);
    return (int)err;
}

// build strings from pointer and array of offsets
int NVStrings_init_from_offsets( NVStringsImpl* pImpl, const char* strs, int count, const int* offsets, const unsigned char* bitmask, int nulls )
{
    if( count==nulls )
        return 0; // if all are nulls then we are done
    cudaError_t err = cudaSuccess;
    auto execpol = rmm::exec_policy(0);

    // first compute the size of each string
    size_t nbytes = 0;
    thrust::host_vector<size_t> hoffsets(count+1,0);
    thrust::host_vector<size_t> hlengths(count,0);
    for( int idx=0; idx < count; ++idx )
    {
        int offset = offsets[idx];
        int len = offsets[idx+1] - offset;
        const char* str = strs + offset;
        int nchars = custring_view::chars_in_string(str,len);
        int bytes = custring_view::alloc_size(len,nchars);
        if( bitmask && ((bitmask[idx/8] & (1 << (idx % 8)))==0) ) // from arrow spec
            bytes = 0;
        hlengths[idx] = len;
        nbytes += ALIGN_SIZE(bytes);
        hoffsets[idx+1] = nbytes;
    }
    if( nbytes==0 )
        return 0; // should not happen

    // serialize host memory into a new buffer
    unsigned int cheat = 0;//sizeof(custring_view);
    char* h_flatstrs = (char*)malloc(nbytes);
    for( int idx = 0; idx < count; ++idx )
        memcpy(h_flatstrs + hoffsets[idx] + cheat, strs + offsets[idx], hlengths[idx]);

    // copy whole thing to device memory
    char* d_flatstrs = nullptr;
    rmmError_t rerr = RMM_ALLOC(&d_flatstrs,nbytes,0);
    if( rerr == RMM_SUCCESS )
        err = cudaMemcpyAsync(d_flatstrs, h_flatstrs, nbytes, cudaMemcpyHostToDevice);
    free(h_flatstrs); // no longer needed
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-ofs: alloc/copy %'lu bytes\n",nbytes);
        printCudaError(err);
        return (int)err;
    }

    // copy offsets and lengths to device memory
    rmm::device_vector<size_t> doffsets(hoffsets);
    rmm::device_vector<size_t> dlengths(hlengths);
    size_t* d_offsets = doffsets.data().get();
    size_t* d_lengths = dlengths.data().get();

    // initialize custring objects in device memory
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_flatstrs, d_offsets, d_lengths, cheat, d_strings] __device__(unsigned int idx){
            size_t len = d_lengths[idx];
            size_t offset = d_offsets[idx];
            size_t size = d_offsets[idx+1] - offset;
            if( size < 1 )
                return; // null string
            char* ptr = d_flatstrs + offset;
            char* str = ptr + cheat;
            d_strings[idx] = custring_view::create_from(ptr,str,len);
        });
    //
    pImpl->setMemoryBuffer(d_flatstrs,nbytes);
    return (int)err;
}

// build strings from array of device pointers and sizes
int NVStrings_init_from_device_offsets( NVStringsImpl* pImpl, const char* strs, int count, const int* offsets, const unsigned char* bitmask, int nulls )
{
    if( count==nulls )
        return 0; // if all are nulls then we are done
    auto execpol = rmm::exec_policy(0);

    // first compute the size of each string
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [strs, offsets, bitmask, d_sizes] __device__(unsigned int idx){
            if( bitmask && ((bitmask[idx/8] & (1 << (idx % 8)))==0) ) // from arrow spec
                return;
            int offset = offsets[idx];
            int len = offsets[idx+1] - offset;
            const char* str = strs + offset;
            int nchars = custring_view::chars_in_string(str,len);
            int bytes = custring_view::alloc_size(len,nchars);
            d_sizes[idx] = ALIGN_SIZE(bytes);
        });

    // copy whole thing to device memory
    char* d_buffer = pImpl->createMemoryFor(d_sizes);
    if( !d_buffer )
        return 0; // nothing to do

    // copy offsets and lengths to device memory
    rmm::device_vector<size_t> out_offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),out_offsets.begin());
    size_t* d_out_offsets = out_offsets.data().get();

    // initialize custring objects in device memory
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [strs, offsets, bitmask, d_buffer, d_out_offsets, d_strings] __device__(unsigned int idx){
            if( bitmask && ((bitmask[idx/8] & (1 << (idx % 8)))==0) )
                return; // null string
            int offset = offsets[idx];
            int len = offsets[idx+1] - offset;
            const char* in_str = strs + offset;
            char* out_str = d_buffer + d_out_offsets[idx];
            d_strings[idx] = custring_view::create_from(out_str,in_str,len);
        });
    //
    return 0;
}

int NVStrings_copy_strings( NVStringsImpl* pImpl, std::vector<NVStringsImpl*>& strslist )
{
    auto execpol = rmm::exec_policy(0);
    auto pList = pImpl->pList;
    unsigned int count = (unsigned int)pList->size();
    size_t nbytes = 0;
    for( auto itr=strslist.begin(); itr!=strslist.end(); itr++ )
        nbytes += (*itr)->getMemorySize();

    custring_view_array d_results = pList->data().get();
    char* d_buffer = device_alloc<char>(nbytes,0);
    size_t ptr_offset = 0;
    size_t buffer_offset = 0;

    for( auto itr=strslist.begin(); itr!=strslist.end(); itr++ )
    {
        NVStringsImpl* strs = *itr;
        unsigned int size = strs->getCount();
        size_t buffer_size = strs->getMemorySize();
        if( size==0 )
            continue;
        rmm::device_vector<custring_view*> strings(size,nullptr);
        custring_view** d_strings = strings.data().get();
        // copy the pointers
        CUDA_TRY( cudaMemcpyAsync( d_strings, strs->getStringsPtr(), size*sizeof(custring_view*), cudaMemcpyDeviceToDevice));
        if( buffer_size )
        {
            // copy string memory
            char* baseaddr = strs->getMemoryPtr();
            char* buffer = d_buffer + buffer_offset;
            CUDA_TRY( cudaMemcpyAsync(buffer, baseaddr, buffer_size, cudaMemcpyDeviceToDevice) );
            // adjust pointers
            custring_view_array results = d_results + ptr_offset;
            thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), size,
                [buffer, baseaddr, d_strings, results] __device__(unsigned int idx){
                    char* dstr = (char*)d_strings[idx];
                    if( !dstr )
                        return;
                    size_t diff = dstr - baseaddr;
                    char* newaddr = buffer + diff;
                    results[idx] = (custring_view*)newaddr;
            });
        }
        ptr_offset += size;
        buffer_offset += buffer_size;
    }
    //
    pImpl->setMemoryBuffer(d_buffer,nbytes);
    return count;
}

int NVStrings_fixup_pointers( NVStringsImpl* pImpl, char* baseaddr )
{
    auto execpol = rmm::exec_policy(0);
    auto pList = pImpl->pList;
    unsigned int count = (unsigned int)pList->size();

    custring_view_array d_strings = pImpl->getStringsPtr();
    //---- the following can be used to find the base-address of the original memory  ----
    //---- instead of passing it across the ipc boundary; leaving it here for now     ----
    //custring_view** first = thrust::min_element(execpol->on(0),d_strings,d_strings+count,
    //    [] __device__ (custring_view* lhs, custring_view* rhs) {
    //        return (lhs && rhs) ? (lhs < rhs) : rhs==0;
    //    });
    //cudaError_t err = cudaMemcpy(&baseaddr,first,sizeof(custring_view*),cudaMemcpyDeviceToHost);
    //if( err!=cudaSuccess )
    //    fprintf(stderr, "fixup: cudaMemcpy(%p,%p,%d)=%d\n",&baseaddr,first,(int)sizeof(custring_view*),(int)err);
    //
    char* buffer = pImpl->getMemoryPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [buffer, baseaddr, d_strings] __device__(unsigned int idx){
            char* dstr = (char*)d_strings[idx];
            if( !dstr )
                return;
            size_t diff = dstr - baseaddr;
            char* newaddr = buffer + diff;
            d_strings[idx] = (custring_view*)newaddr;
        });
    //cudaError_t err = cudaDeviceSynchronize();
    //if( err!=cudaSuccess )
    //    printCudaError(err,"nvs-fixup");
    return count;
}
