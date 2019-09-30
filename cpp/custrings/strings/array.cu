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

#include <stdexcept>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/count.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/error_utils.hpp>

#include "nvstrings/NVStrings.h"

#include "./NVStringsImpl.h"
#include "../custring_view.cuh"
#include "../util.h"

// takes scattered pointers to custring_view objects and
// initializes a new NVStringsImpl
void NVStrings_init_from_custrings( NVStringsImpl* pImpl, custring_view_array d_strings, unsigned int count )
{
    auto execpol = rmm::exec_policy(0);
    // get individual sizes
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_sizes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_sizes[idx] = ALIGN_SIZE(dstr->alloc_size());
        });
    // create output object
    char* d_buffer = pImpl->createMemoryFor(d_sizes);
    if( d_buffer==0 )
        return; // this is valid
    // create offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // finally, copy the strings
    custring_view_array d_results = pImpl->getStringsPtr();
    size_t* d_offsets = offsets.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_buffer, d_offsets, d_results] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            char* buffer = d_buffer + d_offsets[idx];
            d_results[idx] = custring_view::create_from(buffer,*dstr);
        });
    //
}

// create a new instance containing only the strings at the specified positions
// position values can be in any order and can even be repeated
NVStrings* NVStrings::gather( const int* pos, unsigned int elements, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 || elements==0 || pos==0 )
        return new NVStrings(0);

    auto execpol = rmm::exec_policy(0);
    const int* d_pos = pos;
    if( !bdevmem )
    {   // copy indexes to device memory
        d_pos = const_cast<const int*>(device_alloc<int>(elements,0));
        CUDA_TRY(cudaMemcpyAsync((void*)d_pos,pos,elements*sizeof(int),cudaMemcpyHostToDevice))
    }
    // create working memory
    rmm::device_vector<custring_view*> results(elements,nullptr);
    auto d_results = results.data().get();
    rmm::device_vector<bool> flags(elements,false);
    auto d_flags = flags.data().get();
    custring_view_array d_strings = pImpl->getStringsPtr();
    // do the gather
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), elements,
        [d_strings, d_pos, count, d_results, d_flags] __device__(unsigned int idx){
            int pos = d_pos[idx];
            if( (pos < 0) || (pos >= count) )
                d_flags[idx] = true;
            else
                d_results[idx] = d_strings[pos];
        });
    // check for invalid position values
    if( thrust::count(execpol->on(0), flags.begin(), flags.end(), true) )
    {
        if( !bdevmem )
            RMM_FREE((void*)d_pos,0);
        throw std::out_of_range("gather position value out of range");
    }
    // build resulting instance
    NVStrings* rtn = new NVStrings(elements);
    NVStrings_init_from_custrings(rtn->pImpl, d_results, elements);
    if( !bdevmem )
        RMM_FREE((void*)d_pos,0);
    return rtn;
}

// create a new instance containing only the strings where the corresponding mask value is true
NVStrings* NVStrings::gather( const bool* mask, bool bdevmem )
{
    size_t count = size();
    if( count==0 || mask==nullptr )
        return new NVStrings(0);
    // copy mask array to device memory if necessary
    auto execpol = rmm::exec_policy(0);
    const bool* d_mask = mask;
    if( !bdevmem )
    {
        d_mask = const_cast<const bool*>(device_alloc<bool>(count,0));
        CUDA_TRY(cudaMemcpyAsync((void*)d_mask,mask,count*sizeof(mask[0]),cudaMemcpyHostToDevice,0))
    }
    // create list of index positions from the mask array
    rmm::device_vector<int> indexes(count);
    auto d_indexes = indexes.data().get();
    auto d_indexes_end = thrust::copy_if(execpol->on(0), thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(count),
                                         d_indexes, [d_mask] __device__ (int idx) { return d_mask[idx]; });
    // done with the mask
    if( !bdevmem )
        RMM_FREE((void*)d_mask,0);
    count = d_indexes_end - d_indexes;
    return gather( d_indexes, count, true );
}

//
// s1 = ['a','b,'c','d']
// s2 = ['e','f']
// pos = [1,3]  -- must be the same length as s2
// s3 = s1.scatter(s2,pos)
// ['a','e','c','f']
//
NVStrings* NVStrings::scatter( NVStrings& strs, const int* pos, bool bdevmem )
{
    unsigned int count = size();
    unsigned int elements = strs.size();
    if( pos==0 )
        throw std::invalid_argument("position parameter cannot be null");

    auto execpol = rmm::exec_policy(0);
    const int* d_pos = pos;
    if( !bdevmem )
    {   // copy indexes to device memory
        d_pos = const_cast<const int*>(device_alloc<int>(elements,0));
        CUDA_TRY(cudaMemcpyAsync((void*)d_pos,pos,elements*sizeof(int),cudaMemcpyHostToDevice))
    }
    // The most efficient method here is to build pointer array
    // applying the parameters to the specified positions and
    // then build a new instance from the resulting pointers.
    rmm::device_vector<custring_view*> results(count,nullptr);
    auto d_results = results.data().get();
    custring_view_array d_strings = pImpl->getStringsPtr();
    custring_view_array d_new_strings = strs.pImpl->getStringsPtr();
    thrust::copy( execpol->on(0), d_strings, d_strings+count, d_results );
    thrust::scatter( execpol->on(0), d_new_strings, d_new_strings+elements, d_pos, d_results );
    // build resulting instance
    NVStrings* rtn = new NVStrings(count);
    NVStrings_init_from_custrings(rtn->pImpl, d_results, count);
    if( !bdevmem )
        RMM_FREE((void*)d_pos,0);
    return rtn;
}

//
// s1 = ['a','b,'c','d']
// pos = [1,3]
// s3 = s1.scatter('e',pos,2)
// ['a','e','c','e']
//
NVStrings* NVStrings::scatter( const char* str, const int* pos, unsigned int elements, bool bdevmem )
{
    unsigned int count = size();
    if( pos==nullptr )
        throw std::invalid_argument("parameter cannot be null");
    auto execpol = rmm::exec_policy(0);
    // copy string to device
    custring_view* d_repl = custring_from_host(str);
    const int* d_pos = pos;
    if( !bdevmem )
    {   // copy indexes to device memory
        d_pos = const_cast<const int*>(device_alloc<int>(elements,0));
        CUDA_TRY(cudaMemcpyAsync((void*)d_pos,pos,elements*sizeof(int),cudaMemcpyHostToDevice))
    }
    // create result output array
    rmm::device_vector<custring_view*> results(count,nullptr);
    auto d_results = results.data().get();
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::copy( execpol->on(0), d_strings, d_strings+count, d_results );
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), elements,
        [d_pos, count, d_repl, d_results] __device__ (unsigned int idx) {
            int pos = d_pos[idx];
            if( (pos >= 0) && (pos < count) )
                d_results[pos] = d_repl;
        });
    // build resulting instance
    NVStrings* rtn = new NVStrings(count);
    NVStrings_init_from_custrings(rtn->pImpl, d_results, count);
    if( !bdevmem )
        RMM_FREE((void*)d_pos,0);
    RMM_FREE((void*)d_repl,0);
    return rtn;
}

NVStrings* NVStrings::sublist( unsigned int start, unsigned int end, int step )
{
    unsigned int count = size();
    if( end > count )
        end = count;
    if( start > count )
        start = count;
    if( step==0 )
        step = 1;
    if( start == end )
        return new NVStrings(0);
    if( ((step > 0) && (start > end)) ||
        ((step < 0) && (start < end)) )
        return new NVStrings(0);
    unsigned int elems = (unsigned int)std::abs((int)(end-start));
    unsigned int abs_step = (unsigned int)std::abs(step);
    elems = (elems + abs_step -1)/abs_step; // adjust for steps
    auto execpol = rmm::exec_policy(0);
    rmm::device_vector<int> indexes(elems);
    thrust::sequence(execpol->on(0),indexes.begin(),indexes.end(),(int)start,step);
    return gather(indexes.data().get(),elems,true);
}

// remove the specified strings and return a new instance
NVStrings* NVStrings::remove_strings( const int* pos, unsigned int elements, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 )
        return new NVStrings(0);
    if( elements==0 || pos==0 )
        return copy();

    auto execpol = rmm::exec_policy(0);
    int* dpos = device_alloc<int>(elements,0);
    if( bdevmem )
       CUDA_TRY( cudaMemcpyAsync((void*)dpos,pos,elements*sizeof(unsigned int),cudaMemcpyDeviceToDevice))
    else
       CUDA_TRY( cudaMemcpyAsync((void*)dpos,pos,elements*sizeof(unsigned int),cudaMemcpyHostToDevice))
    // sort the position values
    thrust::sort(execpol->on(0),dpos,dpos+elements,thrust::greater<int>());
    // also should remove duplicates
    int* nend = thrust::unique(execpol->on(0),dpos,dpos+elements,thrust::equal_to<int>());
    elements = (unsigned int)(nend - dpos);
    if( count < elements )
    {
        RMM_FREE(dpos,0);
        fprintf(stderr,"remove_strings: more positions (%u) specified than the number of strings (%u)\n",elements,count);
        return nullptr;
    }

    // build array to hold positions which are not to be removed by marking deleted positions with -1
    rmm::device_vector<int> dnpos(count);
    thrust::sequence(execpol->on(0),dnpos.begin(),dnpos.end());
    int* d_npos = dnpos.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), elements,
        [dpos, d_npos, count] __device__ (unsigned int idx) {
            int pos = dpos[idx];
            if( (pos >= 0) && (pos < count) )
                d_npos[pos] = -1;
        });

    // now remove the positions marked with -1
    int* dend = thrust::remove_if(execpol->on(0),d_npos,d_npos+count,[] __device__ (int val) { return val < 0; });
    unsigned int new_count = (unsigned int)(dend-d_npos);
    // gather string pointers based on indexes in dnpos (new-positions)
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<custring_view*> results(new_count,nullptr);
    custring_view_array d_results = results.data().get();
    thrust::gather(execpol->on(0),d_npos,d_npos+new_count,d_strings,d_results);

    // create output object from results pointers
    NVStrings* rtn = new NVStrings(new_count);
    NVStrings_init_from_custrings(rtn->pImpl, d_results, new_count);
    RMM_FREE(dpos,0);
    return rtn;
}


// this sorts the strings into a new instance;
// a sorted strings list can improve performance by reducing divergence
NVStrings* NVStrings::sort( sorttype stype, bool ascending, bool nullfirst )
{
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    auto execpol = rmm::exec_policy(0);

    // copy the pointers so we can sort them
    rmm::device_vector<custring_view*> results(count,nullptr);
    custring_view_array d_results = results.data().get();
    thrust::copy( execpol->on(0), d_strings, d_strings+count, d_results );
    thrust::sort(execpol->on(0), d_results, d_results+count,
        [stype, ascending, nullfirst] __device__( custring_view*& lhs, custring_view*& rhs ) {
            if( lhs==0 || rhs==0 )
                return (nullfirst ? rhs!=0 : lhs!=0); // null < non-null
            // allow sorting by name and length
            int diff = 0;
            if( stype & NVStrings::length )
                diff = lhs->size() - rhs->size();
            if( diff==0 && (stype & NVStrings::name) )
                diff = lhs->compare(*rhs);
            return (ascending ? (diff < 0) : (diff > 0));
        });

    // build new instance from the sorted pointers
    NVStrings* rtn = new NVStrings(count);
    NVStrings_init_from_custrings( rtn->pImpl, d_results, count );
    return rtn;
}

// just provide the index order and leave the strings intact
int NVStrings::order( sorttype stype, bool ascending, unsigned int* indexes, bool nullfirst, bool todevice )
{
    unsigned int count = size();
    unsigned int* d_indexes = indexes;
    auto execpol = rmm::exec_policy(0);
    if( !todevice )
        d_indexes = device_alloc<unsigned int>(count,0);
    thrust::sequence(execpol->on(0), d_indexes, d_indexes+count);
    //
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::sort(execpol->on(0), d_indexes, d_indexes+count,
        [d_strings, stype, ascending, nullfirst] __device__( unsigned int& lidx, unsigned int& ridx ) {
            custring_view* lhs = d_strings[lidx];
            custring_view* rhs = d_strings[ridx];
            if( lhs==0 || rhs==0 )
                return (nullfirst ? rhs!=0 : lhs!=0);
            // allow sorting by name and length
            int diff = 0;
            if( stype & NVStrings::length )
                diff = lhs->size() - rhs->size();
            if( diff==0 && (stype & NVStrings::name) )
                diff = lhs->compare(*rhs);
            return (ascending ? (diff < 0) : (diff > 0));
        });
    //
    if( !todevice )
    {
        CUDA_TRY(cudaMemcpyAsync(indexes,d_indexes,count*sizeof(unsigned int),cudaMemcpyDeviceToHost))
        RMM_FREE(d_indexes,0);
    }
    return 0;
}
