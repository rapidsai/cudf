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
#include <sstream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/error_utils.hpp>

#include "nvstrings/NVStrings.h"

#include "./NVStringsImpl.h"
#include "../custring_view.cuh"
#include "../regex/regex.cuh"
#include "../unicode/is_flags.h"
#include "../util.h"

// This functor does both contains and match to minimize the number
// of regex calls to find() to be inlined.
template<size_t stack_size>
struct contains_fn
{
    dreprog* prog;
    custring_view_array d_strings;
    bool* d_rtn;
    bool bmatch{false};
    __device__ void operator()(unsigned int idx)
    {
        u_char data1[stack_size], data2[stack_size];
        prog->set_stack_mem(data1,data2);
        custring_view* dstr = d_strings[idx];
        if( dstr )
        {
            int begin = 0, end = bmatch ? 1 : dstr->chars_count();
            d_rtn[idx] = prog->find(idx,dstr,begin,end);//prog->contains(idx,dstr)==1;
        }
        else
            d_rtn[idx] = false;
    }
};

// regex version of contains() above
int NVStrings::contains_re( const char* pattern, bool* results, bool todevice )
{
    if( pattern==0 || results==0 )
        return -1;
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    // compile regex into device object
    const char32_t* ptn32 = to_char32(pattern);
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags());
    delete ptn32;
    // allocate regex working memory if necessary
    int regex_insts = prog->inst_counts();
    if( regex_insts > MAX_STACK_INSTS )
    {
        if( !prog->alloc_relists(count) )
        {
            std::ostringstream message;
            message << "nvstrings::contains_re: number of instructions (" << prog->inst_counts() << ") ";
            message << "and number of strings (" << count << ") ";
            message << "exceeds available memory";
            dreprog::destroy(prog);
            throw std::invalid_argument(message.str());
        }
    }

    bool* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<bool>(count,0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    if( (regex_insts > MAX_STACK_INSTS) || (regex_insts <= 10) )
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            contains_fn<RX_STACK_SMALL>{prog, d_strings, d_rtn});
    else if( regex_insts <= 100 )
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            contains_fn<RX_STACK_MEDIUM>{prog, d_strings, d_rtn});
    else
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            contains_fn<RX_STACK_LARGE>{prog, d_strings, d_rtn});
    // count the number of successful finds
    int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    dreprog::destroy(prog);
    return matches;
}

// match is like contains() except the pattern must match the beginning of the string only
int NVStrings::match( const char* pattern, bool* results, bool bdevmem )
{
    if( pattern==0 || results==0 )
        return -1;
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    // compile regex into device object
    const char32_t* ptn32 = to_char32(pattern);
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags());
    delete ptn32;
    // allocate regex working memory if necessary
    int regex_insts = prog->inst_counts();
    if( regex_insts > MAX_STACK_INSTS )
    {
        if( !prog->alloc_relists(count) )
        {
            std::ostringstream message;
            message << "nvstrings::match: number of instructions (" << prog->inst_counts() << ") ";
            message << "and number of strings (" << count << ") ";
            message << "exceeds available memory";
            dreprog::destroy(prog);
            throw std::invalid_argument(message.str());
        }
    }

    bool* d_rtn = results;
    if( !bdevmem )
        d_rtn = device_alloc<bool>(count,0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    if( (regex_insts > MAX_STACK_INSTS) || (regex_insts <= 10) )
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            contains_fn<RX_STACK_SMALL>{prog, d_strings, d_rtn, true});
    else if( regex_insts <= 100 )
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            contains_fn<RX_STACK_MEDIUM>{prog, d_strings, d_rtn, true});
    else
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            contains_fn<RX_STACK_LARGE>{prog, d_strings, d_rtn, true});

    // count the number of successful finds
        int matches = thrust::count(execpol->on(0), d_rtn, d_rtn+count, true);
    if( !bdevmem )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    dreprog::destroy(prog);
    return matches;
}

// Perhaps this can be combined with the contains_fn too.
template<size_t stack_size>
struct count_fn
{
    dreprog* prog;
    custring_view_array d_strings;
    int* d_rtn;
    __device__ void operator()(unsigned int idx)
    {
        u_char data1[stack_size], data2[stack_size];
        prog->set_stack_mem(data1,data2);
        custring_view* dstr = d_strings[idx];
        int fnd = 0;
        if( dstr )
        {
            int nchars = (int)dstr->chars_count();
            int begin = 0;
            while( begin <= nchars )
            {
                int end = nchars;
                int result = prog->find(idx,dstr,begin,end);
                if(result<=0)
                    break;
                ++fnd;
                begin = end>begin ? end : begin + 1;
            }
        }
        d_rtn[idx] = fnd;
    }
};

// counts number of times the regex pattern matches a string within each string
int NVStrings::count_re( const char* pattern, int* results, bool todevice )
{
    if( pattern==0 || results==0 )
        return -1;
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    // compile regex into device object
    const char32_t* ptn32 = to_char32(pattern);
    dreprog* prog = dreprog::create_from(ptn32,get_unicode_flags());
    delete ptn32;
    // allocate regex working memory if necessary
    int regex_insts = prog->inst_counts();
    if( regex_insts > MAX_STACK_INSTS )
    {
        if( !prog->alloc_relists(count) )
        {
            std::ostringstream message;
            message << "nvstrings::count_re: number of instructions (" << prog->inst_counts() << ") ";
            message << "and number of strings (" << count << ") ";
            message << "exceeds available memory";
            dreprog::destroy(prog);
            throw std::invalid_argument(message.str());
        }
    }

    int* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<int>(count,0);

    custring_view_array d_strings = pImpl->getStringsPtr();
    if( (regex_insts > MAX_STACK_INSTS) || (regex_insts <= 10) )
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            count_fn<RX_STACK_SMALL>{prog, d_strings, d_rtn});
    else if( regex_insts <= 100 )
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            count_fn<RX_STACK_MEDIUM>{prog, d_strings, d_rtn});
    else
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            count_fn<RX_STACK_LARGE>{prog, d_strings, d_rtn});
    // count the number of successful finds
    int matches = (int)count - thrust::count(execpol->on(0), d_rtn, d_rtn+count, 0);
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    dreprog::destroy(prog);
    return matches;
}
