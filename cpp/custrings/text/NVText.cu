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
#include <thrust/reduce.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include "NVStrings.h"
#include "NVText.h"

#include "../custring_view.cuh"
#include "../util.h"


// return boolean value for each token if found in the provided strings
unsigned int NVText::contains_strings( NVStrings& strs, NVStrings& tkns, bool* results, bool todevice )
{
    unsigned int count = strs.size();
    unsigned int tcount = tkns.size();
    if( results==0 || count==0 || tcount==0 )
        return 0;
    //
    auto execpol = rmm::exec_policy(0);
    bool* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<bool>(tcount*count,0);

    //
    rmm::device_vector<custring_view*> strings(count,nullptr);
    rmm::device_vector<custring_view*> tokens(tcount,nullptr);
    custring_view** d_strings = strings.data().get();
    custring_view** d_tokens = tokens.data().get();
    strs.create_custring_index(d_strings);
    tkns.create_custring_index(d_tokens);

    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_tokens, tcount, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            for( int jdx=0; jdx < tcount; ++jdx )
            {
                custring_view* dtgt = d_tokens[jdx];
                d_rtn[(idx*tcount)+jdx] = ((dstr && dtgt) ? dstr->find(*dtgt) : -2) >=0 ;
            }
        });
    //
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count*tcount,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return 0;
}

// return the number of occurrences of each string within a set of strings
// this will fill in the provided memory as a matrix:
//           'aa'  'bbb'  'c' ...
// "aaaabc"    2     0     1
// "aabbcc"    1     0     2
// "abbbbc"    0     1     1
// ...
unsigned int NVText::strings_counts( NVStrings& strs, NVStrings& tkns, unsigned int* results, bool todevice )
{
    unsigned int count = strs.size();
    unsigned int tcount = tkns.size();
    if( results==0 || count==0 || tcount==0 )
        return 0;
    //
    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<unsigned int>(tcount*count,0);

    //
    rmm::device_vector<custring_view*> strings(count,nullptr);
    rmm::device_vector<custring_view*> tokens(tcount,nullptr);
    custring_view** d_strings = strings.data().get();
    custring_view** d_tokens = tokens.data().get();
    strs.create_custring_index(d_strings);
    tkns.create_custring_index(d_tokens);

    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_tokens, tcount, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            for( int jdx=0; jdx < tcount; ++jdx )
            {
                custring_view* dtgt = d_tokens[jdx];
                int fnd = 0;
                if( dstr && dtgt )
                {
                    int pos = dstr->find(*dtgt);
                    while( pos >= 0 )
                    {
                        pos = dstr->find(*dtgt,pos+dtgt->chars_count());
                        ++fnd;
                    }
                }
                d_rtn[(idx*tcount)+jdx] = fnd;
            }
        });
    //
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(unsigned int)*count*tcount,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return 0;
}

//
NVStrings* NVText::scatter_count( NVStrings& strs, unsigned int* counts, bool bdevmem )
{
    unsigned int count = strs.size();
    if( count==0 || counts==nullptr )
        return nullptr;

    //
    auto execpol = rmm::exec_policy(0);
    unsigned int* d_counts = counts;
    if( !bdevmem )
    {
        d_counts = device_alloc<unsigned int>(count,0);
        cudaMemcpyAsync( d_counts, counts, count*sizeof(unsigned int), cudaMemcpyHostToDevice);
    }

    //
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);

    // determine the size of the NVStrings instance we will create
    unsigned int total_count = thrust::reduce(execpol->on(0), d_counts, d_counts+count );
    // create array of result pointers
    rmm::device_vector< thrust::pair<const char*,size_t> > results(total_count);
    thrust::pair<const char*,size_t>* d_results = results.data().get();
    // offsets point to each individual range
    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan( execpol->on(0), d_counts, d_counts+count, offsets.begin() );
    // now set the ranges
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_counts, d_offsets, d_results] __device__(unsigned int idx){
            thrust::pair<const char*,size_t> dnewstr{nullptr,0};
            custring_view* dstr = d_strings[idx];
            if( dstr )
                dnewstr = {dstr->data(),dstr->size()};
            auto dstr_range = d_results + d_offsets[idx];
            unsigned int dcount = d_counts[idx];
            for( int ridx=0; ridx < dcount; ++ridx )
                dstr_range[ridx] = dnewstr;
        });
    //
    if( !bdevmem )
        RMM_FREE(d_counts,0);
    // build strings object from elements
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_results,total_count);
}
