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
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include "NVStrings.h"
#include "NVText.h"

#include "../custring_view.cuh"
#include "../util.h"

//
NVStrings* NVText::create_ngrams(NVStrings& strs, unsigned int ngrams, const char* separator )
{
    if( ngrams==0 )
        ngrams = 2;
    if( separator==nullptr )
        separator = "";
    unsigned int count = strs.size();
    if( count==0 )
        return strs.copy();

    auto execpol = rmm::exec_policy(0);
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);

    // first let's remove any nulls or empty strings
    auto end = thrust::remove_if(execpol->on(0), d_strings, d_strings + count,
        [] __device__ ( custring_view* ds ) { return (ds==nullptr) || ds->empty(); } );
    count = (unsigned int)(end - d_strings); // new count
    if( count <= ngrams )
        return strs.join(separator,""); // this not quite right if there are nulls we removed
    if( ngrams==1 )
        return strs.copy(); // same with this one; need method to create NVStrings from custring_views

    custring_view* d_separator = custring_from_host(separator);

    // compute size of new strings
    unsigned int ngrams_count = count - ngrams +1;
    rmm::device_vector<size_t> sizes(ngrams_count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), ngrams_count,
        [d_strings, ngrams, d_separator, d_sizes] __device__(unsigned int idx) {
            size_t size = 0;
            for( unsigned int n=0; n < ngrams; ++n )
            {
                custring_view* dstr = d_strings[n+idx];
                size += dstr->size();
                if( (n+1) < ngrams )
                    size += d_separator->size();
            }
            d_sizes[idx] = size;
        });

    size_t bufsize = thrust::reduce(execpol->on(0), d_sizes, d_sizes+ngrams_count );
    rmm::device_vector<char> buffer(bufsize);
    char* d_buffer = buffer.data().get();
    rmm::device_vector<size_t> offsets(ngrams_count,0);
    thrust::exclusive_scan( execpol->on(0), sizes.begin(), sizes.end(), offsets.begin() );
    size_t* d_offsets = offsets.data().get();
    // build the memory and a list of pointers
    rmm::device_vector< thrust::pair<const char*,size_t> > results(ngrams_count);
    thrust::pair<const char*,size_t>* d_results = results.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), ngrams_count,
        [d_strings, d_separator, ngrams, d_offsets, d_buffer, d_results] __device__(unsigned int idx) {
            char* buffer = d_buffer + d_offsets[idx];
            char* sptr = buffer;
            size_t length = 0;
            for( unsigned int n=0; n < ngrams; ++n )
            {
                custring_view* dstr = d_strings[n+idx];
                unsigned int bytes = dstr->size();
                length += bytes;
                memcpy( sptr, dstr->data(), bytes );
                sptr += bytes;
                if( (n+1) >= ngrams )
                    continue;
                bytes = d_separator->size();
                length += bytes;
                memcpy( sptr, d_separator->data(), bytes );
                sptr += bytes;
            }
            d_results[idx].first = buffer;
            d_results[idx].second = length;
        });
    //
    RMM_FREE(d_separator,0);
    // build strings object from results elements
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_results,ngrams_count);
}

