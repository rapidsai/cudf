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
#include <thrust/scan.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include <utilities/error_utils.hpp>
#include <cudf/cudf.h>

#include "nvstrings/NVStrings.h"
#include "nvstrings/NVText.h"

#include "../custring_view.cuh"
#include "../util.h"



// Documentation here: https://www.cuelogic.com/blog/the-levenshtein-algorithm
// And here: https://en.wikipedia.org/wiki/Levenshtein_distances
struct edit_distance_levenshtein_algorithm
{
    custring_view** d_strings; // trying match
    custring_view* d_tgt;      // match with this
    custring_view** d_tgts;    // or these
    short* d_buffer;           // compute buffer
    size_t* d_offsets;         // locate sub-buffer
    unsigned int* d_results;   // edit-distances

    // single string
    edit_distance_levenshtein_algorithm( custring_view** strings,
                                        custring_view* tgt,
                                        short* buffer,
                                        size_t* offsets,
                                        unsigned int* results )
                                      : d_strings(strings),
                                        d_tgt(tgt),
                                        d_tgts(0),
                                        d_buffer(buffer),
                                        d_offsets(offsets),
                                        d_results(results) {}

    // multiple strings
    edit_distance_levenshtein_algorithm( custring_view** strings,
                                        custring_view** tgts,
                                        short* buffer,
                                        size_t* offsets,
                                        unsigned int* results )
                                      : d_strings(strings),
                                        d_tgt(0),
                                        d_tgts(tgts),
                                        d_buffer(buffer),
                                        d_offsets(offsets),
                                        d_results(results) {}

    // pairwise strings
    edit_distance_levenshtein_algorithm( custring_view** strings,
                                        short* buffer,
                                        size_t* offsets,
                                        unsigned int* results )
                                      : d_strings(strings),
                                        d_tgt(0), d_tgts(0),
                                        d_buffer(buffer),
                                        d_offsets(offsets),
                                        d_results(results) {}

    __device__ void operator() (unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        short* buf = (short*)d_buffer + d_offsets[idx];
        custring_view* dtgt = d_tgt;
        if( !d_tgt)
            dtgt = d_tgts[idx];
        d_results[idx] = compute_distance(dstr,dtgt,buf);
    }

    __device__ unsigned int compute_distance( custring_view* dstr,
                                              custring_view* dtgt,
                                              short* buf )
    {
        if( !dstr || dstr->empty() )
            return dtgt ? dtgt->chars_count() : 0;
        if( !dtgt || dtgt->empty() )
            return dstr->chars_count();
        //
        custring_view* strA = dstr;
        custring_view* strB = dtgt;
        int lenA = (int)dstr->chars_count();
        int lenB = (int)dtgt->chars_count();
        if( lenA > lenB )
        {
            lenB = lenA;
            lenA = dtgt->chars_count();
            strA = dtgt;
            strB = dstr;
        }
        //
        short* line2 = buf;
        short* line1 = line2 + lenA;
        short* line0 = line1 + lenA;
        int range = lenA + lenB - 1;
        for (int i = 0; i < range; i++)
        {
            short* tmp = line2;
            line2 = line1;
            line1 = line0;
            line0 = tmp;

            for(int x = (i < lenB ? 0 : i - lenB + 1); (x < lenA) && (x < i+1); x++)
            {
                int y = i - x;
                short u = y > 0 ? line1[x] : x + 1;
                short v = x > 0 ? line1[x - 1] : y + 1;
                short w;
                if((x > 0) && (y > 0))
                    w = line2[x - 1];
                else if(x > y)
                    w = x;
                else
                    w = y;
                u++; v++;
                Char c1 = strA->at(x);
                Char c2 = strB->at(y);
                if(c1 != c2)
                    w++;
                short value = u;
                if(v < value)
                    value = v;
                if(w < value)
                    value = w;
                line0[x] = value;
            }
        }
        return (unsigned int)line0[lenA-1];
    }
};

struct edit_distance_matrix_levenshtein_algorithm :
                    edit_distance_levenshtein_algorithm {

    gdf_index_type count;   // size of nvstrings object

    // pair wise string distance computation
    edit_distance_matrix_levenshtein_algorithm( custring_view** strings,
                                               short* buffer,
                                               size_t* offsets,
                                               unsigned int* results,
                                               gdf_index_type count)
          : edit_distance_levenshtein_algorithm( strings,
                                                buffer,
                                                offsets,
                                                results),
                                                count(count) {}

    __device__ void operator() (gdf_index_type idx)
    {
      gdf_index_type row = idx / count;
      gdf_index_type col = idx % count;
      if( row > col )
         return;
      unsigned int dist = 0;  // diagonal
      if( row < col ){
         custring_view* d_strings1 = d_strings[row];
         custring_view* d_strings2 = d_strings[col];
         short* buf = (short*)d_buffer + d_offsets[idx - ((row+1)*(row+2))/2];
         dist = compute_distance(d_strings1, d_strings2, buf);
      }
      d_results[idx] = dist; // top half
      d_results[col * count + row] = dist; // bottom half
    }
};

unsigned int NVText::edit_distance( distance_type algorithm,
                                    NVStrings& strs,
                                    const char* target,
                                    unsigned int* results,
                                    bool results_is_device_memory )
{
    if( algorithm != levenshtein || target==0 || results==0 )
        throw std::invalid_argument("invalid algorithm");
    unsigned int count = strs.size();
    if( count==0 )
        return 0; // nothing to do
    auto execpol = rmm::exec_policy(0);
    custring_view* d_target = custring_from_host(target);

    // setup results vector
    unsigned int* d_return = results;
    if( !results_is_device_memory )
        d_return = device_alloc<unsigned int>(count,0);

    // get the string pointers
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);

    // calculate the size of the compute-buffer: 6 * length of string
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n( execpol->on(0),
        thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_target, d_sizes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int len = dstr->chars_count();
            if( d_target->chars_count() < len )
                len = d_target->chars_count();
            d_sizes[idx] = len * 3;
        });
    //
    size_t bufsize = thrust::reduce(execpol->on(0), d_sizes, d_sizes+count );
    rmm::device_vector<short> buffer(bufsize,0);
    short* d_buffer = buffer.data().get();
    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0), sizes.begin(),
                              sizes.end(), offsets.begin() );
    // compute edit distance
    thrust::for_each_n(execpol->on(0),
        thrust::make_counting_iterator<unsigned int>(0), count,
        edit_distance_levenshtein_algorithm(d_strings,
                                           d_target,
                                           d_buffer,
                                           d_offsets,
                                           d_return));
    //
    if( !results_is_device_memory )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_return,
                  count*sizeof(unsigned int),cudaMemcpyDeviceToHost))
        RMM_FREE(d_return,0);
    }
    RMM_FREE(d_target,0);
    return 0;
}

unsigned int NVText::edit_distance( distance_type algorithm,
                                    NVStrings& strs1,
                                    NVStrings& strs2,
                                    unsigned int* results,
                                    bool results_is_device_memory )
{
    if( algorithm != levenshtein )
        throw std::invalid_argument("invalid algorithm");
    unsigned int count = strs1.size();
    if( count != strs2.size() )
        throw std::invalid_argument("sizes must match");
    if( count==0 )
        return 0; // nothing to do

    // setup results vector
    auto execpol = rmm::exec_policy(0);
    unsigned int* d_return = results;
    if( !results_is_device_memory )
        d_return = device_alloc<unsigned int>(count,0);

    // get the string pointers
    rmm::device_vector<custring_view*> strings1(count,nullptr);
    custring_view** d_strings1 = strings1.data().get();
    strs1.create_custring_index(d_strings1);
    rmm::device_vector<custring_view*> strings2(count,nullptr);
    custring_view** d_strings2 = strings2.data().get();
    strs2.create_custring_index(d_strings2);

    // calculate the size of the compute-buffer: 6 * length of string
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0),
        thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings1, d_strings2, d_sizes] __device__(unsigned int idx){
            custring_view* dstr1 = d_strings1[idx];
            custring_view* dstr2 = d_strings2[idx];
            if( !dstr1 || !dstr2 )
                return;
            int len1 = dstr1->chars_count();
            int len = dstr2->chars_count();
            if( len1 < len )
                len = len1;
            d_sizes[idx] = len * 3;
        });
    //
    size_t bufsize = thrust::reduce(execpol->on(0), d_sizes, d_sizes+count );
    rmm::device_vector<short> buffer(bufsize,0);
    short* d_buffer = buffer.data().get();
    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0), sizes.begin(),
                              sizes.end(), offsets.begin() );
    // compute edit distance
    thrust::for_each_n(execpol->on(0),
        thrust::make_counting_iterator<unsigned int>(0), count,
        edit_distance_levenshtein_algorithm(d_strings1,
                                           d_strings2,
                                           d_buffer,
                                           d_offsets,
                                           d_return));
    //
    if( !results_is_device_memory )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_return,
                  count*sizeof(unsigned int),cudaMemcpyDeviceToHost))
        RMM_FREE(d_return,0);
    }
    return 0;
}

unsigned int NVText::edit_distance_matrix( distance_type algorithm,
                                           NVStrings& strs,
                                           unsigned int* results,
                                           bool results_is_device_memory )
{
    CUDF_EXPECTS(algorithm == levenshtein, "Unsupported algorithm");
    CUDF_EXPECTS(results != nullptr, "Null results pointer");

    gdf_index_type count = strs.size();
    if( count==0 )
        return 0; // nothing to do
    auto execpol = rmm::exec_policy(0);

    // setup results vector
    unsigned int* d_return = results;
    if( !results_is_device_memory )
        d_return = device_alloc<unsigned int>(count*count,0);

    // get the string pointers
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);

    // calculate the size of the compute-buffer: 6 * length of string
    // considering only the upper diagonal matrix: N*(N-1)/2 elements
    gdf_index_type n_upper = (count*(count-1))/2;
    rmm::device_vector<size_t> sizes(n_upper,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0),
        thrust::make_counting_iterator<gdf_index_type>(0), count*count,
        [d_strings, d_sizes, count] __device__(gdf_index_type idx){
            gdf_index_type row = idx / count;
            gdf_index_type col = idx % count;
            if( row >= col )
                return;
            custring_view* d_strings1 = d_strings[row];
            custring_view* d_strings2 = d_strings[col];

            if( !d_strings1 || !d_strings2)
                return;
            int len = d_strings1->chars_count();
            if( d_strings2->chars_count() < len )
                len = d_strings2->chars_count();
            d_sizes[idx - ((row+1)*(row+2))/2 ] = len * 3;
        });

    size_t bufsize = thrust::reduce(execpol->on(0), d_sizes, d_sizes+n_upper );
    rmm::device_vector<short> buffer(bufsize,0);
    short* d_buffer = buffer.data().get();
    rmm::device_vector<size_t> offsets(n_upper,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0), sizes.begin(),
                           sizes.end(), offsets.begin() );

    thrust::for_each_n(execpol->on(0),
            thrust::make_counting_iterator<gdf_index_type>(0), count*count,
            edit_distance_matrix_levenshtein_algorithm(d_strings,
                                                      d_buffer,
                                                      d_offsets,
                                                      d_return,
                                                      count));

    if( !results_is_device_memory )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_return,
                  count*count*sizeof(unsigned int),cudaMemcpyDeviceToHost))
        RMM_FREE(d_return,0);
    }
    return 0;
}