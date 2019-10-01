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
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/error_utils.hpp>

#include "nvstrings/NVStrings.h"

#include "./NVStringsImpl.h"
#include "../custring_view.cuh"
#include "../util.h"

// common token counter for all split methods
struct token_counter
{
    custring_view_array d_strings;
    char* d_delimiter;
    unsigned int dellen;
    int tokens;
    int* d_counts;
    //
    token_counter(custring_view_array dstrs, char* delim, unsigned int dlen, int t, int* counts)
    : d_strings(dstrs), d_delimiter(delim), dellen(dlen), tokens(t), d_counts(counts) {}
    __device__ void operator()(unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        if( dstr )
            d_counts[idx] = dstr->split_size(d_delimiter,dellen,0,tokens);
    }
};

// special-case token counter for whitespace delimiter
// leading and trailing and duplicate delimiters are ignored
struct whitespace_token_counter
{
    custring_view_array d_strings;
    int tokens;
    int* d_counts;

    // count the 'words' only between non-whitespace characters
    whitespace_token_counter(custring_view_array dstrs, int t, int* counts)
    : d_strings(dstrs), tokens(t), d_counts(counts) {}
    __device__ void operator()(unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        if( !dstr )
            return;
        int dcount = 0;
        bool spaces = true;
        custring_view::iterator itr = dstr->begin();
        while( itr != dstr->end() )
        {
            Char ch = *itr;
            if( spaces == (ch <= ' ') )
                itr++;
            else
            {
                dcount += (int)spaces;
                spaces = !spaces;
            }
        }
        if( tokens && (dcount > tokens) )
            dcount = tokens;
        if( dcount==0 )
            dcount = 1; // always allow empty string
        d_counts[idx] = dcount;
        //printf("dcount=%d\n",dcount);
    }
};

//
// Coded form Pandas split algorithm as documented here:
// https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.split.html#pandas.Series.str.split
//
// Example:
//
//   import pandas as pd
//   pd_series = pd.Series(['', None, 'a_b', '_a_b_', '__aa__bb__', '_a__bbb___c', '_aa_b__ccc__'])
//   print(pd_series.str.split(pat='_', expand=False))
//      0                      []
//      1                    None
//      2                  [a, b]
//      3              [, a, b, ]
//      4      [, , aa, , bb, , ]
//      5     [, a, , bbb, , , c]
//      6    [, aa, b, , ccc, , ]
//
//   print(pd_series.str.split(pat='_', n=1, expand=False))
//      0                 []
//      1               None
//      2             [a, b]
//      3           [, a_b_]
//      4      [, _aa__bb__]
//      5     [, a__bbb___c]
//      6    [, aa_b__ccc__]
//
//   print(pd_series.str.split(pat='_', n=2, expand=False))
//      0                  []
//      1                None
//      2              [a, b]
//      3           [, a, b_]
//      4      [, , aa__bb__]
//      5     [, a, _bbb___c]
//      6    [, aa, b__ccc__]
//
//
int NVStrings::split_record( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results)
{
    if( delimiter==0 )
        return split_record(maxsplit,results);

    auto execpol = rmm::exec_policy(0);
    unsigned int dellen = (unsigned int)strlen(delimiter);
    char* d_delimiter = device_alloc<char>(dellen+1,0);
    CUDA_TRY( cudaMemcpyAsync(d_delimiter,delimiter,dellen+1,cudaMemcpyHostToDevice))
    int tokens = 0;
    if( maxsplit > 0 )
        tokens = maxsplit + 1; // makes consistent with Pandas

    // need to count how many output strings per string
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            token_counter(d_strings,d_delimiter,dellen,tokens,d_counts));

    // build int arrays to hold each string's split size
    int totalSizes = thrust::reduce(execpol->on(0), counts.begin(), counts.end());
    rmm::device_vector<int> sizes(totalSizes,0), offsets(count,0), totals(count,0);
    thrust::exclusive_scan(execpol->on(0),counts.begin(),counts.end(),offsets.begin());
    int* d_offsets = offsets.data().get();
    int* d_sizes = sizes.data().get();
    int* d_totals = totals.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, dellen, d_counts, d_offsets, d_sizes, d_totals] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int* dsizes = d_sizes + d_offsets[idx];
            int dcount = d_counts[idx];
            d_totals[idx] = dstr->split_size(d_delimiter,dellen,dsizes,dcount);
        });
    //
    cudaDeviceSynchronize();

    // now build an array of custring_views* arrays for each value
    int totalNewStrings = 0;
    thrust::host_vector<int> h_counts(counts);
    thrust::host_vector<int> h_totals(totals);
    thrust::host_vector<custring_view_array> h_splits(count,nullptr);
    thrust::host_vector<char*> h_buffers(count,nullptr);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        int splitCount = h_counts[idx];
        if( splitCount==0 )
        {
            results.push_back(0);
            continue;
        }

        NVStrings* splitResult = new NVStrings(splitCount);
        results.push_back(splitResult);
        h_splits[idx] = splitResult->pImpl->getStringsPtr();

        int totalSize = h_totals[idx];
        char* d_buffer = device_alloc<char>(totalSize,0);
        splitResult->pImpl->setMemoryBuffer(d_buffer,totalSize);
        h_buffers[idx] = d_buffer;

        totalNewStrings += splitCount;
    }

    //
    rmm::device_vector<custring_view_array> splits(h_splits);
    custring_view_array* d_splits = splits.data().get();
    rmm::device_vector<char*> buffers(h_buffers);
    char** d_buffers = buffers.data().get();

    // do the splits and fill in the arrays
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, dellen, d_counts, d_buffers, d_sizes, d_offsets, d_splits] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int d_count = d_counts[idx];
            if( d_count < 1 )
                return;
            char* buffer = (char*)d_buffers[idx];
            int* dsizes = d_sizes + d_offsets[idx];
            custring_view_array d_strs = d_splits[idx];
            for( int i=0; i < d_count; ++i )
            {
                int size = ALIGN_SIZE(dsizes[i]);
                d_strs[i] = (custring_view*)buffer;
                buffer += size;
            }
            dstr->split(d_delimiter,dellen,d_count,d_strs);
        });

    //
    printCudaError(cudaDeviceSynchronize(),"nvs-split_record");
    RMM_FREE(d_delimiter,0);
    return totalNewStrings;
}

//
// Whitespace delimiter algorithm is very different.
// It follows the Python str.split algorithm as defined in Pandas: https://docs.python.org/3/library/stdtypes.html#str.split
// Paraphrased as follows (for null delimiter):
//   Runs of consecutive whitespace are regarded as a single separator,
//   and the result will contain no empty strings at the start orend if
//   the string has leading or trailing whitespace.
// Also whitespace is not just space.
// The algorithm below uses the shortcut (<=' ') to catch \t\r\n or any other control character.
// The above statement does not account for maxplit as seen in the following examples where n=maxpslit.
//
//  import pandas as pd
//  pd_series = pd.Series(['', None, 'a b', ' a b ', '  aa  bb  ', ' a  bbb   c', ' aa b  ccc  '])
//  print(pd_series.str.split(pat=None, expand=False))
//      0              []
//      1            None
//      2          [a, b]
//      3          [a, b]
//      4        [aa, bb]
//      5     [a, bbb, c]
//      6    [aa, b, ccc]
//
//  print(pd_series.str.split(pat=None, n=1, expand=False))
//      0                []
//      1              None
//      2            [a, b]
//      3           [a, b ]
//      4        [aa, bb  ]
//      5      [a, bbb   c]
//      6    [aa, b  ccc  ]
//
//  print(pd_series.str.split(pat=None, n=2, expand=False))
//      0                []
//      1              None
//      2            [a, b]
//      3            [a, b]
//      4          [aa, bb]
//      5       [a, bbb, c]
//      6    [aa, b, ccc  ]
//
// Note:
// - lack of empty strings
// - trailing and leading characters are ignored (sometimes)
// - multiple whitespace characters are ignored (sometimes)
//
int NVStrings::split_record( int maxsplit, std::vector<NVStrings*>& results)
{
    auto execpol = rmm::exec_policy(0);
    int tokens = 0;
    if( maxsplit > 0 )
        tokens = maxsplit + 1; // makes consistent with Pandas

    // need to count how many output strings per string
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            whitespace_token_counter(d_strings,tokens,d_counts));
    //cudaDeviceSynchronize();

    // build int arrays to hold each string's split size
    int totalSizes = thrust::reduce(execpol->on(0), counts.begin(), counts.end());
    rmm::device_vector<int> sizes(totalSizes,0), offsets(count,0), totals(count,0);
    thrust::exclusive_scan(execpol->on(0),counts.begin(),counts.end(),offsets.begin());
    int* d_offsets = offsets.data().get();
    int* d_sizes = sizes.data().get();
    int* d_totals = totals.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, tokens, d_counts, d_offsets, d_sizes, d_totals] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return; // null string
            int* dsizes = d_sizes + d_offsets[idx];
            int dcount = d_counts[idx];
            int bytes = 0, sidx = 0, spos = 0, nchars = dstr->chars_count();
            //printf("tokens=%d,dcount=%d,nchars=%d\n",tokens,dcount,nchars);
            bool spaces = true;
            for( int pos=0; (pos < nchars) && (sidx < dcount); ++pos )
            {
                Char ch = dstr->at(pos);
                if( spaces == (ch <= ' ') )
                {
                    if( spaces )
                        spos = pos+1;
                    continue;
                }
                if( !spaces )
                {
                    if( (sidx+1)==tokens )
                        break;
                    int size = dstr->substr_size(spos,pos-spos);
                    dsizes[sidx++] = size;
                    //printf("%d:pos=%d,spos=%d,size=%d\n",(sidx-1),pos,spos,size);
                    bytes += ALIGN_SIZE(size);
                    spos = pos + 1;
                }
                spaces = !spaces;
            }
            if( sidx < dcount )
            {
                int size = 0;
                if( spos < nchars )
                    size = dstr->substr_size(spos,nchars-spos);
                else
                    size = (int)custring_view::alloc_size((unsigned)0,(unsigned)0);
                dsizes[sidx] = size;
                //printf("spos=%d,nchars=%d,size=%d\n",spos,nchars,size);
                bytes += ALIGN_SIZE(size);
            }
            //printf("bytes=%d\n",bytes);
            d_totals[idx] = bytes;
        });

    //
    cudaDeviceSynchronize();

    // now build an array of custring_views* arrays for each value
    int totalNewStrings = 0;
    thrust::host_vector<int> h_counts(counts);
    thrust::host_vector<int> h_totals(totals);
    thrust::host_vector<custring_view_array> h_splits(count,nullptr);
    thrust::host_vector<char*> h_buffers(count,nullptr);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        int splitCount = h_counts[idx];
        if( splitCount==0 )
        {
            results.push_back(0);
            continue;
        }

        NVStrings* splitResult = new NVStrings(splitCount);
        results.push_back(splitResult);
        h_splits[idx] = splitResult->pImpl->getStringsPtr();

        int totalSize = h_totals[idx];
        char* d_buffer = device_alloc<char>(totalSize,0);
        splitResult->pImpl->setMemoryBuffer(d_buffer,totalSize);
        h_buffers[idx] = d_buffer;

        totalNewStrings += splitCount;
    }

    //
    rmm::device_vector<custring_view_array> splits(h_splits);
    custring_view_array* d_splits = splits.data().get();
    rmm::device_vector<char*> buffers(h_buffers);
    char** d_buffers = buffers.data().get();

    // do the splits and fill in the arrays
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, tokens, d_counts, d_buffers, d_sizes, d_offsets, d_splits] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return; // null string
            int dcount = d_counts[idx];
            char* buffer = (char*)d_buffers[idx];
            int* dsizes = d_sizes + d_offsets[idx];
            custring_view_array d_strs = d_splits[idx];
            int emptysize = (int)custring_view::alloc_size((unsigned)0,(unsigned)0);
            if( dcount==0 || dsizes[0]==emptysize )
            {
                d_strs[0] = custring_view::create_from(buffer,buffer,0);
                return; // empty string
            }
            for( int i=0; i < dcount; ++i )
            {
                int size = ALIGN_SIZE(dsizes[i]);
                d_strs[i] = (custring_view*)buffer;
                buffer += size;
            }
            int sidx = 0, spos = 0, nchars = dstr->chars_count();
            //printf(">tokens=%d,dcount=%d,nchars=%d",tokens,dcount,nchars);
            bool spaces = true;
            for( int pos=0; (pos < nchars) && (sidx < dcount); ++pos )
            {
                Char ch = dstr->at(pos);
                if( spaces == (ch <= ' ') )
                {
                    if( spaces )
                        spos = pos+1;
                    continue;
                }
                if( !spaces )
                {
                    if( (sidx+1)==tokens )
                        break;
                    d_strs[sidx] = dstr->substr(spos,pos-spos,1,(void*)d_strs[sidx]);
                    //printf(">%d:pos=%d,spos=%d\n",sidx,pos,spos);
                    ++sidx;
                    spos = pos + 1;
                }
                spaces = !spaces;
            }
            if( (sidx < dcount) && (spos < nchars) )
            {
                d_strs[sidx] = dstr->substr(spos,nchars-spos,1,(void*)d_strs[sidx]);
                //printf(">%d:spos=%d,nchars=%d\n",sidx,spos,nchars);
            }
        });

    //
    printCudaError(cudaDeviceSynchronize(),"nvs-split_record_ws");
    return totalNewStrings;
}

//
// This is just the split-from-the-right version of above.
//
int NVStrings::rsplit_record( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results)
{
    if( delimiter==0 )
        return rsplit_record(maxsplit,results);

    auto execpol = rmm::exec_policy(0);
    unsigned int dellen = (unsigned int)strlen(delimiter);
    char* d_delimiter = device_alloc<char>(dellen+1,0);
    CUDA_TRY( cudaMemcpyAsync(d_delimiter,delimiter,dellen+1,cudaMemcpyHostToDevice))
    int tokens = 0;
    if( maxsplit > 0 )
        tokens = maxsplit + 1; // makes consistent with Pandas

    // need to count how many output strings per string
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            token_counter(d_strings,d_delimiter,dellen,tokens,d_counts));

    // build int arrays to hold each string's split size
    int totalSizes = thrust::reduce(execpol->on(0), counts.begin(), counts.end());
    rmm::device_vector<int> sizes(totalSizes,0), offsets(count,0), totals(count,0);
    thrust::exclusive_scan(execpol->on(0),counts.begin(),counts.end(),offsets.begin());
    int* d_offsets = offsets.data().get();
    int* d_sizes = sizes.data().get();
    int* d_totals = totals.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, dellen, d_counts, d_offsets, d_sizes, d_totals] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int dcount = d_counts[idx];
            int* dsizes = d_sizes + d_offsets[idx];
            d_totals[idx] = dstr->rsplit_size(d_delimiter,dellen,dsizes,dcount);
        });
    cudaDeviceSynchronize();

    // now build an array of custring_views* arrays for each value
    int totalNewStrings = 0;
    thrust::host_vector<int> h_counts(counts);
    thrust::host_vector<int> h_totals(totals);
    thrust::host_vector<custring_view_array> h_splits(count,nullptr);
    thrust::host_vector<char*> h_buffers(count,nullptr);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        int splitCount = h_counts[idx];
        if( splitCount==0 )
        {
            results.push_back(0);
            continue;
        }
        NVStrings* splitResult = new NVStrings(splitCount);
        results.push_back(splitResult);
        h_splits[idx] = splitResult->pImpl->getStringsPtr();

        int totalSize = h_totals[idx];
        char* d_buffer = device_alloc<char>(totalSize,0);
        splitResult->pImpl->setMemoryBuffer(d_buffer,totalSize);
        h_buffers[idx] = d_buffer;

        totalNewStrings += splitCount;
    }

    //
    rmm::device_vector<custring_view_array> splits(h_splits);
    custring_view_array* d_splits = splits.data().get();
    rmm::device_vector<char*> buffers(h_buffers);
    char** d_buffers = buffers.data().get();

    // do the splits and fill in the arrays
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, dellen, d_counts, d_buffers, d_sizes, d_offsets, d_splits] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int d_count = d_counts[idx];
            if( d_count < 1 )
                return;
            char* buffer = (char*)d_buffers[idx];
            int* dsizes = d_sizes + d_offsets[idx];
            custring_view_array d_strs = d_splits[idx];
            for( int i=0; i < d_count; ++i )
            {
                d_strs[i] = (custring_view*)buffer;
                int size = ALIGN_SIZE(dsizes[i]);
                buffer += size;
                //printf("%d:%d=%d\n",(int)idx,i,size);
            }
            dstr->rsplit(d_delimiter,dellen,d_count,d_strs);
        });

    //
    printCudaError(cudaDeviceSynchronize(),"nvs-rsplit_record");
    RMM_FREE(d_delimiter,0);
    return totalNewStrings;
}

//
// And the whitespace-delimited version of rsplit_record
//
int NVStrings::rsplit_record( int maxsplit, std::vector<NVStrings*>& results)
{
    auto execpol = rmm::exec_policy(0);
    int tokens = 0;
    if( maxsplit > 0 )
        tokens = maxsplit + 1; // makes consistent with Pandas

    // need to count how many output strings per string
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            whitespace_token_counter(d_strings,tokens,d_counts));

    // build int arrays to hold each string's split size
    int totalSizes = thrust::reduce(execpol->on(0), counts.begin(), counts.end());
    rmm::device_vector<int> sizes(totalSizes,0), offsets(count,0), totals(count,0);
    thrust::exclusive_scan(execpol->on(0),counts.begin(),counts.end(),offsets.begin());
    int* d_offsets = offsets.data().get();
    int* d_sizes = sizes.data().get();
    int* d_totals = totals.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, tokens, d_counts, d_offsets, d_sizes, d_totals] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int* dsizes = d_sizes + d_offsets[idx];
            int dcount = d_counts[idx];
            int sidx = (dcount-1), nchars = dstr->chars_count();
            int bytes = 0, epos = nchars;
            //printf("tokens=%d,dcount=%d,nchars=%d\n",tokens,dcount,nchars);
            bool spaces = true;
            for( int pos=nchars; (pos>0) && (sidx>=0); --pos )
            {
                Char ch = dstr->at(pos-1);
                if( spaces == (ch <= ' ') )
                {
                    if( spaces )
                        epos = pos-1;
                    continue;
                }
                if( !spaces )
                {
                    if( (dcount-sidx)==tokens )
                        break;
                    int size = dstr->substr_size(pos,epos-pos);
                    dsizes[sidx--] = size;
                    //printf("%d:pos=%d,epos=%d,size=%d\n",(sidx+1),pos,epos,size);
                    bytes += ALIGN_SIZE(size);
                    epos = pos-1;
                }
                spaces = !spaces;
            }
            if( sidx==0 )
            {
                int size = 0;
                if( epos > 0 )
                    size = dstr->substr_size(0,epos);
                else
                    size = (int)custring_view::alloc_size((unsigned)0,(unsigned)0);
                //printf("%d:epos=%d,size=%d\n",sidx,epos,size);
                dsizes[sidx] = size;
                bytes += ALIGN_SIZE(size);
            }
            //printf("bytes=%d\n",bytes);
            d_totals[idx] = bytes;
        });

    cudaDeviceSynchronize();

    // now build an array of custring_views* arrays for each value
    int totalNewStrings = 0;
    thrust::host_vector<int> h_counts(counts);
    thrust::host_vector<int> h_totals(totals);
    thrust::host_vector<custring_view_array> h_splits(count,nullptr);
    thrust::host_vector<char*> h_buffers(count,nullptr);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        int splitCount = h_counts[idx];
        if( splitCount==0 )
        {
            results.push_back(0);
            continue;
        }
        NVStrings* splitResult = new NVStrings(splitCount);
        results.push_back(splitResult);
        h_splits[idx] = splitResult->pImpl->getStringsPtr();

        int totalSize = h_totals[idx];
        char* d_buffer = device_alloc<char>(totalSize,0);
        splitResult->pImpl->setMemoryBuffer(d_buffer,totalSize);
        h_buffers[idx] = d_buffer;

        totalNewStrings += splitCount;
    }

    //
    rmm::device_vector<custring_view_array> splits(h_splits);
    custring_view_array* d_splits = splits.data().get();
    rmm::device_vector<char*> buffers(h_buffers);
    char** d_buffers = buffers.data().get();

    // do the splits and fill in the arrays
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, tokens, d_counts, d_buffers, d_sizes, d_offsets, d_splits] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int dcount = d_counts[idx];
            char* buffer = (char*)d_buffers[idx];
            int* dsizes = d_sizes + d_offsets[idx];
            custring_view_array d_strs = d_splits[idx];
            int emptysize = (int)custring_view::alloc_size((unsigned)0,(unsigned)0);
            if( dcount==0 || dsizes[0]==emptysize )
            {
                d_strs[0] = custring_view::create_from(buffer,buffer,0);
                return; // empty string
            }
            for( int i=0; i < dcount; ++i )
            {
                int size = ALIGN_SIZE(dsizes[i]);
                d_strs[i] = (custring_view*)buffer;
                buffer += size;
            }
            int sidx = (dcount-1), nchars = dstr->chars_count();
            int epos = nchars;
            //printf(">tokens=%d,dcount=%d,nchars=%d\n",tokens,dcount,nchars);
            bool spaces = true;
            for( int pos=nchars; (pos > 0) && (sidx >= 0); --pos )
            {
                Char ch = dstr->at(pos-1);
                if( spaces == (ch <= ' ') )
                {
                    if( spaces )
                        epos = pos-1;
                    continue;
                }
                if( !spaces )
                {
                    if( (dcount-sidx)==tokens )
                        break;
                    d_strs[sidx] = dstr->substr(pos,epos-pos,1,(void*)d_strs[sidx]);
                    //printf(">%d:pos=%d,epos=%d\n",sidx,pos,epos);
                    --sidx;
                    epos = pos-1;
                }
                spaces = !spaces;
            }
            if( (sidx>=0) && (epos > 0) )
            {
                d_strs[sidx] = dstr->substr(0,epos,1,(void*)d_strs[sidx]);
                //printf(">%d:epos=%d\n",sidx,epos);
            }
        });
    //
    printCudaError(cudaDeviceSynchronize(),"nvs-rsplit_record_ws");
    return totalNewStrings;
}

//
// This will create new columns by splitting the array of strings vertically.
// All the first tokens go in the first column, all the second tokens go in the second column, etc.
// It is comparable to Pandas split with expand=True but the rows/columns are transposed.
// Example:
//   import pandas as pd
//   pd_series = pd.Series(['', None, 'a_b', '_a_b_', '__aa__bb__', '_a__bbb___c', '_aa_b__ccc__'])
//   print(pd_series.str.split(pat='_', expand=True))
//            0     1     2     3     4     5     6
//      0    ''  None  None  None  None  None  None
//      1  None  None  None  None  None  None  None
//      2     a     b  None  None  None  None  None
//      3    ''     a     b    ''  None  None  None
//      4    ''    ''    aa    ''    bb    ''    ''
//      5    ''     a    ''   bbb    ''    ''     c
//      6    ''    aa     b    ''   ccc    ''    ''
//
//   print(pd_series.str.split(pat='_', n=1, expand=True))
//            0            1
//      0    ''         None
//      1  None         None
//      2     a            b
//      3    ''         a_b_
//      4    ''    _aa__bb__
//      5    ''   a__bbb___c
//      6    ''  aa_b__ccc__
//
//   print(pd_series.str.split(pat='_', n=2, expand=True))
//            0     1         2
//      0    ''  None      None
//      1  None  None      None
//      2     a     b      None
//      3    ''     a        b_
//      4    ''        aa__bb__
//      5    ''     a  _bbb___c
//      6    ''    aa  b__ccc__
//
unsigned int NVStrings::split( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results)
{
    if( delimiter==0 )
        return split(maxsplit,results);
    auto execpol = rmm::exec_policy(0);
    unsigned int dellen = (unsigned int)strlen(delimiter);
    char* d_delimiter = device_alloc<char>(dellen+1,0);
    CUDA_TRY( cudaMemcpyAsync(d_delimiter,delimiter,dellen+1,cudaMemcpyHostToDevice))
    int tokens = 0;
    if( maxsplit > 0 )
        tokens = maxsplit + 1; // makes consistent with Pandas

    // need to count how many output strings per string
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            token_counter(d_strings,d_delimiter,dellen,tokens,d_counts));

    int columnsCount = *thrust::max_element(execpol->on(0), counts.begin(), counts.end() );
    // boundary case: if no columns, return one null column (issue #119)
    if( columnsCount==0 )
        results.push_back(new NVStrings(count));

    // create each column
    for( int col=0; col < columnsCount; ++col )
    {
        // first, build a vector of pair<char*,int>'s' for each column
        // each pair points to a string for this column for each row
        //st = GetTime();
        rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
        thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [d_strings, col, d_delimiter, dellen, d_counts, d_indexes] __device__(unsigned int idx){
                custring_view* dstr = d_strings[idx];
                d_indexes[idx].first = nullptr;   // initialize to
                d_indexes[idx].second = 0;  // null string
                if( !dstr )
                    return;
                // dcount already accounts for the maxsplit value
                int dcount = d_counts[idx];
                if( col >= dcount )
                    return; // passed the end for this string
                // skip delimiters until we reach this column
                int dchars = custring_view::chars_in_string(d_delimiter,dellen);
                int spos = 0, nchars = dstr->chars_count();
                int epos = nchars;
                for( int c=0; c < (dcount-1); ++c )
                {
                    epos = dstr->find(d_delimiter,dellen,spos);
                    if( epos < 0 )
                    {
                        epos = nchars;
                        break;
                    }
                    if( c==col )  // found our column
                        break;
                    spos = epos + dchars;
                    epos = nchars;
                }
                // this will be the string for this column
                if( spos < epos )
                {
                    spos = dstr->byte_offset_for(spos); // convert char pos
                    epos = dstr->byte_offset_for(epos); // to byte offset
                    d_indexes[idx].first = dstr->data() + spos;
                    d_indexes[idx].second = (epos-spos);
                }
                else
                {   // this will create empty string instead of null one
                    d_indexes[idx].first = dstr->data();
                }
            });

        //cudaError_t err = cudaDeviceSynchronize();
        //if( err != cudaSuccess )
        //{
        //    fprintf(stderr,"nvs-split(%s,%d), col=%d\n",delimiter,maxsplit,col);
        //    printCudaError(err);
        //}
        //
        NVStrings* column = NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
        results.push_back(column);
    }
    //
    RMM_FREE(d_delimiter,0);
    return (unsigned int)results.size();
}

//
// This is the whitespace-delimiter version of the column split function.
// Like the one above, it can be compared to Pandas split with expand=True but
// with the rows/columns transposed.
//
//  import pandas as pd
//  pd_series = pd.Series(['', None, 'a b', ' a b ', '  aa  bb  ', ' a  bbb   c', ' aa b  ccc  '])
//  print(pd_series.str.split(pat=None, expand=True))
//            0     1     2
//      0  None  None  None
//      1  None  None  None
//      2     a     b  None
//      3     a     b  None
//      4    aa    bb  None
//      5     a   bbb     c
//      6    aa     b   ccc
//
//  print(pd_series.str.split(pat=None, n=1, expand=True))
//            0         1
//      0  None      None
//      1  None      None
//      2     a         b
//      3     a        b
//      4    aa      bb
//      5     a   bbb   c
//      6    aa  b  ccc
//
//  print(pd_series.str.split(pat=None, n=2, expand=True))
//            0     1      2
//      0  None  None   None
//      1  None  None   None
//      2     a     b   None
//      3     a     b   None
//      4    aa    bb   None
//      5     a   bbb      c
//      6    aa     b  ccc
//
// Like the split_record method, there are no empty strings here.
//
unsigned int NVStrings::split( int maxsplit, std::vector<NVStrings*>& results)
{
    auto execpol = rmm::exec_policy(0);
    int tokens = 0;
    if( maxsplit > 0 )
        tokens = maxsplit + 1; // makes consistent with Pandas

    // need to count how many output strings per string
    unsigned int count = size();
    custring_view_array d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            whitespace_token_counter(d_strings,tokens,d_counts));

    int columnsCount = *thrust::max_element(execpol->on(0), counts.begin(), counts.end() );
    // boundary case: if no columns, return one null column (issue #119)
    if( columnsCount==0 )
        results.push_back(new NVStrings(count));

    // create each column
    for( int col=0; col < columnsCount; ++col )
    {
        // first, build a vector of pair<char*,int>'s' for each column
        // each pair points to a string for this column for each row
        //st = GetTime();
        rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
        thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [d_strings, col, tokens, d_counts, d_indexes] __device__(unsigned int idx){
                custring_view* dstr = d_strings[idx];
                d_indexes[idx].first = nullptr;   // initialize to
                d_indexes[idx].second = 0;  // null string
                if( !dstr )
                    return; // null string
                int dcount = d_counts[idx];
                if( col >= dcount )
                    return;
                int c = 0, nchars = dstr->chars_count();
                int spos = 0, epos = nchars;
                //printf(">%d:tokens=%d,dcount=%d,nchars=%d\n",col,tokens,dcount,nchars);
                bool spaces = true;
                for( int pos=0; pos < nchars; ++pos )
                {
                    Char ch = dstr->at(pos);
                    if( spaces == (ch <= ' ') )
                    {
                        if( spaces )
                            spos = pos+1;
                        else
                            epos = pos+1;
                        continue;
                    }
                    if( !spaces )
                    {
                        epos = nchars;
                        if( (c+1)==tokens )
                            break;
                        epos = pos;
                        if( c==col )
                            break;
                        spos = pos+1;
                        epos = nchars;
                        ++c;
                    }
                    spaces = !spaces;
                }
                if( spos < epos )
                {
                    spos = dstr->byte_offset_for(spos); // convert char pos
                    epos = dstr->byte_offset_for(epos); // to byte offset
                    //printf(">%d:spos=%d,epos=%d\n",c,spos,epos);
                    d_indexes[idx].first = dstr->data() + spos;
                    d_indexes[idx].second = (epos-spos);
                }
                //else
                //{   no empty strings in split-column-whitespace
                //    d_indexes[idx].first = dstr->data();
                //}
            });

        //cudaError_t err = cudaDeviceSynchronize();
        //if( err != cudaSuccess )
        //{
        //    fprintf(stderr,"nvs-split-ws(%d), col=%d\n",maxsplit,col);
        //    printCudaError(err);
        //}
        //
        NVStrings* column = NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
        results.push_back(column);
    }
    //
    return (unsigned int)results.size();
}
//
// The split-from-the-right version of split
//
unsigned int NVStrings::rsplit( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results)
{
    if( delimiter==0 )
        return rsplit(maxsplit,results);
    auto execpol = rmm::exec_policy(0);
    unsigned int dellen = (unsigned int)strlen(delimiter);
    char* d_delimiter = device_alloc<char>(dellen+1,0);
    CUDA_TRY( cudaMemcpyAsync(d_delimiter,delimiter,dellen+1,cudaMemcpyHostToDevice))
    int tokens = 0;
    if( maxsplit > 0 )
        tokens = maxsplit + 1; // makes consistent with Pandas

    // need to count how many output strings per string
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            token_counter(d_strings,d_delimiter,dellen,tokens,d_counts));

    int columnsCount = *thrust::max_element(execpol->on(0), counts.begin(), counts.end() );
    // boundary case: if no columns, return one null column (issue #119)
    if( columnsCount==0 )
        results.push_back(new NVStrings(count));

    // create each column
    for( int col = 0; col < columnsCount; ++col )
    {
        // first, build a vector of pair<char*,int>'s' for each column
        // each pair points to a string for this column for each row
        rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
        thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [d_strings, col, d_delimiter, dellen, d_counts, d_indexes] __device__(unsigned int idx){
                custring_view* dstr = d_strings[idx];
                d_indexes[idx].first = nullptr;   // initialize to
                d_indexes[idx].second = 0;  // null string
                if( !dstr )
                    return;
                // dcount already accounts for the maxsplit value
                int dcount = d_counts[idx];
                if( col >= dcount )
                    return; // passed the end for this string
                // skip delimiters until we reach this column
                int dchars = custring_view::chars_in_string(d_delimiter,dellen);
                int spos = 0, nchars = dstr->chars_count();
                int epos = nchars, pos = dstr->size()-1;
                for( int c=(dcount-1); c > 0; --c )
                {
                    spos = dstr->rfind(d_delimiter,dellen,0,epos);
                    if( spos < 0 )
                    {
                        spos = 0;
                        break;
                    }
                    if( c==col ) // found our column
                    {
                        spos += dchars;  // do not include delimiter
                        break;
                    }
                    epos = spos;
                    spos = 0;
                }
                // this will be the string for this column
                if( spos < epos )
                {
                    spos = dstr->byte_offset_for(spos); // convert char pos
                    epos = dstr->byte_offset_for(epos); // to byte offset
                    d_indexes[idx].first = dstr->data() + spos;
                    d_indexes[idx].second = (epos-spos);
                }
                else
                {   // this will create empty string instead of null one
                    d_indexes[idx].first = dstr->data();
                }
            });

        //cudaError_t err = cudaDeviceSynchronize();
        //if( err != cudaSuccess )
        //{
        //    fprintf(stderr,"nvs-rsplit(%s,%d)\n",delimiter,maxsplit);
        //    printCudaError(err);
        //}
        //
        NVStrings* column = NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
        results.push_back(column);
    }
    //
    RMM_FREE(d_delimiter,0);
    return (unsigned int)results.size();
}

//
// The whitespace-delimited version of rsplit.
//
unsigned int NVStrings::rsplit( int maxsplit, std::vector<NVStrings*>& results)
{
    auto execpol = rmm::exec_policy(0);
    int tokens = 0;
    if( maxsplit > 0 )
        tokens = maxsplit + 1; // makes consistent with Pandas

    // need to count how many output strings per string
    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            whitespace_token_counter(d_strings,tokens,d_counts));

    int columnsCount = *thrust::max_element(execpol->on(0), counts.begin(), counts.end() );
    // boundary case: if no columns, return one null column (issue #119)
    if( columnsCount==0 )
        results.push_back(new NVStrings(count));

    // create each column
    for( int col = 0; col < columnsCount; ++col )
    {
        // first, build a vector of pair<char*,int>'s' for each column
        // each pair points to a string for this column for each row
        rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
        thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [d_strings, col, columnsCount, tokens, d_counts, d_indexes] __device__(unsigned int idx){
                custring_view* dstr = d_strings[idx];
                d_indexes[idx].first = nullptr;   // initialize to
                d_indexes[idx].second = 0;  // null string
                if( !dstr )
                    return; // null string
                int dcount = d_counts[idx];
                if( col >= dcount )
                    return;
                int c = (dcount-1), nchars = dstr->chars_count();
                int spos = 0, epos = nchars;
                //printf(">%d:tokens=%d,dcount=%d,nchars=%d\n",col,tokens,dcount,nchars);
                bool spaces = true;
                for( int pos=nchars; pos > 0; --pos )
                {
                    Char ch = dstr->at(pos-1);
                    if( spaces == (ch <= ' ') )
                    {
                        if( spaces )
                            epos = pos-1;
                        else
                            spos = pos-1;
                        continue;
                    }
                    if( !spaces )
                    {
                        spos = 0;
                        if( (columnsCount-c)==tokens )
                            break;
                        spos = pos;
                        if( c==col )
                            break;
                        epos = pos-1;
                        spos = 0;
                        --c;
                    }
                    spaces = !spaces;
                }
                if( spos < epos )
                {
                    spos = dstr->byte_offset_for(spos); // convert char pos
                    epos = dstr->byte_offset_for(epos); // to byte offset
                    //printf(">%d:spos=%d,epos=%d\n",c,spos,epos);
                    d_indexes[idx].first = dstr->data() + spos;
                    d_indexes[idx].second = (epos-spos);
                }
                //else
                //{   no empty strings in rsplit column whitespace
                //    d_indexes[idx].first = dstr->data();
                //}
            });

        //cudaError_t err = cudaDeviceSynchronize();
        //if( err != cudaSuccess )
        //{
        //    fprintf(stderr,"nvs-rsplit-ws(%d)\n",maxsplit);
        //    printCudaError(err);
        //}
        //
        NVStrings* column = NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
        results.push_back(column);
    }
    //
    return (unsigned int)results.size();
}

//
// Partition is split the string at the first occurrence of delimiter, and return 3 elements containing
// the part before the delimiter, the delimiter itself, and the part after the delimiter.
// If the delimiter is not found, return 3 elements containing the string itself, followed by two empty strings.
//
// >>> import pandas as pd
// >>> strs = pd.Series(['héllo', None, 'a_bc_déf', 'a__bc', '_ab_cd', 'ab_cd_'])
// >>> strs.str.partition('_')
//        0     1       2
// 0  héllo
// 1   None  None    None
// 2      a     _  bc_déf
// 3      a     _     _bc
// 4            _   ab_cd
// 5     ab     _     cd_
//
int NVStrings::partition( const char* delimiter, std::vector<NVStrings*>& results)
{
    if( delimiter==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(delimiter);
    if( bytes==0 )
        return 0; // just return original list?

    auto execpol = rmm::exec_policy(0);
    // copy delimiter to device
    char* d_delimiter = device_alloc<char>(bytes,0);
    CUDA_TRY( cudaMemcpyAsync(d_delimiter,delimiter,bytes,cudaMemcpyHostToDevice))
    int d_asize = custring_view::alloc_size((char*)delimiter,bytes);
    d_asize = ALIGN_SIZE(d_asize);

    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    // build int arrays to hold each string's partition sizes
    int totalSizes = 2 * count;
    rmm::device_vector<int> sizes(totalSizes,0), totals(count,0);
    int* d_sizes = sizes.data().get();
    int* d_totals = totals.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), count,
        [d_strings, d_delimiter, bytes, d_asize, d_sizes, d_totals] __device__(size_t idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int* dsizes = &(d_sizes[idx*2]);
            d_totals[idx] = dstr->split_size(d_delimiter,bytes,dsizes,2) + d_asize;
        });

    cudaDeviceSynchronize();

    // build an output array of custring_views* arrays for each value
    // there will always be 3 per string
    thrust::host_vector<int> h_totals(totals);
    thrust::host_vector<char*> h_buffers(count,nullptr);
    thrust::host_vector<custring_view_array> h_splits(count,nullptr);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        NVStrings* result = new NVStrings(3);
        results.push_back(result);
        h_splits[idx] = result->pImpl->getStringsPtr();

        int totalSize = h_totals[idx];
        char* d_buffer = device_alloc<char>(totalSize,0);
        result->pImpl->setMemoryBuffer(d_buffer,totalSize);
        h_buffers[idx] = d_buffer;
    }

    rmm::device_vector<custring_view_array> splits(h_splits);
    custring_view_array* d_splits = splits.data().get();
    rmm::device_vector<char*> buffers(h_buffers);
    char** d_buffers = buffers.data().get();

    // do the partition and fill in the arrays
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
    [d_strings, d_delimiter, bytes, d_buffers, d_sizes, d_splits] __device__(unsigned int idx){
        custring_view* dstr = d_strings[idx];
        if( !dstr )
            return;
        char* buffer = (char*)d_buffers[idx];
        int* dsizes = &(d_sizes[idx*2]);
        custring_view_array d_strs = d_splits[idx];

        d_strs[0] = custring_view::create_from(buffer,0,0);
        buffer += ALIGN_SIZE(dsizes[0]);
        d_strs[1] = custring_view::create_from(buffer,0,0);
        buffer += ALIGN_SIZE(dsizes[1]);
        d_strs[2] = custring_view::create_from(buffer,0,0);

        //
        int dcount = dstr->rsplit_size(d_delimiter,bytes,0,2);
        dstr->split(d_delimiter,bytes,2,d_strs);
        if( dcount==2 )
        {   // insert delimiter element in the middle
            custring_view* tmp  = d_strs[1];
            d_strs[1] = custring_view::create_from(buffer,d_delimiter,bytes);
            d_strs[2] = tmp;
        }
    });

    printCudaError(cudaDeviceSynchronize(),"nvs-partition");
    RMM_FREE(d_delimiter,0);
    return count;
}

//
// This follows most of the same logic as partition above except that the delimiter
// search starts from the end of the string. Also, if no delimiter is found the
// resulting array includes two empty strings followed by the original string.
//
// >>> import pandas as pd
// >>> strs = pd.Series(['héllo', None, 'a_bc_déf', 'a__bc', '_ab_cd', 'ab_cd_'])
// >>> strs.str.rpartition('_')
//        0     1      2
// 0               héllo
// 1   None  None   None
// 2   a_bc     _    déf
// 3     a_     _     bc
// 4    _ab     _     cd
// 5  ab_cd     _
//
int NVStrings::rpartition( const char* delimiter, std::vector<NVStrings*>& results)
{
    if( delimiter==0 )
        return 0;
    unsigned int bytes = (unsigned int)strlen(delimiter);
    if( bytes==0 )
        return 0; // just return original list?

    auto execpol = rmm::exec_policy(0);
    // copy delimiter to device
    char* d_delimiter = device_alloc<char>(bytes,0);
    CUDA_TRY( cudaMemcpyAsync(d_delimiter,delimiter,bytes,cudaMemcpyHostToDevice))
    int d_asize = custring_view::alloc_size((char*)delimiter,bytes);
    d_asize = ALIGN_SIZE(d_asize);

    unsigned int count = size();
    custring_view** d_strings = pImpl->getStringsPtr();
    // build int arrays to hold each string's partition sizes
    int totalSizes = 2 * count;
    rmm::device_vector<int> sizes(totalSizes,0), totals(count,0);
    int* d_sizes = sizes.data().get();
    int* d_totals = totals.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, bytes, d_asize, d_sizes, d_totals] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int* dsizes = &(d_sizes[idx*2]);
            //d_totals[idx] = dstr->rpartition_size(d_delimiter,bytes,dsizes);
            d_totals[idx] = dstr->rsplit_size(d_delimiter,bytes,dsizes,2) + d_asize;
        });

    cudaDeviceSynchronize();

    // now build an output array of custring_views* arrays for each value
    // there will always be 3 per string
    thrust::host_vector<int> h_totals(totals);
    thrust::host_vector<char*> h_buffers(count,nullptr);
    thrust::host_vector<custring_view_array> h_splits(count,nullptr);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        NVStrings* result = new NVStrings(3);
        results.push_back(result);
        h_splits[idx] = result->pImpl->getStringsPtr();

        int totalSize = h_totals[idx];
        char* d_buffer = device_alloc<char>(totalSize,0);
        result->pImpl->setMemoryBuffer(d_buffer,totalSize);
        h_buffers[idx] = d_buffer;
    }

    rmm::device_vector<custring_view_array> splits(h_splits);
    custring_view_array* d_splits = splits.data().get();
    rmm::device_vector<char*> buffers(h_buffers);
    char** d_buffers = buffers.data().get();

    // do the partition and fill in the arrays
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
    [d_strings, d_delimiter, bytes, d_buffers, d_sizes, d_splits] __device__(unsigned int idx){
        custring_view* dstr = d_strings[idx];
        if( !dstr )
            return;
        char* buffer = (char*)d_buffers[idx];
        int* dsizes = &(d_sizes[idx*2]);
        custring_view_array d_strs = d_splits[idx];

        d_strs[0] = custring_view::create_from(buffer,0,0);
        buffer += ALIGN_SIZE(dsizes[0]);
        d_strs[1] = custring_view::create_from(buffer,0,0);
        buffer += ALIGN_SIZE(dsizes[1]);
        d_strs[2] = custring_view::create_from(buffer,0,0);

        //
        int dcount = dstr->rsplit_size(d_delimiter,bytes,0,2);
        dstr->rsplit(d_delimiter,bytes,2,d_strs);
        // reorder elements
        if( dcount==1 )
        {   // if only one element, it goes on the end
            custring_view* tmp  = d_strs[2];
            d_strs[2] = d_strs[0];
            d_strs[0] = tmp;
        }
        if( dcount==2 )
        {   // insert delimiter element in the middle
            custring_view* tmp  = d_strs[1];
            d_strs[1] = custring_view::create_from(buffer,d_delimiter,bytes);
            d_strs[2] = tmp;
        }
    });

    printCudaError(cudaDeviceSynchronize(),"nvs-rpartition");
    RMM_FREE(d_delimiter,0);
    return count;
}
