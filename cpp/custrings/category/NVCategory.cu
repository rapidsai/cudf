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
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/error_utils.hpp>

#include "nvstrings/NVCategory.h"
#include "nvstrings/NVStrings.h"
#include "nvstrings/ipc_transfer.h"

#include "../custring_view.cuh"
#include "../custring.cuh"
#include "../util.h"

//
typedef custring_view** custring_view_array;

//#define ALIGN_SIZE(v)  (((v+7)/8)*8)

//static void printDeviceInts( const char* title, int* d_ints, int count )
//{
//    thrust::host_vector<int> ints(count);
//    int* h_ints = ints.data();
//    cudaMemcpy( h_ints, d_ints, count * sizeof(int), cudaMemcpyDeviceToHost);
//    if( title )
//        printf("%s:\n",title);
//    for( int i=0; i < count; ++i )
//        printf(" %d",h_ints[i]);
//    printf("\n");
//}

//
class NVCategoryImpl
{
public:
    //
    rmm::device_vector<custring_view*>* pList;
    rmm::device_vector<int>* pMap;
    void* memoryBuffer;
    size_t bufferSize; // total memory size
    cudaStream_t stream_id;
    bool bIpcHandle;

    //
    NVCategoryImpl()
    : bufferSize(0), memoryBuffer(0), pList(0), pMap(0), stream_id(0), bIpcHandle(false)
    {}

    ~NVCategoryImpl()
    {
        if( memoryBuffer )
        {
            if( bIpcHandle )
                cudaIpcCloseMemHandle(memoryBuffer);
            else
                RMM_FREE(memoryBuffer,0);
        }
        delete pList;
        delete pMap;
        memoryBuffer = nullptr;
        bufferSize = 0;
    }

    inline custring_view_array getStringsPtr()
    {
        custring_view_array rtn = nullptr;
        if( pList )
            rtn = pList->data().get();
        return rtn;
    }

    inline custring_view_array createStringsListFrom( custring_view_array strings, unsigned int keys )
    {
        pList = new rmm::device_vector<custring_view*>(keys);
        thrust::copy( rmm::exec_policy(0)->on(0), strings, strings+keys, pList->data().get() );
        //cudaMemcpy(pList->data().get(), strings, keys*sizeof(custring_view*), cudaMemcpyDeviceToDevice);
        return pList->data().get();
    }

    inline char* getMemoryPtr() { return (char*)memoryBuffer; }

    inline int* getMapPtr()
    {
        int* rtn = nullptr;
        if( pMap )
            rtn = pMap->data().get();
        return rtn;
    }

    inline int* createMapFrom( int* vals, unsigned int count )
    {
        pMap = new rmm::device_vector<int>(count);
        thrust::copy( rmm::exec_policy(0)->on(0), vals, vals+count, pMap->data().get());
        //cudaMemcpy(pMap->data().get(), vals, count*sizeof(int), cudaMemcpyDeviceToDevice);
        return pMap->data().get();
    }

    inline void setMemoryBuffer( void* ptr, size_t memSize )
    {
        bufferSize = memSize;
        memoryBuffer = ptr;
    }

    inline void setMemoryHandle( void* ptr, size_t memSize )
    {
        setMemoryBuffer(ptr,memSize);
        bIpcHandle = true;
    }
};

//
NVCategory::NVCategory()
{
    pImpl = new NVCategoryImpl;
}

NVCategory::~NVCategory()
{
    delete pImpl;
}

// utility to create keys from array of string pointers
// pImpl must exist but it's pList should be null -- this method will create it
void NVCategoryImpl_keys_from_index( NVCategoryImpl* pImpl, thrust::pair<const char*,size_t>* d_pairs, unsigned int ucount )
{
    auto execpol = rmm::exec_policy(0);
    // add up the lengths
    rmm::device_vector<size_t> lengths(ucount,0);
    size_t* d_lengths = lengths.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), ucount,
        [d_pairs, d_lengths] __device__(size_t idx){
            const char* str = d_pairs[idx].first;
            int bytes = (int)d_pairs[idx].second;
            if( str )
                d_lengths[idx] = ALIGN_SIZE(custring_view::alloc_size((char*)str,bytes));
        });
    // create output buffer to hold the string keys
    size_t outsize = thrust::reduce(execpol->on(0), lengths.begin(), lengths.end());
    char* d_buffer = device_alloc<char>(outsize,0);
    pImpl->setMemoryBuffer(d_buffer,outsize);
    rmm::device_vector<size_t> offsets(ucount,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    size_t* d_offsets = offsets.data().get();
    // create the vector to hold the pointers
    rmm::device_vector<custring_view*>* pList = new rmm::device_vector<custring_view*>(ucount,nullptr);
    custring_view_array d_results = pList->data().get();
    // copy keys strings to new memory buffer
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), ucount,
        [d_pairs, d_buffer, d_offsets, d_results] __device__ (size_t idx) {
            const char* str = d_pairs[idx].first;
            int bytes = (int)d_pairs[idx].second;
            if( str )
                d_results[idx] = custring_view::create_from(d_buffer+d_offsets[idx],(char*)str,bytes);
        });
    pImpl->pList = pList;
}

// utility to create keys from array of custrings
// pImpl must exist but it's pList should be null -- this method will create it
void NVCategoryImpl_keys_from_custringarray( NVCategoryImpl* pImpl, custring_view_array d_keys, unsigned int ucount )
{
    auto execpol = rmm::exec_policy(0);
    // add up the lengths
    rmm::device_vector<size_t> lengths(ucount,0);
    size_t* d_lengths = lengths.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), ucount,
        [d_keys, d_lengths] __device__(size_t idx){
            custring_view* dstr = d_keys[idx];
            if( dstr )
                d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size());
        });
    // create output buffer to hold the string keys
    size_t outsize = thrust::reduce(execpol->on(0), lengths.begin(), lengths.end());
    char* d_buffer = device_alloc<char>(outsize,0);
    pImpl->setMemoryBuffer(d_buffer,outsize);
    rmm::device_vector<size_t> offsets(ucount,0);
    thrust::exclusive_scan(execpol->on(0),lengths.begin(),lengths.end(),offsets.begin());
    size_t* d_offsets = offsets.data().get();
    // create the vector to hold the pointers
    rmm::device_vector<custring_view*>* pList = new rmm::device_vector<custring_view*>(ucount,nullptr);
    custring_view_array d_results = pList->data().get();
    // copy keys strings to new memory buffer
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), ucount,
        [d_keys, d_buffer, d_offsets, d_results] __device__ (size_t idx) {
            custring_view* dstr = d_keys[idx];
            if( dstr )
                d_results[idx] = custring_view::create_from(d_buffer+d_offsets[idx],*dstr);
        });
    pImpl->pList = pList;
}

// Utility to create category instance data from array of string pointers (in device memory).
// It does all operations using the given pointers (or copies) to build the map.
// This method can be given the index values from the NVStrings::create_index.
// So however an NVStrings can be created can also create an NVCategory.
//
// Should investigating converting this use custring pointers instead of index pairs.
// It would likely save some processing since we can create custrings from custrings.
void NVCategoryImpl_init(NVCategoryImpl* pImpl, std::pair<const char*,size_t>* pairs, unsigned int count, bool bdevmem, bool bindexescopied=false )
{
    auto execpol = rmm::exec_policy(0);

    // make a copy of the indexes so we can sort them, etc
    thrust::pair<const char*,size_t>* d_pairs = nullptr;
    if( bdevmem )
    {
        if( bindexescopied )                                    // means caller already made a temp copy
            d_pairs = (thrust::pair<const char*,size_t>*)pairs; // and we can just use it here
        else
        {
            d_pairs = device_alloc<thrust::pair<const char*,size_t>>(count,0);
            CUDA_TRY(cudaMemcpyAsync(d_pairs,pairs,sizeof(thrust::pair<const char*,size_t>)*count,cudaMemcpyDeviceToDevice))
        }
    }
    else
    {
        d_pairs = device_alloc<thrust::pair<const char*,size_t>>(count,0);
        CUDA_TRY(cudaMemcpyAsync(d_pairs,pairs,sizeof(thrust::pair<const char*,size_t>)*count,cudaMemcpyHostToDevice))
    }

    //
    // example strings used in comments                                e,a,d,b,c,c,c,e,a
    //
    rmm::device_vector<int> indexes(count);
    thrust::sequence(execpol->on(0),indexes.begin(),indexes.end()); // 0,1,2,3,4,5,6,7,8
    int* d_indexes = indexes.data().get();
    // sort by key (string)                                            a,a,b,c,c,c,d,e,e
    // and indexes go along for the ride                               1,8,3,4,5,6,2,0,7
    thrust::sort_by_key(execpol->on(0), d_pairs, d_pairs+count, d_indexes,
        [] __device__( thrust::pair<const char*,size_t>& lhs, thrust::pair<const char*,size_t>& rhs ) {
            if( lhs.first==0 || rhs.first==0 )
                return rhs.first!=0; // null < non-null
            return custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second) < 0;
        });

    // build the map; this will let us lookup strings by index
    rmm::device_vector<int>* pMap = new rmm::device_vector<int>(count,0);
    int* d_map = pMap->data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<int>(0), count,
        [d_pairs, d_map] __device__ (int idx) {
            if( idx==0 )
                return;
            const char* ptr1 = d_pairs[idx-1].first;
            const char* ptr2 = d_pairs[idx].first;
            unsigned int len1 = (unsigned int)d_pairs[idx-1].second, len2 = (unsigned int)d_pairs[idx].second;
            //d_map[idx] = (int)(custr::compare(ptr1,len1,ptr2,len2)!=0);
            int cmp = 0; // vvvvv - probably faster than - ^^^^^
            if( !ptr1 || !ptr2 )
                cmp = (int)(ptr1!=ptr2);
            else if( len1 != len2 )
                cmp = 1;
            else
                for( int i=0; !cmp && (i < len1); ++i)
                    cmp = (int)(*ptr1++ != *ptr2++);
            d_map[idx] = cmp;
        });
    //
    // d_map now identifies just string changes                        0,0,1,1,0,0,1,1,0
    int ucount = thrust::reduce(execpol->on(0), pMap->begin(), pMap->end()) + 1;
    // scan converts to index values                                   0,0,1,2,2,2,3,4,4
    thrust::inclusive_scan(execpol->on(0), pMap->begin(), pMap->end(), pMap->begin());
    // re-sort will complete the map                                   4,0,3,1,2,2,2,4,0
    thrust::sort_by_key(execpol->on(0), indexes.begin(), indexes.end(), pMap->begin());
    pImpl->pMap = pMap;  // index -> str is now just a lookup in the map

    // now remove duplicates from string list                          a,b,c,d,e
    thrust::unique(execpol->on(0), d_pairs, d_pairs+count,
        [] __device__ ( thrust::pair<const char*,size_t> lhs, thrust::pair<const char*,size_t> rhs ) {
            if( lhs.first==0 || rhs.first==0 )
                return lhs.first==rhs.first;
            if( lhs.second != rhs.second )
                return false;
            return custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second)==0;
        });

    // finally, create new string vector of just the keys
    NVCategoryImpl_keys_from_index(pImpl,d_pairs,ucount);
    //err = cudaDeviceSynchronize();
    //if( err!=cudaSuccess )
    //    fprintf(stderr,"category: error(%d) creating %'d strings\n",(int)err,ucount);
    if( !bindexescopied )
        RMM_FREE(d_pairs,0);
}

NVCategory* NVCategory::create_from_index(std::pair<const char*,size_t>* strs, unsigned int count, bool devmem )
{
    NVCategory* rtn = new NVCategory;
    if( count )
        NVCategoryImpl_init(rtn->pImpl,strs,count,devmem);
    return rtn;
}

NVCategory* NVCategory::create_from_array(const char** strs, unsigned int count)
{
    NVCategory* rtn = new NVCategory;
    if( count==0 )
        return rtn;
    NVStrings* dstrs = NVStrings::create_from_array(strs,count);
    rmm::device_vector<std::pair<const char*,size_t>> indexes(count);
    dstrs->create_index(indexes.data().get());
    NVCategoryImpl_init(rtn->pImpl,indexes.data().get(),count,true,true);
    NVStrings::destroy(dstrs);
    return rtn;
}

NVCategory* NVCategory::create_from_strings(NVStrings& strs)
{
    NVCategory* rtn = new NVCategory;
    unsigned int count = strs.size();
    if( count==0 )
        return rtn;
    rmm::device_vector<std::pair<const char*,size_t>> indexes(count);
    strs.create_index(indexes.data().get());
    NVCategoryImpl_init(rtn->pImpl,indexes.data().get(),count,true,true);
    return rtn;
}

NVCategory* NVCategory::create_from_strings(std::vector<NVStrings*>& strs)
{
    NVCategory* rtn = new NVCategory;
    unsigned int count = 0;
    for( unsigned int idx=0; idx < (unsigned int)strs.size(); idx++ )
        count += strs[idx]->size();
    if( count==0 )
        return rtn;
    rmm::device_vector<std::pair<const char*,size_t>> indexes(count);
    std::pair<const char*,size_t>* ptr = indexes.data().get();
    for( unsigned int idx=0; idx < (unsigned int)strs.size(); idx++ )
    {
        strs[idx]->create_index(ptr);
        ptr += strs[idx]->size();
    }
    NVCategoryImpl_init(rtn->pImpl,indexes.data().get(),count,true,true);
    return rtn;
}

// bitmask is in arrow format
NVCategory* NVCategory::create_from_offsets(const char* strs, unsigned int count, const int* offsets, const unsigned char* nullbitmask, int nulls, bool bdevmem)
{
    NVCategory* rtn = new NVCategory;
    if( count==0 )
        return rtn;
    NVStrings* dstrs = NVStrings::create_from_offsets(strs,count,offsets,nullbitmask,nulls,bdevmem);
    rmm::device_vector<std::pair<const char*,size_t>> indexes(count);
    dstrs->create_index(indexes.data().get()); // try using the custring one; may be more efficient
    NVCategoryImpl_init(rtn->pImpl,indexes.data().get(),count,true,true);
    NVStrings::destroy(dstrs);
    return rtn;
}

// create instance from ipc handle(s)
NVCategory* NVCategory::create_from_ipc( nvcategory_ipc_transfer& ipc )
{
    NVCategory* rtn = new NVCategory;
    unsigned int keys = ipc.keys;
    if( keys==0 )
        return rtn;
    rtn->pImpl->setMemoryHandle(ipc.getMemoryPtr(),ipc.size);
    custring_view_array d_strings = rtn->pImpl->createStringsListFrom((custring_view_array)ipc.getStringsPtr(),ipc.keys);
    // fix up the pointers for this context
    auto execpol = rmm::exec_policy(0);
    char* baseaddr = (char*)ipc.base_address;
    char* buffer = rtn->pImpl->getMemoryPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), keys,
        [buffer, baseaddr, d_strings] __device__(unsigned int idx){
            char* dstr = (char*)d_strings[idx];
            if( !dstr )
                return;
            size_t diff = dstr - baseaddr;
            char* newaddr = buffer + diff;
            d_strings[idx] = (custring_view*)newaddr;
        });
    // set the map values
    rtn->pImpl->createMapFrom( (int*)ipc.getMapPtr(), ipc.count );
    // done
    return rtn;
}

//
// Example merging two categories and remapping the values:
//
//   category1:---------           category2:---------
//   | strs1       key1 |          | strs2       key2 |
//   | abbfcf  ->  abcf |          | aadcce  ->  acde |
//   | 012345      0123 |          | 012345      0123 |
//   | 011323    <-'    |          | 002113    <-'    |
//    ------------------            ------------------
//
//   merge-remap should result in new category like:
//    strs              key
//    abbfcfaadcce  ->  abcdef
//                      012345
//    011525003224    <-'
//
//    abcfacde  ->  w = aabccdef
//    01234567      x = 04125673
//                  y = 00110111
//                  y'= 00122345 = scan(y)
//                  y"= 01250234 = sort(x,y')
//    v = 0125:0234      = this is y"
//    m = 011323:002113  = orig values from each category
//    m'= r1[v1]:r2[v2] -> 011525:003224
//    w'= unique(w)     -> abcdef
//
// This logic works for any number of categories.
// Loop is required at the beginning to combine all the keys.
// And loop is required at the end to combine and remap the values.
//
NVCategory* NVCategory::create_from_categories(std::vector<NVCategory*>& cats)
{
    NVCategory* rtn = new NVCategory();
    if( cats.empty() )
        return rtn;
    unsigned int count = 0;
    unsigned int mcount = 0;
    for( unsigned int idx=0; idx < cats.size(); ++idx )
    {
        NVCategory* cat = cats[idx];
        count += cat->keys_size();
        mcount += cat->size();
    }
    if( count==0 )
        return rtn;

    auto execpol = rmm::exec_policy(0);
    // first combine the keys into one array
    rmm::device_vector<custring_view*> wstrs(count);
    custring_view_array d_w = wstrs.data().get();
    for( unsigned int idx=0; idx < cats.size(); ++idx )
    {
        NVCategory* cat = cats[idx];
        custring_view_array d_keys = cat->pImpl->getStringsPtr();
        unsigned int ksize = cat->keys_size();
        if( ksize )
            //cudaMemcpy(d_w, d_keys, ksize*sizeof(custring_view*),cudaMemcpyDeviceToDevice);
            thrust::copy( execpol->on(0), d_keys, d_keys+ksize, d_w );
        d_w += ksize;
    }
    d_w = wstrs.data().get(); // reset pointer
    rmm::device_vector<int> x(count);
    int* d_x = x.data().get(); // [0:count)
    thrust::sequence( execpol->on(0), d_x, d_x+count );
    thrust::stable_sort_by_key( execpol->on(0), d_w, d_w+count, d_x,
        [] __device__ (custring_view*& lhs, custring_view*& rhs) { return ((lhs && rhs) ? (lhs->compare(*rhs)<0) : (rhs!=0)); });
    // x-vector is sorted sequence we'll use to remap values
    rmm::device_vector<int> y(count,0); // y-vector will identify unique keys
    int* d_y = y.data().get();
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(1), (count-1),
        [d_y, d_w] __device__ (int idx) {
            custring_view* lhs = d_w[idx];
            custring_view* rhs = d_w[idx-1];
            if( lhs && rhs )
                d_y[idx] = (int)(lhs->compare(*rhs)!=0);
            else
                d_y[idx] = (int)(lhs!=rhs);
        });
    unsigned int kcount = (unsigned int)thrust::reduce( execpol->on(0), d_y, d_y+count )+1;
    // use gather to get unique keys
    // theory is that copy_if + gather on ints is faster than unique on strings
    //rmm::device_vector<int> nidxs(kcount);
    //thrust::counting_iterator<int> citr(0);
    //thrust::copy_if( execpol->on(0), citr, citr + count, nidxs.data().get(), [d_y] __device__ (const int& idx) { return (idx==0 || d_y[idx]); });
    //rmm::device_vector<custring_view*>* pNewList = new rmm::device_vector<custring_view*>(kcount,nullptr);
    //custring_view_array d_keys = pNewList->data().get(); // this will hold the merged keyset
    //thrust::gather( execpol->on(0), nidxs.begin(), nidxs.end(), d_w, d_keys );
    thrust::unique( execpol->on(0), d_w, d_w+count, [] __device__ (custring_view* lhs, custring_view* rhs) { return (lhs && rhs) ? (lhs->compare(*rhs)==0) : (lhs==rhs); });
    NVCategoryImpl_keys_from_custringarray(rtn->pImpl,d_w,kcount);
    // now create map to remap the values
    thrust::inclusive_scan(execpol->on(0), d_y, d_y+count, d_y );
    thrust::sort_by_key(execpol->on(0), d_x, d_x+count, d_y );
    rmm::device_vector<int>* pNewMap = new rmm::device_vector<int>(mcount);
    int* d_map = pNewMap->data().get();
    int* d_v = d_y;
    for( int idx=0; idx < (int)cats.size(); ++idx )
    {
        NVCategory* cat = cats[idx];
        unsigned int msize = cat->size();
        if( msize )
        {
            int* d_catmap = cat->pImpl->getMapPtr();
            thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(0), msize,
                [d_catmap, d_v, d_map] __device__ (int idx) {
                    int v = d_catmap[idx];
                    d_map[idx] = ( v<0 ? v : d_v[v] );
                });
        }
        d_v += cat->keys_size();
        d_map += msize;
    }
    // done
    rtn->pImpl->pMap = pNewMap;
    return rtn;
}

void NVCategory::destroy(NVCategory* inst)
{
    delete inst;
}

// dest should already be empty
void NVCategoryImpl_copy( NVCategoryImpl& dest, NVCategoryImpl& src )
{
    if( src.pList==0 )
        return;
    auto execpol = rmm::exec_policy(0);
    if( src.pMap )
    {
        unsigned int mcount = (unsigned int)src.pMap->size();
        rmm::device_vector<int>* pNewMap = new rmm::device_vector<int>(mcount,0);
        // copy map values from non-empty category instance
        //cudaMemcpy( pNewMap->data().get(), src.pMap->data().get(), mcount*sizeof(int), cudaMemcpyDeviceToDevice );
        thrust::copy( execpol->on(0), src.pMap->begin(), src.pMap->end(), pNewMap->begin() );
        dest.pMap = pNewMap;
    }
    // copy key strings buffer
    unsigned int ucount = (unsigned int)src.pList->size();
    rmm::device_vector<custring_view*>* pNewList = new rmm::device_vector<custring_view*>(ucount,nullptr);
    char* d_buffer = (char*)src.memoryBuffer;
    size_t bufsize = src.bufferSize;
    char* d_newbuffer = device_alloc<char>(bufsize,0);
    thrust::copy( execpol->on(0), d_buffer, d_buffer+bufsize, d_newbuffer);
    //cudaMemcpy(d_newbuffer,d_buffer,bufsize,cudaMemcpyDeviceToDevice);
    // need to set custring_view ptrs
    custring_view_array d_strings = src.getStringsPtr();
    custring_view_array d_results = pNewList->data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), ucount,
        [d_strings, d_buffer, d_newbuffer, d_results] __device__ (size_t idx) {
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {
                char* buffer = d_newbuffer + ((char*)dstr - d_buffer);
                d_results[idx] = (custring_view*)buffer;
            }
        });
    dest.pList = pNewList;
    dest.setMemoryBuffer( d_newbuffer, bufsize );
}

NVCategory::NVCategory(const NVCategory& cat)
{
    pImpl = new NVCategoryImpl;
    NVCategoryImpl_copy(*pImpl,*(cat.pImpl));
}

NVCategory& NVCategory::operator=(const NVCategory& cat)
{
    delete pImpl;
    pImpl = new NVCategoryImpl;
    NVCategoryImpl_copy(*pImpl,*(cat.pImpl));
    return *this;
}

NVCategory* NVCategory::copy()
{
    NVCategory* rtn = new NVCategory;
    NVCategoryImpl_copy(*(rtn->pImpl),*pImpl);
    return rtn;
}

const char* NVCategory::get_type_name()
{
    return "custring";
}

// return number of items
unsigned int NVCategory::size()
{
    unsigned int size = 0;
    if( pImpl->pMap )
        size = pImpl->pMap->size();
    return size;
}

// return number of keys
unsigned int NVCategory::keys_size()
{
    unsigned int size = 0;
    if( pImpl->pList )
        size =  pImpl->pList->size();
    return size;
}

// true if any null values exist
bool NVCategory::has_nulls()
{
    unsigned int count = keys_size();
    auto execpol = rmm::exec_policy(0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    int n = thrust::count_if(execpol->on(0), d_strings, d_strings+count,
            []__device__(custring_view* dstr) { return dstr==0; } );
    return n > 0;
}

// bitarray is for the values; bits are set in arrow format
// return the number of null values found
int NVCategory::set_null_bitarray( unsigned char* bitarray, bool devmem )
{
    unsigned int count = size();
    if( count==0 )
        return 0;
    auto execpol = rmm::exec_policy(0);
    unsigned int size = (count + 7)/8;
    unsigned char* d_bitarray = bitarray;
    if( !devmem )
        d_bitarray = device_alloc<unsigned char>(size,0);

    int nidx = -1;
    {
        custring_view_array d_strings = pImpl->getStringsPtr();
        rmm::device_vector<int> nulls(1,-1);
        thrust::copy_if( execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), thrust::make_counting_iterator<unsigned int>(keys_size()), nulls.begin(),
                [d_strings] __device__ (unsigned int idx) { return d_strings[idx]==0; } );
        nidx = nulls[0]; // should be the index of the null entry (or -1)
    }

    if( nidx < 0 )
    {   // no nulls, set everything to 1s
        cudaMemset(d_bitarray,255,size); // actually sets more bits than we need to
        if( !devmem )
        {
            CUDA_TRY(cudaMemcpyAsync(bitarray,d_bitarray,size,cudaMemcpyDeviceToHost))
            RMM_FREE(d_bitarray,0);
        }
        return 0; // no nulls;
    }

    // count nulls in range for return value
    int* d_map = pImpl->getMapPtr();
    unsigned int ncount = thrust::count_if(execpol->on(0), d_map, d_map + count,
        [nidx] __device__ (int index) { return (index==nidx); });
    // fill in the bitarray
    // the bitmask is in arrow format which means for each byte
    // the null indicator is in bit position right-to-left: 76543210
    // logic sets the high-bit and shifts to the right
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), size,
        [d_map, nidx, count, d_bitarray] __device__(unsigned int byteIdx){
            unsigned char byte = 0; // init all bits to zero
            for( unsigned int i=0; i < 8; ++i )
            {
                unsigned int idx = i + (byteIdx*8);
                byte = byte >> 1;
                if( idx < count )
                {
                    int index = d_map[idx];
                    byte |= (unsigned char)((index!=nidx) << 7);
                }
            }
            d_bitarray[byteIdx] = byte;
        });
    //
    if( !devmem )
    {
        CUDA_TRY(cudaMemcpyAsync(bitarray,d_bitarray,size,cudaMemcpyDeviceToHost))
        RMM_FREE(d_bitarray,0);
    }
    return ncount; // number of nulls
}

// build a string-index from this instances strings
int NVCategory::create_index(std::pair<const char*,size_t>* strs, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    custring_view_array d_strings = pImpl->getStringsPtr();
    int* d_map = pImpl->getMapPtr();
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_map, d_indexes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[d_map[idx]];
            if( dstr )
            {
                d_indexes[idx].first = (const char*)dstr->data();
                d_indexes[idx].second = (size_t)dstr->size();
            }
            else
            {
                d_indexes[idx].first = nullptr;
                d_indexes[idx].second = 0;
            }
        });

    //
    if( bdevmem )
        CUDA_TRY(cudaMemcpyAsync( strs, indexes.data().get(), count * sizeof(std::pair<const char*,size_t>), cudaMemcpyDeviceToDevice ))
    else
        CUDA_TRY(cudaMemcpyAsync( strs, indexes.data().get(), count * sizeof(std::pair<const char*,size_t>), cudaMemcpyDeviceToHost ))
    return 0;
}

int NVCategory::create_ipc_transfer( nvcategory_ipc_transfer& ipc )
{
    ipc.setStrsHandle(pImpl->getStringsPtr(),pImpl->getMemoryPtr(),keys_size());
    ipc.setMemHandle(pImpl->getMemoryPtr(),pImpl->bufferSize);
    ipc.setMapHandle(pImpl->getMapPtr(),size());
    return 0;
}

// return strings keys for this instance
NVStrings* NVCategory::get_keys()
{
    int count = keys_size();
    if( count==0 )
        return NVStrings::create_from_index(0,0);

    auto execpol = rmm::exec_policy(0);
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), count,
        [d_strings, d_indexes] __device__(size_t idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {
                d_indexes[idx].first = (const char*)dstr->data();
                d_indexes[idx].second = (size_t)dstr->size();
            }
            else
            {
                d_indexes[idx].first = nullptr;
                d_indexes[idx].second = 0;
            }
        });

    // create strings from index
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
}

//
int NVCategory::get_value(unsigned int index)
{
    if( index >= size() )
        return -1;
    int* d_map = pImpl->getMapPtr();
    int rtn = -1;
    if( d_map )
        CUDA_TRY(cudaMemcpy(&rtn,d_map+index,sizeof(int),cudaMemcpyDeviceToHost))
    return rtn;
}

//
int NVCategory::get_value(const char* str)
{
    char* d_str = nullptr;
    unsigned int bytes = 0;
    auto execpol = rmm::exec_policy(0);
    if( str )
    {
        bytes = (unsigned int)strlen(str);
        d_str = device_alloc<char>(bytes+1,0);
        CUDA_TRY(cudaMemcpy(d_str,str,bytes,cudaMemcpyHostToDevice))
    }
    int count = keys_size();
    custring_view_array d_strings = pImpl->getStringsPtr();

    // find string in this instance
    rmm::device_vector<int> keys(1,-1);
    thrust::copy_if( execpol->on(0), thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(count), keys.begin(),
         [d_strings, d_str, bytes] __device__ (int idx) {
             custring_view* dstr = d_strings[idx];
             if( (char*)dstr==d_str ) // only true if both are null
                 return true;
             return ( dstr && dstr->compare(d_str,bytes)==0 );
         } );
    //
    if( d_str )
        RMM_FREE(d_str,0);
    return keys[0];
}

std::pair<int,int> NVCategory::get_value_bounds(const char* str)
{
    std::pair<int,int> rtn(-1,-1);
    // first check if key exists (saves alot work below)
    int value = get_value(str);
    if( value>=0 )
    {
        rtn.first = value;
        rtn.second = value;
        return rtn;
    }
    // not found in existing keyset
    auto execpol = rmm::exec_policy(0);
    unsigned int count = keys_size();
    custring_view** d_strings = pImpl->getStringsPtr();
    // create index of the keys
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count+1);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), count,
        [d_strings, d_indexes] __device__(size_t idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
            {
                d_indexes[idx].first = (const char*)dstr->data();
                d_indexes[idx].second = (size_t)dstr->size();
            }
            else
            {
                d_indexes[idx].first = nullptr;
                d_indexes[idx].second = 0;
            }
        });
    //
    // and add the passed in string to the indexes
    size_t len = 0;
    char* d_str = nullptr;
    if( str )
    {
        len = strlen(str);
        d_str = device_alloc<char>(len+1,0);
        CUDA_TRY(cudaMemcpyAsync(d_str,str,len+1,cudaMemcpyHostToDevice))
    }
    thrust::pair<const char*,size_t> newstr(d_str,len); // add to the end
    CUDA_TRY(cudaMemcpyAsync(d_indexes+count,&newstr,sizeof(thrust::pair<const char*,size_t>),cudaMemcpyHostToDevice))

    // sort the keys with attached sequence numbers
    rmm::device_vector<int> seqdata(count+1);
    thrust::sequence(execpol->on(0),seqdata.begin(),seqdata.end()); // [0:count]
    int* d_seqdata = seqdata.data().get();
    thrust::sort_by_key(execpol->on(0), d_indexes, d_indexes+(count+1), d_seqdata,
        [] __device__( thrust::pair<const char*,size_t>& lhs, thrust::pair<const char*,size_t>& rhs ) {
            if( lhs.first==0 || rhs.first==0 )
                return rhs.first!=0;
            return custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second) < 0;
        });
    // now find the new position of the argument
    // this will be where the sequence number equals the count
    rmm::device_vector<int> keys(1,-1);
    thrust::copy_if( execpol->on(0), thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(count+1), keys.begin(),
        [d_seqdata, count, d_indexes] __device__ (int idx) { return d_seqdata[idx]==count; });
    //
    int first = 0; // get the position back into host memory
    CUDA_TRY(cudaMemcpy(&first,keys.data().get(),sizeof(int),cudaMemcpyDeviceToHost))
    rtn.first = first-1; // range is always
    rtn.second = first;  // position and previous one
    if( d_str )
        RMM_FREE(d_str,0);
    return rtn;
}

// return category values for all indexes
int NVCategory::get_values( int* results, bool bdevmem )
{
    int count = (int)size();
    int* d_map = pImpl->getMapPtr();
    if( count && d_map )
    {
        if( bdevmem )
            CUDA_TRY(cudaMemcpyAsync(results,d_map,count*sizeof(int),cudaMemcpyDeviceToDevice))
        else
            CUDA_TRY(cudaMemcpyAsync(results,d_map,count*sizeof(int),cudaMemcpyDeviceToHost))
    }
    return count;
}

const int* NVCategory::values_cptr()
{
    return pImpl->getMapPtr();
}

int NVCategory::get_indexes_for( unsigned int index, int* results, bool bdevmem )
{
    unsigned int count = size();
    if( index >= count )
        return -1;

    auto execpol = rmm::exec_policy(0);
    int* d_map = pImpl->getMapPtr();
    if( !d_map )
        return 0;
    int matches = thrust::count_if( execpol->on(0), d_map, d_map+count, [index] __device__(int idx) { return idx==(int)index; });
    if( matches <= 0 )
        return 0; // done, found nothing, not likely
    if( results==0 )
        return matches; // caller just wants the count

    int* d_results = results;
    if( !bdevmem )
        d_results = device_alloc<int>(matches,0);

    thrust::counting_iterator<unsigned int> itr(0);
    thrust::copy_if( execpol->on(0), itr, itr+count, d_results,
                     [index, d_map] __device__(unsigned int idx) { return d_map[idx]==(int)index; });
    //
    if( !bdevmem )
    {
        CUDA_TRY(cudaMemcpyAsync(results,d_results,matches*sizeof(int),cudaMemcpyDeviceToHost))
        RMM_FREE(d_results,0);
    }
    return matches;
}

int NVCategory::get_indexes_for( const char* str, int* results, bool bdevmem )
{
    int id = get_value(str);
    if( id < 0 )
        return id;
    return get_indexes_for((unsigned int)id, results, bdevmem);
}

// creates a new instance incorporating the new strings
NVCategory* NVCategory::add_strings(NVStrings& strs)
{
    // create one large index of both datasets
    unsigned int count1 = size();
    unsigned int count2 = strs.size();
    unsigned int count = count1 + count2;
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    create_index((std::pair<const char*,size_t>*)d_indexes,count1);
    strs.create_index((std::pair<const char*,size_t>*)d_indexes+count1,count2);
    // build the category from this new set
    return create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
}

// creates a new instance without the specified strings
// deprecated by remove_keys?
NVCategory* NVCategory::remove_strings(NVStrings& strs)
{
    auto execpol = rmm::exec_policy(0);
    unsigned int count = size();
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    create_index((std::pair<const char*,size_t>*)d_indexes,count);

    unsigned int delete_count = strs.size();
    rmm::device_vector< thrust::pair<const char*,size_t> > deletes(delete_count);
    thrust::pair<const char*,size_t>* d_deletes = deletes.data().get();
    strs.create_index((std::pair<const char*,size_t>*)d_deletes,delete_count);

    // this would be inefficient if strs is very large
    thrust::pair<const char*,size_t>* newend = thrust::remove_if(execpol->on(0), d_indexes, d_indexes + count,
        [d_deletes,delete_count] __device__ (thrust::pair<const char*,size_t> lhs) {
            for( unsigned int idx=0; idx < delete_count; ++idx )
            {
                thrust::pair<const char*,size_t> rhs = d_deletes[idx];
                if( lhs.first == rhs.first )
                    return true;
                if( lhs.second != rhs.second )
                    continue;
                if( custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second)==0 )
                    return true;
            }
            return false;
        });
    // return value ensures a dev-sync has already been performed by thrust
    count = (unsigned int)(newend - d_indexes); // new count of strings
    // build the category from this new set
    return create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
}

// basically recreates the original string list
NVStrings* NVCategory::to_strings()
{
    int count = (int)size();
    int* d_map = pImpl->getMapPtr();
    if( count==0 || d_map==0 )
        return nullptr;
    custring_view** d_strings = pImpl->getStringsPtr();
    // use the map to build the indexes array
    auto execpol = rmm::exec_policy(0);
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), count,
        [d_strings, d_map, d_indexes] __device__(size_t idx){
            int stridx = d_map[idx];
            custring_view* dstr = nullptr;
            if( stridx >=0 )
                dstr = d_strings[stridx];
            if( dstr )
            {
                d_indexes[idx].first = (const char*)dstr->data();
                d_indexes[idx].second = (size_t)dstr->size();
            }
            else
            {
                d_indexes[idx].first = nullptr;
                d_indexes[idx].second = 0;
            }
        });
    //
    // create strings from index
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
}

// creates a new NVStrings instance using the specified index values
NVStrings* NVCategory::gather_strings( const int* pos, unsigned int count, bool bdevmem )
{
    auto execpol = rmm::exec_policy(0);
    const int* d_pos = pos;
    if( !bdevmem )
    {
        d_pos = const_cast<const int*>(device_alloc<int>(count,0));
        CUDA_TRY(cudaMemcpy((void*)d_pos,pos,count*sizeof(int),cudaMemcpyHostToDevice))
    }

    custring_view** d_strings = pImpl->getStringsPtr();
    // need to check for invalid values
    unsigned int size = keys_size();
    rmm::device_vector<int> check(count,0);
    int* d_check = check.data().get();
    // use the map to build the indexes array
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), count,
        [d_strings, d_pos, size, d_check, d_indexes] __device__(size_t idx){
            int stridx = d_pos[idx];
            if( (stridx < 0) || (stridx >= size) )
            {
                d_check[idx] = 1;
                return;
            }
            custring_view* dstr = nullptr;
            if( stridx >=0 )
                dstr = d_strings[stridx];
            if( dstr )
            {
                d_indexes[idx].first = (const char*)dstr->data();
                d_indexes[idx].second = (size_t)dstr->size();
            }
            else
            {
                d_indexes[idx].first = nullptr;
                d_indexes[idx].second = 0;
            }
        });
    //
    if( !bdevmem )
        RMM_FREE((void*)d_pos,0);
    int invalidcount = thrust::reduce( execpol->on(0), d_check, d_check+count );
    if( invalidcount )
        throw std::out_of_range("");
    // create strings from index
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
}

//
// Create a new category by gathering strings from this category.
// If specific keys are not referenced, the values are remapped.
// This is a shortcut method to calling gather_strings() and then
// just converting the resulting NVStrings instance into an new NVCategory.
//
// Example category
//    category :---------
//    | strs        key  |
//    | abbfcf  ->  abcf |
//    | 012345      0123 |
//    | 011323    <-'    |
//     ------------------
//
// Specify strings in pos parameter:
//  v = 1 3 2 3 1 2   bfcfbc
//  x = 0 1 1 1       x[v[idx]] = 1  (set 1 for values in v)
//  y = 0 0 1 2       excl-scan(x)
//
// Remap values using:
//  v[idx] = y[v[idx]]  -> 021201
// New key list is copy_if of keys where x==1  -> bcf
//
NVCategory* NVCategory::gather_and_remap( const int* pos, unsigned int count, bool bdevmem )
{
    auto execpol = rmm::exec_policy(0);
    const int* d_v = pos;
    if( !bdevmem )
    {
        d_v = const_cast<const int*>(device_alloc<int>(count,0));
        CUDA_TRY(cudaMemcpyAsync((void*)d_v,pos,count*sizeof(int),cudaMemcpyHostToDevice))
    }

    unsigned int kcount = keys_size();
    // first, do bounds check on input values
    int invalidcount = thrust::count_if(execpol->on(0), d_v, d_v+count,
        [kcount] __device__ (int v) { return ((v < 0) || (v >= kcount)); } );
    if( invalidcount )
    {
        if( !bdevmem )
            RMM_FREE((void*)d_v,0);
        throw std::out_of_range("");
    }

    // build x vector which has 1s for each value in v
    rmm::device_vector<int> x(kcount,0);
    int* d_x = x.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), count,
        [d_v, d_x] __device__ (unsigned int idx) { d_x[d_v[idx]] = 1; });
    // y vector is scan of x values
    rmm::device_vector<int> y(kcount,0);
    int* d_y = y.data().get();
    thrust::exclusive_scan(execpol->on(0),d_x,d_x+kcount,d_y,0);
    // use y to map input to new values
    rmm::device_vector<int>* pNewMap = new rmm::device_vector<int>(count,0);
    int* d_map = pNewMap->data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<size_t>(0), count,
        [d_v, d_y, d_map] __device__ (unsigned int idx) { d_map[idx] = d_y[d_v[idx]]; });
    // done creating the map
    NVCategory* rtn = new NVCategory;
    rtn->pImpl->pMap = pNewMap;
    // copy/gather the keys
    custring_view_array d_keys = pImpl->getStringsPtr();
    unsigned int ucount = kcount;
    {   // reuse the y vector for gather positions
        thrust::counting_iterator<int> citr(0);
        auto nend = thrust::copy_if( execpol->on(0), citr, citr + kcount, d_y, [d_x] __device__ (const int& idx) { return d_x[idx]==1; });
        ucount = (unsigned int)(nend - d_y); // how many were copied
    }
    // gather keys into new vector
    rmm::device_vector<custring_view*> newkeys(ucount,nullptr);
    thrust::gather( execpol->on(0), d_y, d_y + ucount, d_keys, newkeys.data().get() );
    // build keylist for new category
    NVCategoryImpl_keys_from_custringarray(rtn->pImpl,newkeys.data().get(),ucount);
    //
    if( !bdevmem )
        RMM_FREE((void*)d_v,0);
    return rtn;
}

// this method simply copies the keys and the passed in values to create a new category instance
NVCategory* NVCategory::gather( const int* pos, unsigned int count, bool bdevmem )
{
    unsigned int kcount = keys_size();
    NVCategory* rtn = new NVCategory;
    auto execpol = rmm::exec_policy(0);
    if( count )
    {
        auto pMap = new rmm::device_vector<int>(count,0);
        auto d_pos = pMap->data().get();
        if( bdevmem )
            CUDA_TRY(cudaMemcpyAsync(d_pos,pos,count*sizeof(int),cudaMemcpyDeviceToDevice))
        else
            CUDA_TRY(cudaMemcpyAsync(d_pos,pos,count*sizeof(int),cudaMemcpyHostToDevice))
        // first, do bounds check on input values; also -1 is allowed
        // need to re-evaluate if this check is really necessary here
        int invalidcount = thrust::count_if(execpol->on(0), d_pos, d_pos+count,
            [kcount] __device__ (int v) { return ((v < -1) || (v >= kcount)); } );
        if( invalidcount )
        {
            delete pMap;
            delete rtn;
            throw std::out_of_range("");
        }
        rtn->pImpl->pMap = pMap;
    }

    custring_view_array d_keys = pImpl->getStringsPtr();
    NVCategoryImpl_keys_from_custringarray(rtn->pImpl,d_keys,kcount);
    //
    return rtn;
}

//
// Merge two categories and maintain the values and key positions of the this category.
// Very complicated to avoid looping over values or keys from either category.
//
// Example shows logic used:
//
//   category1:---------           category2:---------
//   | strs1       key1 |          | strs2       key2 |
//   | abbfcf  ->  abcf |          | aadcce  ->  acde |
//   | 012345      0123 |          | 012345      0123 |
//   | 011323    <-'    |          | 002113    <-'    |
//    ------------------            ------------------
//
//   merge/append should result in new category:
//    strs              key
//    abbfcfaadcce  ->  abcfde
//                      012345
//    011323004225    <-'
//
// 1. build vector of all the keys and seq vector (x) and diff vector (y)
//    concat keys key2,key1 (w);    stable-sort-by-key(w,x) and
//    create neg,pos sequence (x)   build diff vector (y)
//     a  c  d  e   a  b  c  f  ->  w =  a  a  b  c  c  d  e  f
//     0  1  2  3  -1 -2 -3 -4      x =  0 -1 -2  1 -3  2  3 -4
//                                  y =  1  0  0  1  0  0  0  0
//
// 2. compute keys diff using w,x,y:
//    copy-if/gather(w)  (x>=0) && (y==0)  -->  d e
//    reduce(y) = 2  -> how many keys matched
//    new key:  abcf + de = abcfde
//
//          a  b  c  d  e  f  :unique-by-key(w,x)
//   ubl =  0 -2  1  2  3 -4  :x = unique val--^
//   sws =  0  1  2  4  5  3  :sort new key (abcfde) with seq (012345)
//
// 3. gather new indexes for map2
//    copy-if/gather(sws)  where ubl>=0  -->  0 2 4 5   (remap)
//    remove-if(ubl)  ubl<0   --> 0 1 2 3
//    sort(ubl,remap)         --> 0 2 4 5
//    new map2 values gathered using original map to remap values:
//    002113 -> 004225
//
//   result:
//    new key: abcfde
//    new map: 011323004225
//             abbfcfaadcce
//
// The end result is not guaranteed to be a sorted keyset.
//
NVCategory* NVCategory::merge_category(NVCategory& cat2)
{
    unsigned int count1 = keys_size();
    unsigned int mcount1 = size();
    unsigned int count2 = cat2.keys_size();
    unsigned int mcount2 = cat2.size();
    NVCategory* rtn = new NVCategory();
    if( (count1==0) && (count2==0) )
        return rtn;
    unsigned int count12 = count1 + count2;
    unsigned int mcount = mcount1 + mcount2;
    // if either category is empty, just copy the non-empty one
    if( (count1==0) || (count2==0) )
    {
        NVCategory* dcat = ((count1==0) ? &cat2 : this);
        return dcat->copy();
    }
    auto execpol = rmm::exec_policy(0);
    // both this cat and cat2 are non-empty
    // init working vars
    custring_view_array d_keys1 = pImpl->getStringsPtr();
    int* d_map1 = pImpl->getMapPtr();
    custring_view_array d_keys2 = cat2.pImpl->getStringsPtr();
    int* d_map2 = cat2.pImpl->getMapPtr();
    // create some vectors we can sort
    rmm::device_vector<custring_view*> wstrs(count12); // w = keys2 + keys1
    custring_view_array d_w = wstrs.data().get();
    thrust::copy( execpol->on(0), d_keys2, d_keys2+count2, d_w );
    //cudaMemcpy(d_w, d_keys2, count2*sizeof(custring_view*),cudaMemcpyDeviceToDevice);
    thrust::copy( execpol->on(0), d_keys1, d_keys1+count1, d_w+count2);
    //cudaMemcpy(d_w+count2, d_keys1, count1*sizeof(custring_view*),cudaMemcpyDeviceToDevice);
    rmm::device_vector<int> x(count12);  // 0,1,....count2,-1,...,-count1
    int* d_x = x.data().get();
    // sequence and for-each-n could be combined into for-each-n logic
    thrust::sequence( execpol->on(0), d_x, d_x+count2 );   // first half is 0...count2
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<int>(0), count1,
        [d_x, count2] __device__ (int idx) { d_x[idx+count2]= -idx-1; }); // 2nd half is -1...-count1
    thrust::stable_sort_by_key( execpol->on(0), d_w, d_w + count12, d_x,  // preserves order for
        [] __device__ (custring_view*& lhs, custring_view*& rhs) {        // strings that match
            return ((lhs && rhs) ? (lhs->compare(*rhs)<0) : (rhs!=0));
        });
    rmm::device_vector<int> y(count12,0); // y-vector will identify overlapped keys
    int* d_y = y.data().get();
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(0), (count12-1),
        [d_y, d_w] __device__ (int idx) {
            custring_view* lhs = d_w[idx];
            custring_view* rhs = d_w[idx+1];
            if( lhs && rhs )
                d_y[idx] = (int)(lhs->compare(*rhs)==0);
            else
                d_y[idx] = (int)(lhs==rhs);
        });
    int matched = thrust::reduce( execpol->on(0), d_y, d_y + count12 ); // how many keys matched
    unsigned int ncount = count2 - (unsigned int)matched; // new keys count
    unsigned int ucount = count1 + ncount; // total unique keys count
    rmm::device_vector<custring_view*> keys(ucount,nullptr);
    custring_view_array d_keys = keys.data().get(); // this will hold the merged keyset
    rmm::device_vector<int> nidxs(ucount); // needed for various gather methods below
    int* d_nidxs = nidxs.data().get(); // indexes of 'new' keys from key2 not in key1
    {
        thrust::counting_iterator<int> citr(0);
        thrust::copy_if( execpol->on(0), citr, citr + (count12), d_nidxs,
            [d_x, d_y] __device__ (const int& idx) { return (d_x[idx]>=0) && (d_y[idx]==0); });
    }
    // first half of merged keyset is direct copy of key1
    //cudaMemcpy( d_keys, d_keys1, count1*sizeof(custring_view*), cudaMemcpyDeviceToDevice);
    thrust::copy( execpol->on(0), d_keys1, d_keys1+count1, d_keys );
    // append the 'new' keys from key2: extract them from w as identified by nidxs
    thrust::gather( execpol->on(0), d_nidxs, d_nidxs + ncount, d_w, d_keys + count1 );
    int* d_ubl = d_x; // reuse d_x for unique-bias-left values
    thrust::unique_by_key( execpol->on(0), d_w, d_w + count12, d_ubl,
         [] __device__ (custring_view* lhs, custring_view* rhs) {
            return ((lhs && rhs) ? (lhs->compare(*rhs)==0) : (lhs==rhs));
         });  // ubl now contains new index values for key2
    int* d_sws = d_y; // reuse d_y for sort-with-seq values
    thrust::sequence( execpol->on(0), d_sws, d_sws + ucount); // need to assign new index values
    rmm::device_vector<custring_view*> keySort(ucount);    // for all the original key2 values
    //cudaMemcpy( keySort.data().get(), d_keys, ucount * sizeof(custring_view*), cudaMemcpyDeviceToDevice);
    thrust::copy( execpol->on(0), d_keys, d_keys+ucount, keySort.begin());
    thrust::sort_by_key( execpol->on(0), keySort.begin(), keySort.end(), d_sws,
        [] __device__ (custring_view*& lhs, custring_view*& rhs ) {
            return ((lhs && rhs) ? (lhs->compare(*rhs)<0) : (rhs!=0));
        }); // sws is now key index values for the new keyset
    //printDeviceInts("d_sws",d_sws,ucount);
    {
        thrust::counting_iterator<int> citr(0); // generate subset of just the key2 values
        thrust::copy_if( execpol->on(0), citr, citr + ucount, d_nidxs, [d_ubl] __device__ (const int& idx) { return d_ubl[idx]>=0; });
    }
    // nidxs has the indexes to the key2 values in the new keyset but they are sorted when key2 may not have been
    rmm::device_vector<int> remap2(count2); // need to remap the indexes to the original positions
    int* d_remap2 = remap2.data().get();       // do this by de-sorting the key2 values from the full keyset
    thrust::gather( execpol->on(0), d_nidxs, d_nidxs + count2, d_sws, d_remap2 ); // here grab new positions for key2
    // first, remove the key1 indexes from the sorted sequence values; ubl will then have only key2 orig. pos values
    thrust::remove_if( execpol->on(0), d_ubl, d_ubl + ucount, [] __device__ (int v) { return v<0; });
    thrust::sort_by_key( execpol->on(0), d_ubl, d_ubl+count2, d_remap2 ); // does a de-sort of key2 only
    // build new map
    rmm::device_vector<int>* pNewMap = new rmm::device_vector<int>(mcount,0);
    int* d_map = pNewMap->data().get(); // first half is identical to map1
    //cudaMemcpy( d_map, d_map1, mcount1 * sizeof(int), cudaMemcpyDeviceToDevice);
    thrust::copy( execpol->on(0), d_map1, d_map1 + mcount1, d_map);
    //cudaMemcpy( d_map+mcount1, d_map2, mcount2 * sizeof(int), cudaMemcpyDeviceToDevice);
    thrust::copy( execpol->on(0), d_map2, d_map2 + mcount2, d_map + mcount1 );
    // remap map2 values to their new positions in the full keyset
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(mcount1), mcount2,
        [d_map, d_remap2] __device__ (int idx) {
            int v = d_map[idx];
            if( v >= 0 )
                d_map[idx] = d_remap2[v];
        });
    //
    NVCategoryImpl_keys_from_custringarray(rtn->pImpl,d_keys,ucount);
    rtn->pImpl->pMap = pNewMap;
    return rtn;
}

// see create_from_categories method above for logic details
NVCategory* NVCategory::merge_and_remap(NVCategory& cat2)
{
    std::vector<NVCategory*> cats;
    cats.push_back(this);
    cats.push_back(&cat2);
    return create_from_categories(cats);
}

//
// Creates a new instance adding the specified strings as keys and remapping the values.
// Pandas maintains the original position values. This function does a remap.
//
// Example:
//
//    category :---------
//    | strs        key  |
//    | abbfcf  ->  abcf |
//    | 012345      0123 |
//    | 011323    <-'    |
//     ------------------
//
//    new keys:  abcd
//    duplicate keys can be ignored; new keyset may shift the indexes
//
//     a  b  c  f  :  a  b  c  d  -> w =  a  a  b  b  c  c  d  f
//     0  1  2  3    -1 -2 -3 -4     x =  0 -1  1 -2  2 -3 -4  3
//
//                                  u  =  a  b  c  d  f
//                                  ux =  0  1  2 -4  3
//
//   values map:  a b c f
//                0 1 2 4
//
//   new values:  a  b  b  f  c  f
//                0  1  1  4  2  4
//
NVCategory* NVCategory::add_keys_and_remap(NVStrings& strs)
{
    unsigned int kcount = keys_size();
    unsigned int mcount = size();
    unsigned int count = strs.size();
    if( (kcount==0) && (count==0) )
        return new NVCategory;
    if( count==0 )
        return copy();
    auto execpol = rmm::exec_policy(0);
    // get the keys from the argument
    rmm::device_vector<custring_view*> addKeys(count,nullptr);
    custring_view_array d_addKeys = addKeys.data().get();
    strs.create_custring_index(d_addKeys);
    NVCategory* rtn = new NVCategory;
    if( kcount==0 )
    {
        // just take the keys; values are not effected
        // need to sort and unique them
        thrust::sort(execpol->on(0), d_addKeys, d_addKeys + count, [] __device__( custring_view*& lhs, custring_view*& rhs ) { return ( (lhs && rhs) ? (lhs->compare(*rhs)<0) : (rhs!=0) ); });
        // now remove duplicates from string list
        auto nend = thrust::unique(execpol->on(0), d_addKeys, d_addKeys + count, [] __device__ (custring_view* lhs, custring_view* rhs) { return ((lhs && rhs) ? (lhs->compare(*rhs)==0) : (lhs==rhs)); });
        unsigned int ucount = nend - d_addKeys;
        NVCategoryImpl_keys_from_custringarray(rtn->pImpl,d_addKeys,ucount);
        // copy the values
        if( mcount )
        {
            rtn->pImpl->pMap = new rmm::device_vector<int>(mcount,0);
            //cudaMemcpy(rtn->pImpl->getMapPtr(),pImpl->getMapPtr(),mcount*sizeof(int),cudaMemcpyDeviceToDevice);
            thrust::copy(execpol->on(0), pImpl->pMap->begin(), pImpl->pMap->end(), rtn->pImpl->pMap->begin());
        }
        return rtn;
    }
    // both kcount and count are non-zero
    custring_view_array d_keys = pImpl->getStringsPtr();
    int akcount = kcount + count;
    rmm::device_vector<custring_view*> wstrs(akcount);
    custring_view_array d_w = wstrs.data().get();
    //cudaMemcpy(d_w, d_keys, kcount*sizeof(custring_view*),cudaMemcpyDeviceToDevice);
    thrust::copy( execpol->on(0), d_keys, d_keys + kcount, d_w);
    //cudaMemcpy(d_w+kcount, d_addKeys, count*sizeof(custring_view*),cudaMemcpyDeviceToDevice);
    thrust::copy( execpol->on(0), d_addKeys, d_addKeys+count, d_w+kcount );
    rmm::device_vector<int> x(akcount);  // values arranged like 0,...,(kcount-1),-1,...,-count
    int* d_x = x.data().get();
    // sequence and for-each-n could be combined into single for-each-n logic
    thrust::sequence( execpol->on(0), d_x, d_x + kcount );   // first half is [0:kcount)
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<int>(0), count, [d_x, kcount] __device__ (int idx) { d_x[idx+kcount]= -idx-1; }); // 2nd half is [-1:-count]
    // stable-sort preserves order for strings that match
    thrust::stable_sort_by_key( execpol->on(0), d_w, d_w + akcount, d_x, [] __device__ (custring_view*& lhs, custring_view*& rhs) { return ((lhs && rhs) ? (lhs->compare(*rhs)<0) : (rhs!=0)); });
    auto nend = thrust::unique_by_key( execpol->on(0), d_w, d_w + akcount, d_x, [] __device__ (custring_view* lhs, custring_view* rhs) { return ((lhs && rhs) ? (lhs->compare(*rhs)==0) : (lhs==rhs)); });
    int ucount = nend.second - d_x;
    // d_w,ucount are now the keys
    NVCategoryImpl_keys_from_custringarray(rtn->pImpl,d_w,ucount);
    // remapping the values
    rmm::device_vector<int> y(kcount,-1);
    int* d_y = y.data().get();
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(0), ucount,
        [d_y, d_x] __device__ (int idx) {
            int u = d_x[idx];
            if( u >= 0 )
                d_y[u] = idx;
        });
    // allocate and fill new map
    int* d_map = pImpl->getMapPtr();
    if( mcount && d_map )
    {
        rmm::device_vector<int>* pNewMap = new rmm::device_vector<int>(mcount,-1);
        int* d_newmap = pNewMap->data().get();
        thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(0), mcount,
            [d_map, d_y, d_newmap] __device__ (int idx) {
                int v = d_map[idx];
                d_newmap[idx] = (v < 0 ? v : d_y[v]);
            });
        //
        rtn->pImpl->pMap = pNewMap;
    }
    return rtn;
}

// Creates a new instance removing the keys matching the specified strings and remapping the values.
// Pandas maintains the original position values. Below does a remap.
//
//    category :---------
//    | strs        key  |
//    | abbfcf  ->  abcf |
//    | 012345      0123 |
//    | 011323    <-'    |
//     ------------------
//
//    remove keys:  bd
//    unknown keys can be ignored; new keyset may shift the indexes
//
//     a  b  c  f  :  b  d  -> w =  a  b  b  c  d  f
//     0  1  2  3    -1 -2     x =  0  1 -1  2 -2  3
//                             y =  0  1  0  0  0  0
//
//   remove keys:  x<0 || y==1 : b d
//                            u  =  a  c  f
//                            ux =  0  2  3
//
//   values map:  a  b  c  f
//                0 -1  1  2
//
//   new values:  a  b  b  f  c  f
//                0 -1 -1  2  1  2
//
//
NVCategory* NVCategory::remove_keys_and_remap(NVStrings& strs)
{
    unsigned int kcount = keys_size();
    unsigned int count = strs.size();
    if( kcount==0 || count==0 )
        return copy();
    // both kcount and count are non-zero
    auto execpol = rmm::exec_policy(0);
    // get the keys from the parameter
    rmm::device_vector<custring_view*> removeKeys(count,nullptr);
    custring_view_array d_removeKeys = removeKeys.data().get();
    strs.create_custring_index(d_removeKeys);
    // keys for this instance
    custring_view_array d_keys = pImpl->getStringsPtr();
    // combine the keys into one set to be evaluated
    int akcount = kcount + count;
    rmm::device_vector<custring_view*> wstrs(akcount);
    custring_view_array d_w = wstrs.data().get();
    //cudaMemcpy(d_w, d_keys, kcount*sizeof(custring_view*),cudaMemcpyDeviceToDevice);
    thrust::copy( execpol->on(0), d_keys, d_keys+kcount, d_w);
    //cudaMemcpy(d_w+kcount, d_removeKeys, count*sizeof(custring_view*),cudaMemcpyDeviceToDevice);
    thrust::copy( execpol->on(0), d_removeKeys, d_removeKeys+count, d_w+kcount );
    rmm::device_vector<int> x(akcount);  // 0,1,...,kcount,-1,...,-count
    int* d_x = x.data().get();
    // sequence and for-each-n could be combined into single for-each-n logic
    thrust::sequence( execpol->on(0), d_x, d_x + kcount );   // [0:kcount)
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<int>(0), count, [d_x, kcount] __device__ (int idx) { d_x[idx+kcount]= -idx-1; }); // 2nd half is [-1:-count]
    // stable-sort preserves order for strings that match
    thrust::stable_sort_by_key( execpol->on(0), d_w, d_w + akcount, d_x, [] __device__ (custring_view*& lhs, custring_view*& rhs) { return ((lhs && rhs) ? (lhs->compare(*rhs)<0) : (rhs!=0)); });
    rmm::device_vector<int> y(akcount,0); // matches resulting from
    int* d_y = y.data().get();            // sort are marked with '1'
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(0), (akcount-1),
        [d_y, d_w] __device__ (int idx) {
            custring_view* lhs = d_w[idx];
            custring_view* rhs = d_w[idx+1];
            if( lhs && rhs )
                d_y[idx] = (int)(lhs->compare(*rhs)==0);
            else
                d_y[idx] = (int)(lhs==rhs);
        });
    //
    int cpcount = akcount; // how many keys copied
    {   // scoping to get rid of temporary memory sooner
        rmm::device_vector<int> nidxs(akcount); // needed for gather
        int* d_nidxs = nidxs.data().get(); // indexes of keys from key1 not in key2
        thrust::counting_iterator<int> citr(0);
        int* nend = thrust::copy_if( execpol->on(0), citr, citr + (akcount), d_nidxs,
            [d_x, d_y] __device__ (const int& idx) { return (d_x[idx]>=0) && (d_y[idx]==0); });
        cpcount = nend - d_nidxs;
        // the gather()s here will select the remaining keys
        rmm::device_vector<custring_view*> wstrs2(cpcount);
        rmm::device_vector<int> x2(cpcount);
        thrust::gather( execpol->on(0), d_nidxs, d_nidxs + cpcount, wstrs.begin(), wstrs2.begin() );
        thrust::gather( execpol->on(0), d_nidxs, d_nidxs + cpcount, x.begin(), x2.begin() );
        wstrs.swap(wstrs2);
        d_w = wstrs.data().get();
        x.swap(x2);
        d_x = x.data().get();
    }
    NVCategory* rtn = new NVCategory;
    int ucount = cpcount; // final number of unique keys
    NVCategoryImpl_keys_from_custringarray(rtn->pImpl,d_w,ucount); // and d_w are those keys
    // now remap the values: positive values in d_x are [0:ucount)
    thrust::fill( execpol->on(0), d_y, d_y + kcount, -1);
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(0), ucount,
        [d_y, d_x] __device__ (int idx) { d_y[d_x[idx]] = idx; });
    unsigned int mcount = size();
    int* d_map = pImpl->getMapPtr();
    if( mcount && d_map )
    {
        rmm::device_vector<int>* pNewMap = new rmm::device_vector<int>(mcount,0);
        int* d_newmap = pNewMap->data().get(); // new map will go here
        thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(0), mcount,
            [d_map, d_y, d_newmap] __device__ (int idx) {
                int v = d_map[idx];                  // get old index
                d_newmap[idx] = ( v < 0 ? v : d_y[v]);  // set new index (may be negative)
            });
        //
        rtn->pImpl->pMap = pNewMap;
    }
    return rtn;
}

// keys that are not represented in the list of values are removed
// this may cause the values to be remapped if the keys positions are moved
NVCategory* NVCategory::remove_unused_keys_and_remap()
{
    unsigned int kcount = keys_size();
    unsigned int mcount = size();
    if( kcount==0 )
        return copy();
    // both kcount and count are non-zero
    auto execpol = rmm::exec_policy(0);
    // keys for this instance
    custring_view_array d_keys = pImpl->getStringsPtr();
    int* d_map = pImpl->getMapPtr();
    rmm::device_vector<unsigned int> usedkeys(kcount,0);
    unsigned int* d_usedkeys = usedkeys.data().get();
    // find the keys that not being used
    unsigned int count = 0;
    if( d_map )
    {
        thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(0), mcount,
            [d_map, d_usedkeys] __device__ (int idx) {
                int pos = d_map[idx];
                if( pos >= 0 )
                    d_usedkeys[pos] = 1; // race condition not important
            });
        // compute how many are not used
        count = kcount - thrust::reduce(execpol->on(0),d_usedkeys,d_usedkeys+kcount,(unsigned int)0);
    }
    if( count==0 )
        return copy();
    // gather the unused keys
    rmm::device_vector<custring_view*> removeKeys(count,nullptr);
    custring_view_array d_removeKeys = removeKeys.data().get();
    {
        rmm::device_vector<int> nidxs(count);
        int* d_nidxs = nidxs.data().get();
        thrust::counting_iterator<int> citr(0);
        thrust::copy_if( execpol->on(0), citr, citr + kcount, d_nidxs,
            [d_usedkeys] __device__ (const int& idx) { return (d_usedkeys[idx]==0); });
        thrust::gather( execpol->on(0), d_nidxs, d_nidxs + count, d_keys, d_removeKeys );
    }
    // the remainder is common with remove_keys_and_remap
    // --------------------------------------------------
    // combine the keys into one set to be evaluated
    int akcount = kcount + count;
    rmm::device_vector<custring_view*> wstrs(akcount);
    custring_view_array d_w = wstrs.data().get();
    //cudaMemcpy(d_w, d_keys, kcount*sizeof(custring_view*),cudaMemcpyDeviceToDevice);
    thrust::copy( execpol->on(0), d_keys, d_keys+kcount, d_w);
    //cudaMemcpy(d_w+kcount, d_removeKeys, count*sizeof(custring_view*),cudaMemcpyDeviceToDevice);
    thrust::copy( execpol->on(0), d_removeKeys, d_removeKeys+count, d_w+kcount);
    rmm::device_vector<int> x(akcount);  // 0,1,...,kcount,-1,...,-count
    int* d_x = x.data().get();
    // sequence and for-each-n could be combined into single for-each-n logic
    thrust::sequence( execpol->on(0), d_x, d_x + kcount );   // [0:kcount)
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<int>(0), count, [d_x, kcount] __device__ (int idx) { d_x[idx+kcount]= -idx-1; }); // 2nd half is [-1:-count]
    // stable-sort preserves order for strings that match
    thrust::stable_sort_by_key( execpol->on(0), d_w, d_w + akcount, d_x, [] __device__ (custring_view*& lhs, custring_view*& rhs) { return ((lhs && rhs) ? (lhs->compare(*rhs)<0) : (rhs!=0)); });
    rmm::device_vector<int> y(akcount,0); // matches resulting from
    int* d_y = y.data().get();            // sort are marked with '1'
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(0), (akcount-1),
        [d_y, d_w] __device__ (int idx) {
            custring_view* lhs = d_w[idx];
            custring_view* rhs = d_w[idx+1];
            if( lhs && rhs )
                d_y[idx] = (int)(lhs->compare(*rhs)==0);
            else
                d_y[idx] = (int)(lhs==rhs);
        });
    int cpcount = akcount; // how many keys copied
    {   // scoping to get rid of temporary memory sooner
        rmm::device_vector<int> nidxs(akcount); // needed for gather
        int* d_nidxs = nidxs.data().get(); // indexes of keys from key1 not in key2
        thrust::counting_iterator<int> citr(0);
        int* nend = thrust::copy_if( execpol->on(0), citr, citr + (akcount), d_nidxs,
            [d_x, d_y] __device__ (const int& idx) { return (d_x[idx]>=0) && (d_y[idx]==0); });
        cpcount = nend - d_nidxs;
        // the gather()s here will select the remaining keys
        rmm::device_vector<custring_view*> wstrs2(cpcount);
        rmm::device_vector<int> x2(cpcount);
        thrust::gather( execpol->on(0), d_nidxs, d_nidxs + cpcount, wstrs.begin(), wstrs2.begin() );
        thrust::gather( execpol->on(0), d_nidxs, d_nidxs + cpcount, x.begin(), x2.begin() );
        wstrs.swap(wstrs2);
        d_w = wstrs.data().get();
        x.swap(x2);
        d_x = x.data().get();
    }
    NVCategory* rtn = new NVCategory;
    int ucount = cpcount; // final number of unique keys
    NVCategoryImpl_keys_from_custringarray(rtn->pImpl,d_w,ucount); // and d_w are those keys
    // now remap the values: positive values in d_x are [0:ucount)
    thrust::fill( execpol->on(0), d_y, d_y + kcount, -1);
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(0), ucount,
        [d_y, d_x] __device__ (int idx) { d_y[d_x[idx]] = idx; });
    rmm::device_vector<int>* pNewMap = new rmm::device_vector<int>(mcount,0);
    int* d_newmap = pNewMap->data().get(); // new map will go here
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(0), mcount,
        [d_map, d_y, d_newmap] __device__ (int idx) {
            int v = d_map[idx];                     // get old index
            d_newmap[idx] = ( v < 0 ? v : d_y[v]);  // set new index (may be negative)
        });
    //
    rtn->pImpl->pMap = pNewMap;
    return rtn;
}


// Creates a new instance using the specified strings as keys causing add/remove as appropriate.
// Values are also remapped.
//
// Example:
//   category :---------
//   | strs        key  |
//   | abbfcf  ->  abcf |
//   | 012345      0123 |
//   | 011323    <-'    |
//    ------------------
//   new keyset: bcde
//               0123
//   new values: a  b  b  f  c  f
//              -1  0  0 -1  1 -1
//
// Logic:
//  a  b  c  f  :  b  c  e  d  -> w =  a  b  b  c  c  d  e  f
//  0  1  2  3    -1 -2 -3 -4     x =  0  1 -1  2 -2 -4 -3  3
//                                y =  0  1  0  1  0  0  0  0
//
// remove keys:  x>=0 && y==0 : a f -> [0,3] -> [-1,-1]
//                               w' =  b  b  c  c  d  e
//                               x' =  1 -1  2 -2 -4 -3
//                               u  =  b  c  d  e
//                                     1  2 -4 -3
//
// need map to set values like:   0  1  2  3
//                               -1  0  1 -1
//
// so create map using:
//   m[]=-1   -init all to -1; we don't need to worry about removed keys
//   if(u[idx]>=0): m[u[idx]]=idx
//
// and create new values using:
//   v[idx] = m[v[idx]]    -- make sure v[idx]>=0
//
NVCategory* NVCategory::set_keys_and_remap(NVStrings& strs)
{
    unsigned int kcount = keys_size();
    unsigned int mcount = size();
    unsigned int count = strs.size();
    NVCategory* rtn = new NVCategory;
    if( (kcount==0) && (count==0) )
        return rtn;
    if( count==0 )
    {
        rtn->pImpl->pMap = new rmm::device_vector<int>(mcount,-1);
        return rtn;
    }
    auto execpol = rmm::exec_policy(0);
    // get the keys
    rmm::device_vector<custring_view*> newKeys(count,nullptr);
    custring_view_array d_newKeys = newKeys.data().get();
    strs.create_custring_index(d_newKeys);
    if( kcount==0 )
    {
        // just take the new keys
        thrust::sort(execpol->on(0), d_newKeys, d_newKeys + count, [] __device__( custring_view*& lhs, custring_view*& rhs ) { return ( (lhs && rhs) ? (lhs->compare(*rhs)<0) : (rhs!=0) ); });
        // now remove duplicates from string list
        auto nend = thrust::unique(execpol->on(0), d_newKeys, d_newKeys + count, [] __device__ (custring_view* lhs, custring_view* rhs) { return ((lhs && rhs) ? (lhs->compare(*rhs)==0) : (lhs==rhs)); });
        unsigned int ucount = nend - d_newKeys;
        NVCategoryImpl_keys_from_custringarray(rtn->pImpl,d_newKeys,ucount);
        // copy the values
        if( mcount )
        {
            rtn->pImpl->pMap = new rmm::device_vector<int>(mcount,0);
            //cudaMemcpy(rtn->pImpl->getMapPtr(),pImpl->getMapPtr(),mcount*sizeof(int),cudaMemcpyDeviceToDevice);
            thrust::copy( execpol->on(0), pImpl->pMap->begin(), pImpl->pMap->end(), rtn->pImpl->pMap->begin());
        }
        return rtn;
    }
    // both kcount and count are non-zero
    custring_view_array d_keys = pImpl->getStringsPtr();
    // combine the keys into single array
    int akcount = kcount + count;
    rmm::device_vector<custring_view*> wstrs(akcount);
    custring_view_array d_w = wstrs.data().get();
    //cudaMemcpy(d_w, d_keys, kcount*sizeof(custring_view*),cudaMemcpyDeviceToDevice);
    thrust::copy( execpol->on(0), d_keys, d_keys+kcount, d_w);
    //cudaMemcpy(d_w+kcount, d_newKeys, count*sizeof(custring_view*),cudaMemcpyDeviceToDevice);
    thrust::copy( execpol->on(0), d_newKeys, d_newKeys+count, d_w+kcount);
    rmm::device_vector<int> x(akcount);  // 0,...,(kcount-),-1,...,-count
    int* d_x = x.data().get();
    // sequence and for-each-n could be combined into single for-each-n logic
    thrust::sequence( execpol->on(0), d_x, d_x + kcount );   // first half is [0:kcount)
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<int>(0), count, [d_x, kcount] __device__ (int idx) { d_x[idx+kcount]= -idx-1; }); // 2nd half is [-1:-count]
    // stable-sort preserves order for strings that match
    thrust::stable_sort_by_key( execpol->on(0), d_w, d_w + akcount, d_x, [] __device__ (custring_view*& lhs, custring_view*& rhs) { return ((lhs && rhs) ? (lhs->compare(*rhs)<0) : (rhs!=0)); });
    rmm::device_vector<int> y(akcount,0); // holds matches resulting from
    int* d_y = y.data().get();            // sort are marked with '1'
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(0), (akcount-1),
        [d_y, d_w] __device__ (int idx) {
            custring_view* lhs = d_w[idx];
            custring_view* rhs = d_w[idx+1];
            if( lhs && rhs )
                d_y[idx] = (int)(lhs->compare(*rhs)==0);
            else
                d_y[idx] = (int)(lhs==rhs);
        });
    //
    int matched = thrust::reduce( execpol->on(0), d_y, d_y + akcount ); // how many keys matched
    rmm::device_vector<int> nidxs(akcount); // needed for gather methods
    int* d_nidxs = nidxs.data().get(); // indexes of keys from key1 not in key2
    int cpcount = akcount; // how many keys copied
    {
        thrust::counting_iterator<int> citr(0);
        int* nend = thrust::copy_if( execpol->on(0), citr, citr + (akcount), d_nidxs,
            [d_x, d_y] __device__ (const int& idx) { return (d_x[idx]<0) || d_y[idx]; });
        cpcount = nend - d_nidxs;
    }
    if( cpcount < akcount )
    {   // if keys are removed, we need to make a copy;
        // the gather()s here will select the remaining keys
        rmm::device_vector<custring_view*> wstrs2(cpcount);
        rmm::device_vector<int> x2(cpcount);
        thrust::gather( execpol->on(0), d_nidxs, d_nidxs + cpcount, wstrs.begin(), wstrs2.begin() );
        thrust::gather( execpol->on(0), d_nidxs, d_nidxs + cpcount, x.begin(), x2.begin() );
        wstrs.swap(wstrs2);
        d_w = wstrs.data().get();
        x.swap(x2);
        d_x = x.data().get();
        akcount = cpcount;
    }
    thrust::unique_by_key( execpol->on(0), d_w, d_w + akcount, d_x, [] __device__ (custring_view* lhs, custring_view* rhs) { return ((lhs && rhs) ? (lhs->compare(*rhs)==0) : (lhs==rhs)); });
    int ucount = akcount - matched;
    // d_w,ucount are now the keys
    NVCategoryImpl_keys_from_custringarray(rtn->pImpl,d_w,ucount);
    // now remap the values: positive values in d_x are [0:ucount)
    thrust::fill( execpol->on(0), d_y, d_y + kcount, -1);
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(0), ucount,
        [d_y, d_x] __device__ (int idx) {
            int u = d_x[idx];
            if( u >= 0 )
                d_y[u] = idx;
        });
    // allocate new map
    int* d_map = pImpl->getMapPtr();
    if( mcount && d_map )
    {
        rmm::device_vector<int>* pNewMap = new rmm::device_vector<int>(mcount,0);
        int* d_newmap = pNewMap->data().get(); // new map goes in here
        thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int>(0), mcount,
            [d_map, d_y, d_newmap] __device__ (int idx) {
                int v = d_map[idx];
                d_newmap[idx] = (v < 0 ? v : d_y[v]);
            });
        //
        rtn->pImpl->pMap = pNewMap;
    }
    return rtn;
}
