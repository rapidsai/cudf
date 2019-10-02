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
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/error_utils.hpp>

#include "nvstrings/NVStrings.h"
#include "nvstrings/ipc_transfer.h"
#include "nvstrings/StringsStatistics.h"

#include "./NVStringsImpl.h"
#include "../custring_view.cuh"
#include "../unicode/is_flags.h"
#include "../util.h"

#ifdef __INTELLISENSE__
unsigned int atomicAdd(unsigned int* address, unsigned int val);
#endif


// ctor and dtor are private to control the memory allocation in a single shared-object module
NVStrings::NVStrings(unsigned int count)
{
    pImpl = new NVStringsImpl(count);
}

NVStrings::NVStrings()
{
    pImpl = new NVStringsImpl(0);
}

NVStrings::NVStrings(const NVStrings& strs)
{
    unsigned int count = strs.size();
    pImpl = new NVStringsImpl(count);
    if( count )
    {
        std::vector<NVStringsImpl*> strslist;
        strslist.push_back(strs.pImpl);
        NVStrings_copy_strings(pImpl,strslist);
    }
}

NVStrings::~NVStrings()
{
    delete pImpl;
}

NVStrings* NVStrings::create_from_array( const char** strs, unsigned int count)
{
    NVStrings* rtn = new NVStrings(count);
    if( count )
    {
        if( NVStrings_init_from_strings(rtn->pImpl,strs,count) )
        {
            delete rtn;
            throw std::runtime_error("create_from_array runtime_error");
        }
    }
    return rtn;
}

NVStrings* NVStrings::create_from_index(std::pair<const char*,size_t>* strs, unsigned int count, bool devmem, sorttype stype)
{
    NVStrings* rtn = new NVStrings(count);
    if( !count )
        return rtn;

    int rc = NVStrings_init_from_indexes(rtn->pImpl,strs,count,devmem,stype);
    if( rc )
    {
        // cannot make any other CUDA calls if IllegalAddress error occurs
        if( rc==(int)cudaErrorIllegalAddress )
            throw std::invalid_argument("nvstrings::create_from_index bad_device_ptr");
        else
        {
            delete rtn;
            throw std::runtime_error("nvstrings::create_from_index runtime_error");
        }
    }
    return rtn;
}

NVStrings* NVStrings::create_from_offsets(const char* strs, int count, const int* offsets, const unsigned char* nullbitmask, int nulls, bool bdevmem)
{
    NVStrings* rtn = new NVStrings(count);
    if( !count )
        return rtn;
    if( bdevmem )
        NVStrings_init_from_device_offsets(rtn->pImpl,strs,count,offsets,nullbitmask,nulls);
    else
        NVStrings_init_from_offsets(rtn->pImpl,strs,count,offsets,nullbitmask,nulls);
    return rtn;
}

NVStrings* NVStrings::create_from_strings( std::vector<NVStrings*> strs )
{
    unsigned int count = 0;
    for( auto itr=strs.begin(); itr!=strs.end(); itr++ )
        count += (*itr)->size();
    NVStrings* rtn = new NVStrings(count);
    if( count )
    {
        std::vector<NVStringsImpl*> impls;
        for( auto itr=strs.begin(); itr!=strs.end(); itr++ )
            impls.push_back( (*itr)->pImpl );
        NVStrings_copy_strings(rtn->pImpl,impls);
    }
    return rtn;
}

NVStrings* NVStrings::create_from_ipc( nvstrings_ipc_transfer& ipc )
{
    unsigned count = ipc.count;
    NVStrings* rtn = new NVStrings(count);
    if( count==0 )
        return rtn;
    rtn->pImpl->setMemoryHandle(ipc.getMemoryPtr(),ipc.size);
    custring_view_array strings = (custring_view_array)ipc.getStringsPtr();
    // copy the pointers so they can be fixed up
    cudaError_t err = cudaMemcpy(rtn->pImpl->getStringsPtr(),strings,count*sizeof(custring_view*),cudaMemcpyDeviceToDevice);
    cudaIpcCloseMemHandle((void *) strings);
    if( err!=cudaSuccess )
        printCudaError(err,"nvs-create-ipc");
    // fix up the pointers for this context
    NVStrings_fixup_pointers(rtn->pImpl,ipc.base_address);
    return rtn;
}

NVStrings* NVStrings::create_from_csv( const char* csvfile, unsigned int column, unsigned int lines, sorttype stype, bool nullIsEmpty)
{
    unsigned int flags = nullIsEmpty ? CSV_NULL_IS_EMPTY : 0;
    if( stype & NVStrings::length )
        flags |= CSV_SORT_LENGTH;
    if( stype & NVStrings::name )
        flags |= CSV_SORT_NAME;
    std::string fpath = csvfile;
    return createFromCSV(fpath,column,lines,flags);
}

void NVStrings::destroy(NVStrings* inst)
{
    delete inst;
}

size_t NVStrings::memsize() const
{
    return pImpl->getMemorySize() + pImpl->getPointerSize();
}

NVStrings* NVStrings::copy()
{
    unsigned int count = size();
    NVStrings* rtn = new NVStrings(count);
    if( count )
    {
        std::vector<NVStringsImpl*> strslist;
        strslist.push_back(pImpl);
        NVStrings_copy_strings(rtn->pImpl,strslist);
    }
    return rtn;
}

//
void NVStrings::print( int start, int end, int maxwidth, const char* delimiter )
{
    unsigned int count = size();
    if( end < 0 || end > (int)count )
        end = count;
    if( start < 0 )
        start = 0;
    if( start >= end )
        return;
    count = end - start;
    //
    auto execpol = rmm::exec_policy(0);
    custring_view** d_strings = pImpl->getStringsPtr();
    rmm::device_vector<size_t> lens(count,0);
    size_t* d_lens = lens.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(start), end,
        [d_strings, start, maxwidth, d_lens] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int len = dstr->size();
            if( maxwidth > 0 )
                len = dstr->byte_offset_for(maxwidth);
            d_lens[idx-start] = len +1; // include null-terminator;
        });

    // allocate large device buffer to hold all the strings
    size_t msize = thrust::reduce(execpol->on(0),lens.begin(),lens.end());
    if( msize==0 )
    {
        printf("all %d strings are null\n",count);
        return;
    }
    char* d_buffer = device_alloc<char>(msize,0);
    // convert lengths to offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lens.begin(),lens.end(),offsets.begin());
    size_t* d_offsets = offsets.data().get();
    // copy strings into single buffer
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(start), end,
        [d_strings, start, maxwidth, d_offsets, d_lens, d_buffer] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            size_t offset = d_offsets[idx-start];
            char* optr = d_buffer + offset;
            if( dstr )
            {
                dstr->copy(optr,maxwidth);
                size_t len = d_lens[idx-start];
                //memcpy(optr,dstr->data(),len-1);
                *(optr+len-1) = 0;
            }
        });
    //
    //cudaDeviceSynchronize();
    // copy strings to host
    char* h_buffer = new char[msize];
    CUDA_TRY( cudaMemcpyAsync(h_buffer, d_buffer, msize, cudaMemcpyDeviceToHost))
    RMM_FREE(d_buffer,0);
    // print strings to stdout
    thrust::host_vector<custring_view*> h_strings(*(pImpl->pList)); // just for checking nulls
    thrust::host_vector<size_t> h_lens(lens);
    char* hstr = h_buffer;
    for( unsigned int idx=0; idx < count; ++idx )
    {
        printf("%u:",idx);
        if( !h_strings[idx] )
            printf("<null>");
        else
            printf("[%s]",hstr);
        printf("%s",delimiter);
        hstr += h_lens[idx];
    }
    delete h_buffer;
}

//
int NVStrings::to_host(char** list, int start, int end)
{
    unsigned int count = size();
    if( end < 0 || end > (int)count )
        end = count;
    if( start >= end )
        return 0;
    count = end - start;

    // compute size of specified strings
    auto execpol = rmm::exec_policy(0);
    rmm::device_vector<size_t> lens(count,0);
    size_t* d_lens = lens.data().get();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(start), end,
        [d_strings, start, d_lens] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_lens[idx-start] = dstr->size()+1; // include space for null terminator
        });

    cudaError_t err = cudaSuccess;
    size_t msize = thrust::reduce(execpol->on(0),lens.begin(),lens.end());
    if( msize==0 )
        return 0; // every string is null so we are done

    // allocate device memory to copy strings temporarily
    char* d_buffer = nullptr;
    rmmError_t rerr = RMM_ALLOC(&d_buffer,msize,0);
    if( rerr != RMM_SUCCESS )
    {
        fprintf(stderr,"nvs-to_host: RM_ALLOC(%p,%lu)=%d\n", d_buffer,msize,(int)rerr);
        //printCudaError(err);
        return (int)err;
    }
    // convert lengths to offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan(execpol->on(0),lens.begin(),lens.end(),offsets.begin());
    size_t* d_offsets = offsets.data().get();
    // copy strings into temporary buffer
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(start), end,
        [d_strings, start, d_offsets, d_buffer] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            size_t offset = d_offsets[idx-start];
            char* optr = d_buffer + offset;
            if( dstr )
            {
                int len = dstr->size();
                memcpy(optr,dstr->data(),len);
                *(optr + len) = 0;
            }
        });

    // copy strings to host
    char* h_buffer = new char[msize];
    err = cudaMemcpyAsync(h_buffer, d_buffer, msize, cudaMemcpyDeviceToHost);
    RMM_FREE(d_buffer,0); // done with device buffer
    if( err != cudaSuccess )
    {
        printCudaError(err, "nvs-to_host: copying strings device to host");
        delete h_buffer;
        return (int)err;
    }

    // Deserialization host memory to memory provided by the caller
    thrust::host_vector<custring_view*> h_strings(*(pImpl->pList)); // just for checking nulls
    thrust::host_vector<size_t> h_offsets(offsets);
    h_offsets.push_back(msize); // include size as last offset
    for( unsigned int idx=0; idx < count; ++idx )
    {
        if( h_strings[idx]==0 )
            continue;
        size_t offset = h_offsets[idx];
        size_t length = h_offsets[idx+1] - offset;
        const char* p_data = h_buffer + offset;
        if( list[idx] )
            memcpy(list[idx], p_data, length-1);
    }
    delete h_buffer;
    return 0;
}

// build a string-index from this instances strings
int NVStrings::create_index(std::pair<const char*,size_t>* strs, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    custring_view_array d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_indexes] __device__(unsigned int idx){
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

    cudaError_t err = cudaSuccess; //cudaDeviceSynchronize();
    if( bdevmem )
        err = cudaMemcpyAsync( strs, indexes.data().get(), count * sizeof(std::pair<const char*,size_t>), cudaMemcpyDeviceToDevice );
    else
        err = cudaMemcpyAsync( strs, indexes.data().get(), count * sizeof(std::pair<const char*,size_t>), cudaMemcpyDeviceToHost );
    if( err != cudaSuccess )
    {
        printCudaError(err,"nvs-create_index");
        return (int)err;
    }
    return 0;
}

//
int NVStrings::create_custring_index( custring_view** strs, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 )
        return 0;
    custring_view_array d_strings = pImpl->getStringsPtr();
    if( bdevmem )
        CUDA_TRY( cudaMemcpyAsync( strs, d_strings, count * sizeof(custring_view*), cudaMemcpyDeviceToDevice ))
    else
        CUDA_TRY( cudaMemcpyAsync( strs, d_strings, count * sizeof(custring_view*), cudaMemcpyDeviceToHost ))
    return 0;
}

// copy strings into memory provided
int NVStrings::create_offsets( char* strs, int* offsets, unsigned char* nullbitmask, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 )
        return 0;
    if( strs==0 || offsets==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    // first compute offsets/nullbitmask
    int* d_offsets = offsets;
    unsigned char* d_nulls = nullbitmask;
    if( !bdevmem )
    {
        d_offsets = device_alloc<int>((count+1),0);
        if( nullbitmask )
        {
            d_nulls = device_alloc<unsigned char>(((count+7)/8),0);
            cudaMemset(d_nulls,0,((count+7)/8));
        }
    }
    //
    rmm::device_vector<int> sizes(count+1,0);
    int* d_sizes = sizes.data().get();
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_sizes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_sizes[idx] = (int)dstr->size();
        });
    // ^^^-- these two for-each-n's can likely be combined --vvv
    if( d_nulls )
    {
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), (count+7)/8,
            [d_strings, count, d_nulls] __device__(unsigned int idx){
                unsigned int ndx = idx * 8;
                unsigned char nb = 0;
                for( int i=0; i<8; ++i )
                {
                    nb = nb >> 1;
                    if( ndx+i < count )
                    {
                        custring_view* dstr = d_strings[ndx+i];
                        if( dstr )
                            nb |= 128;
                    }
                }
                d_nulls[idx] = nb;
            });
    }
    //
    thrust::exclusive_scan(execpol->on(0),d_sizes, d_sizes+(count+1), d_offsets);
    // compute/allocate of memory
    size_t totalbytes = thrust::reduce(execpol->on(0), d_sizes, d_sizes+count);
    char* d_strs = strs;
    if( !bdevmem )
        d_strs = device_alloc<char>(totalbytes,0);
    // shuffle strings into memory
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_strs, d_offsets] __device__(unsigned int idx){
            char* buffer = d_strs + d_offsets[idx];
            custring_view* dstr = d_strings[idx];
            if( dstr )
                memcpy(buffer,dstr->data(),dstr->size());
        });
    // copy memory to parameters (if necessary)
    if( !bdevmem )
    {
        cudaMemcpyAsync(offsets,d_offsets,(count+1)*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(strs,d_strs,totalbytes,cudaMemcpyDeviceToHost);
        if( nullbitmask )
        {
            cudaMemcpyAsync(nullbitmask,d_nulls,((count+7)/8)*sizeof(unsigned char),cudaMemcpyDeviceToHost);
            RMM_FREE(d_nulls,0);
        }
        RMM_FREE(d_offsets,0);
        RMM_FREE(d_strs,0);
    }
    return 0;
}

int NVStrings::create_ipc_transfer( nvstrings_ipc_transfer& ipc )
{
    ipc.setStrsHandle(pImpl->getStringsPtr(),pImpl->getMemoryPtr(),size());
    ipc.setMemHandle(pImpl->getMemoryPtr(),pImpl->getMemorySize());
    return 0;
}

// fills in a bitarray with 0 for null values and 1 for non-null values
// if emptyIsNull=true, empty strings will have bit values of 0 as well
unsigned int NVStrings::set_null_bitarray( unsigned char* bitarray, bool emptyIsNull, bool devmem )
{
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    unsigned int size = (count + 7)/8; // round up to byte align
    unsigned char* d_bitarray = bitarray;
    if( !devmem )
        d_bitarray = device_alloc<unsigned char>(size,0);

    // count nulls in range for return value
    custring_view** d_strings = pImpl->getStringsPtr();
    unsigned int ncount = thrust::count_if(execpol->on(0), d_strings, d_strings + count,
       [emptyIsNull] __device__ (custring_view*& dstr) { return (dstr==0) || (emptyIsNull && !dstr->size()); });

    // fill in the bitarray
    // the bitmask is in arrow format which means for each byte
    // the null indicator is in bit position right-to-left: 76543210
    // logic sets the high-bit and shifts to the right
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), size,
        [d_strings, count, emptyIsNull, d_bitarray] __device__(unsigned int byteIdx){
            unsigned char byte = 0; // set one byte per thread -- init to all nulls
            for( unsigned int i=0; i < 8; ++i )
            {
                unsigned int idx = i + (byteIdx*8);  // compute d_strings index
                byte = byte >> 1;                    // shift until we are done
                if( idx < count )                    // check boundary
                {
                    custring_view* dstr = d_strings[idx];
                    if( dstr && (!emptyIsNull || dstr->size()) )
                        byte |= 128;                 // string is not null, set high bit
                }
            }
            d_bitarray[byteIdx] = byte;
        });
    //
    //cudaError_t err = cudaDeviceSynchronize();
    //if( err != cudaSuccess )
    //{
    //    fprintf(stderr,"nvs-set_null_bitarray(%p,%d,%d) size=%u\n",bitarray,(int)emptyIsNull,(int)devmem,count);
    //    printCudaError(err);
    //}
    //
    if( !devmem )
    {
        CUDA_TRY( cudaMemcpyAsync(bitarray,d_bitarray,size,cudaMemcpyDeviceToHost))
        RMM_FREE(d_bitarray,0);
    }
    return ncount;
}

// set int array with position of null strings
unsigned int NVStrings::get_nulls( unsigned int* array, bool emptyIsNull, bool devmem )
{
    unsigned int count = size();
    if( count==0 )
        return 0;

    auto execpol = rmm::exec_policy(0);
    rmm::device_vector<int> darray(count,-1);
    int* d_array = darray.data().get();

    custring_view** d_strings = pImpl->getStringsPtr();
    //unsigned int ncount = thrust::count_if(execpol->on(0), d_strings, d_strings + count,
    //   [emptyIsNull] __device__ (custring_view*& dstr) { return (dstr==0) || (emptyIsNull && !dstr->size()); });

    // fill in the array
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, emptyIsNull, d_array] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr && (!emptyIsNull || dstr->size()) )
                d_array[idx] = -1; // not null
            else
                d_array[idx] = idx;  // null
        });
    //
    //cudaError_t err = cudaDeviceSynchronize();
    //if( err != cudaSuccess )
    //{
    //    fprintf(stderr,"nvs-get_nulls(%p,%d,%d) size=%u\n",array,(int)emptyIsNull,(int)devmem,count);
    //    printCudaError(err);
    //}
    // compact out the negative values
    int* newend = thrust::remove_if(execpol->on(0), d_array, d_array + count, [] __device__ (int val) {return val<0;});
    unsigned int ncount = (unsigned int)(newend - d_array);

    //
    cudaError_t err = cudaSuccess;
    if( array )
    {
        if( devmem )
            err = cudaMemcpyAsync(array,d_array,sizeof(int)*ncount,cudaMemcpyDeviceToDevice);
        else
            err = cudaMemcpyAsync(array,d_array,sizeof(int)*ncount,cudaMemcpyDeviceToHost);
    }
    if( err != cudaSuccess )
    {
        fprintf(stderr,"nvs-get_nulls(%p,%d,%d) size=%u\n",array,(int)emptyIsNull,(int)devmem,count);
        printCudaError(err);
    }
    return ncount;
}

// number of strings in this instance
unsigned int NVStrings::size() const
{
    return pImpl->getCount();
}

struct statistics_attrs
{
    custring_view_array d_strings;
    unsigned char* d_flags;
    size_t* d_values;
    unsigned int d_mask;

    statistics_attrs( custring_view_array strings, unsigned char* flags, size_t* values, unsigned int mask )
    : d_strings(strings), d_flags(flags), d_values(values), d_mask(mask) {}

    __device__ void operator()(unsigned int idx)
    {
            custring_view* dstr = d_strings[idx];
            size_t spaces = 0;
            if( dstr )
            {
                for( auto itr = dstr->begin(); itr != dstr->end(); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    unsigned int flg = uni <= 0x00FFFF ? d_flags[uni] : 0;
                    spaces += (size_t)((flg & d_mask)>0);
                }
            }
            d_values[idx] = spaces;
    }
};

void NVStrings::compute_statistics(StringsStatistics& stats)
{
    unsigned int count = size();
    memset((void*)&stats,0,sizeof(stats));
    if( count==0 )
        return;

    stats.total_strings = count;
    auto execpol = rmm::exec_policy(0);
    size_t stringsmem = pImpl->getMemorySize();
    size_t ptrsmem = pImpl->pList->size() * sizeof(custring_view*);
    stats.total_memory = stringsmem + ptrsmem;

    custring_view_array d_strings = pImpl->getStringsPtr();
    rmm::device_vector<size_t> values(count,0);
    size_t* d_values = values.data().get();

    // count strings
    stats.total_nulls = thrust::count_if(execpol->on(0), d_strings, d_strings + count,
        [] __device__ (custring_view* dstr) { return dstr==nullptr; });
    stats.total_empty = thrust::count_if(execpol->on(0), d_strings, d_strings + count,
        [] __device__ (custring_view* dstr) { return dstr && dstr->empty(); });

    // bytes
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_values] __device__ (unsigned int idx) {
            custring_view* dstr = d_strings[idx];
            d_values[idx] = dstr ? dstr->size() : 0;
        });
    {
        thrust::remove(execpol->on(0), values.begin(), values.end(), 0L );
        stats.bytes_max = *thrust::max_element(execpol->on(0), values.begin(), values.end());
        stats.bytes_min = *thrust::min_element(execpol->on(0), values.begin(), values.end());
        stats.total_bytes = thrust::reduce(execpol->on(0), values.begin(), values.end());
        stats.bytes_avg = stats.total_bytes / count;
    }

    // chars
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_values] __device__ (unsigned int idx) {
            custring_view* dstr = d_strings[idx];
            d_values[idx] = dstr ? dstr->chars_count() : 0;
        });
    {
        thrust::remove(execpol->on(0), values.begin(), values.end(), 0L );
        stats.chars_max = *thrust::max_element(execpol->on(0), values.begin(), values.end());
        stats.chars_min = *thrust::min_element(execpol->on(0), values.begin(), values.end());
        stats.total_chars = thrust::reduce(execpol->on(0), values.begin(), values.end());
        stats.chars_avg = stats.total_bytes / count;
    }

    // memory
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_values] __device__ (unsigned int idx) {
            custring_view* dstr = d_strings[idx];
            d_values[idx] = dstr ? dstr->alloc_size() : 0;
        });
    {
        thrust::remove(execpol->on(0), values.begin(), values.end(), 0L );
        stats.mem_max = *thrust::max_element(execpol->on(0), values.begin(), values.end());
        stats.mem_min = *thrust::min_element(execpol->on(0), values.begin(), values.end());
        size_t mem_total = thrust::reduce(execpol->on(0), values.begin(), values.end());
        stats.mem_avg = mem_total / count;
    }

    // attrs
    unsigned char* d_flags = get_unicode_flags();
    // spaces
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        statistics_attrs(d_strings, d_flags, d_values, 16));
    stats.whitespace_count = thrust::reduce(execpol->on(0), values.begin(), values.end());
    // digits
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        statistics_attrs(d_strings, d_flags, d_values, 4));
    stats.digits_count = thrust::reduce(execpol->on(0), values.begin(), values.end());
    // uppercase
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        statistics_attrs(d_strings, d_flags, d_values, 32));
    stats.uppercase_count = thrust::reduce(execpol->on(0), values.begin(), values.end());
    // lowercase
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        statistics_attrs(d_strings, d_flags, d_values, 64));
    stats.lowercase_count = thrust::reduce(execpol->on(0), values.begin(), values.end());

    // unique strings
    {
        // make a copy of the pointers so we can sort them
        rmm::device_vector<custring_view*> sortcopy(*(pImpl->pList));
        custring_view_array d_sortcopy = sortcopy.data().get();
        thrust::sort(execpol->on(0), d_sortcopy, d_sortcopy+count,
            [] __device__ (custring_view*& lhs, custring_view*& rhs) {
                return (lhs && rhs) ? (lhs->compare(*rhs) < 0): rhs!=0;
            });
        auto nend = thrust::unique(execpol->on(0), d_sortcopy, d_sortcopy+count,
            [] __device__ (custring_view* lhs, custring_view* rhs) {
                if( lhs==0 || rhs==0 )
                    return lhs==rhs;
                return lhs->compare(*rhs)==0;
            });
        stats.unique_strings = (size_t)(nend - d_sortcopy);
    }
    // histogram the characters
    {
        unsigned int uset_count = 0x010000;
        rmm::device_vector<unsigned int> charset(uset_count,0);
        unsigned int* d_charset = charset.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [d_strings, d_charset] __device__ (unsigned int idx) {
                custring_view* dstr = d_strings[idx];
                if( !dstr )
                    return;
                for( auto itr = dstr->begin(); itr != dstr->end(); itr++ )
                {
                    unsigned int uni = u82u(*itr);
                    if( uni <= 0x00FFFF )
                        atomicAdd(&(d_charset[uni]),1);
                }
            });
        rmm::device_vector<thrust::pair<unsigned int, unsigned int> > charcounts(uset_count);
        thrust::pair<unsigned int,unsigned int>* d_charcounts = charcounts.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), uset_count,
            [d_charset, d_charcounts] __device__ (unsigned int idx) {
                unsigned int val = d_charset[idx];
                if( val )
                {
                    d_charcounts[idx].first = u2u8(idx);
                    d_charcounts[idx].second = val;
                }
                else
                {
                    d_charcounts[idx].first = 0;
                    d_charcounts[idx].second = 0;
                }
            });
        auto nend = thrust::remove_if(execpol->on(0), d_charcounts, d_charcounts + uset_count,
            [] __device__ (thrust::pair<unsigned int,unsigned int> cc) { return cc.first==0; });
        // allocate host memory
        size_t elems = (size_t)(nend - d_charcounts);
        std::vector<std::pair<unsigned int, unsigned int> > hcharcounts(elems);
        // copy d_charcounts to host memory
        CUDA_TRY( cudaMemcpyAsync(hcharcounts.data(),d_charcounts,elems*sizeof(std::pair<unsigned int,unsigned int>),cudaMemcpyDeviceToHost))
        // copy hcharcounts to stats.char_counts;
        stats.char_counts.reserve(uset_count);
        stats.char_counts.swap(hcharcounts);
    }
}