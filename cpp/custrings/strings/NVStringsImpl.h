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
#pragma once

#include <cuda_runtime.h>
#include <rmm/thrust_rmm_allocator.h>

class custring_view;
typedef custring_view** custring_view_array;

//
class NVStringsImpl
{
public:
    // this holds the strings in device memory
    // so operations can be performed on them through python calls
    rmm::device_vector<custring_view*>* pList;
    char* memoryBuffer;
    size_t bufferSize; // size of memoryBuffer only
    cudaStream_t stream_id;
    bool bIpcHandle; // whether memoryBuffer is ipc-handle or not

    //
    NVStringsImpl(unsigned int count);
    ~NVStringsImpl();
    char* createMemoryFor( size_t* d_lengths );

    inline custring_view_array getStringsPtr()      { return pList->data().get(); }
    inline unsigned int getCount() { return (unsigned int)pList->size(); }
    inline char* getMemoryPtr()    { return memoryBuffer; }
    inline size_t getMemorySize()  { return bufferSize;  }
    inline size_t getPointerSize() { return pList->size()*sizeof(custring_view*); }
    inline cudaStream_t getStream()    { return stream_id;  }
    inline void setMemoryBuffer( void* ptr, size_t memSize )
    {
        memoryBuffer = (char*)ptr;
        bufferSize = memSize;
    }
    void setMemoryHandle(void* ptr, size_t memSize)
    {
        setMemoryBuffer(ptr,memSize);
        bIpcHandle = true;
    }
};

//
void printCudaError( cudaError_t err, const char* prefix="\t" );
char32_t* to_char32( const char* ca );
unsigned char* get_unicode_flags();
unsigned short* get_charcases();

//
int NVStrings_init_from_strings(NVStringsImpl* pImpl, const char** strs, unsigned int count );
int NVStrings_init_from_indexes( NVStringsImpl* pImpl, std::pair<const char*,size_t>* indexes, unsigned int count, bool bdevmem, NVStrings::sorttype stype );
int NVStrings_init_from_offsets( NVStringsImpl* pImpl, const char* strs, int count, const int* offsets, const unsigned char* bitmask, int nulls );
int NVStrings_init_from_device_offsets( NVStringsImpl* pImpl, const char* strs, int count, const int* offsets, const unsigned char* bitmask, int nulls );
int NVStrings_copy_strings( NVStringsImpl* pImpl, std::vector<NVStringsImpl*>& strslist );
int NVStrings_fixup_pointers( NVStringsImpl* pImpl, char* baseaddr );