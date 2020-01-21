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

#include <cstdio>
#include <cuda_runtime.h>

/**
 * @file ipc_transfer.h
 * @brief Definition for IPC transfer structures.
 */

/**
 * @brief This is used by the create_from_ipc and create_ipc_transfer methods.
 *
 * It is used to serialize and deserialize an NVStrings instance to/from another context.
 */
struct nvstrings_ipc_transfer
{
    char* base_address;
    cudaIpcMemHandle_t hstrs;
    unsigned int count;
    void* strs;

    cudaIpcMemHandle_t hmem;
    size_t size;
    void* mem;

    nvstrings_ipc_transfer()
    : base_address(0), count(0), strs(0), size(0), mem(0) {}

    ~nvstrings_ipc_transfer() {}

    /**
     * @brief Sets the strings pointers memory into this context.
     *
     * @param[in] in_ptr Memory pointer to strings array.
     * @param[in] base_address Address of original context.
     * @param[in] count Number of elements in the array.
     */
    void setStrsHandle(void* in_ptr, char* base_address, unsigned int count)
    {
        this->count = count;
        this->base_address = base_address;
        cudaIpcGetMemHandle(&hstrs,in_ptr);
    }

    /**
     * @brief Sets the strings objects memory into this context.
     *
     * @param[in] in_ptr Memory pointer to strings object array.
     * @param[in] size The size of the memory in bytes.
     */
    void setMemHandle(void* in_ptr, size_t size)
    {
        this->size = size;
        cudaIpcGetMemHandle(&hmem,in_ptr);
    }

    /**
     * @brief Creates array pointer that can be transferred.
     */
    void* getStringsPtr()
    {
        if( !strs && count )
        {
            cudaError_t err = cudaIpcOpenMemHandle((void**)&strs,hstrs,cudaIpcMemLazyEnablePeerAccess);
            if( err!=cudaSuccess )
                printf("%d nvs-getStringsPtr", err);
        }
        return strs;
    }

    /**
     * @brief Creates memory pointer that can be transferred.
     */
    void* getMemoryPtr()
    {
        if( !mem && size )
        {
            cudaError_t err = cudaIpcOpenMemHandle((void**)&mem,hmem,cudaIpcMemLazyEnablePeerAccess);
            if( err!=cudaSuccess )
                printf("%d nvs-getMemoryPtr", err);
        }
        return mem;
    }
};

/**
 * @brief This is used by the create_from_ipc and create_ipc_transfer methods.
 *
 * It is used to serialize and deserialize an NVStrings instance to/from another context.
 */
struct nvcategory_ipc_transfer
{
    char* base_address;
    cudaIpcMemHandle_t hstrs;
    unsigned int keys;
    void* strs;

    cudaIpcMemHandle_t hmem;
    size_t size;
    void* mem;

    cudaIpcMemHandle_t hmap;
    unsigned int count;
    void* vals;

    nvcategory_ipc_transfer()
    : base_address(0), keys(0), strs(0), size(0), mem(0), count(0), vals(0) {}

    ~nvcategory_ipc_transfer()
    {
        if( strs )
            cudaIpcCloseMemHandle(strs);
        if( mem )
            cudaIpcCloseMemHandle(mem);
        if( vals )
            cudaIpcCloseMemHandle(vals);
    }

    /**
     * @brief Sets the strings pointers memory into this context.
     *
     * @param[in] in_ptr Memory pointer to strings array.
     * @param[in] base_address Address of original context.
     * @param[in] count Number of elements in the array.
     */
    void setStrsHandle(void* in_ptr, char* base_address, unsigned int count)
    {
        keys = count;
        this->base_address = base_address;
        cudaIpcGetMemHandle(&hstrs,in_ptr);
    }

    /**
     * @brief Sets the strings objects memory into this context.
     *
     * @param[in] in_ptr Memory pointer to strings object array.
     * @param[in] size The size of the memory in bytes.
     */
    void setMemHandle(void* in_ptr, size_t size)
    {
        this->size = size;
        cudaIpcGetMemHandle(&hmem,in_ptr);
    }

    /**
     * @brief Sets the index values memory into this context.
     *
     * @param[in] in_ptr Memory pointer to the array.
     * @param[in] count The number of elements in the array.
     */
    void setMapHandle(void* in_ptr, unsigned int count)
    {
        this->count = count;
        cudaIpcGetMemHandle(&hmap,in_ptr);
    }

    /**
     * @brief Creates strings array pointer that can be transferred.
     */
    void* getStringsPtr()
    {
        if( !strs && keys )
            cudaIpcOpenMemHandle((void**)&strs,hstrs,cudaIpcMemLazyEnablePeerAccess);
        return strs;
    }

    /**
     * @brief Creates memory pointer that can be transferred.
     */
    void* getMemoryPtr()
    {
        if( !mem && size )
            cudaIpcOpenMemHandle((void**)&mem,hmem,cudaIpcMemLazyEnablePeerAccess);
        return mem;
    }

    /**
     * @brief Creates value arrays pointer that can be transferred.
     */
    void* getMapPtr()
    {
        if( !vals && count )
            cudaIpcOpenMemHandle((void**)&vals,hmap,cudaIpcMemLazyEnablePeerAccess);
        return vals;
    }
};
