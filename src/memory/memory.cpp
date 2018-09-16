/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

/**
 * @brief Device Memory Manager implementation. 
 *
 * Efficient allocation, deallocation and tracking of GPU memory.
 *
 */

#include "rmm.h"
#include "memory_manager.h"
#include <fstream>
#include <sstream>

/** ---------------------------------------------------------------------------*
 * @brief Macro wrapper to check for error in RMM API calls.
 * ---------------------------------------------------------------------------**/
#define RMM_CHECK(call) do { \
    rmmError_t error = (call); \
    if( error != RMM_SUCCESS ) return error; \
} while(0)

/** ---------------------------------------------------------------------------*
 * @brief Macro wrapper for RMM API calls to return appropriate RMM errors.
 * ---------------------------------------------------------------------------**/
#define RMM_CHECK_CUDA(call) do { \
    cudaError_t cudaError = (call); \
    if( cudaError == cudaErrorMemoryAllocation ) \
        return RMM_ERROR_OUT_OF_MEMORY; \
    else if( cudaError != cudaSuccess ) \
        return RMM_ERROR_CUDA_ERROR; \
} while(0)

// Global instance of the log
rmm::Logger theLog;
rmm::Manager *theManager;

// RAII logger class
class LogIt
{
public:
    LogIt(rmm::Logger::MemEvent_t event, size_t size, cudaStream_t stream) 
    : event(event), device(0), ptr(0), size(size), stream(stream)
    {
        cudaGetDevice(&device);
        start = std::chrono::system_clock::now();
    }

    LogIt(rmm::Logger::MemEvent_t event, void* ptr, size_t size, cudaStream_t stream) 
    : event(event), device(0), ptr(ptr), size(size), stream(stream)
    {
        cudaGetDevice(&device);
        start = std::chrono::system_clock::now();
    }

    /// Sometimes you need to start logging before the pointer address is known
    void setPointer(void* p) { ptr = p; }

    ~LogIt() 
    {
        rmm::Logger::TimePt end = std::chrono::system_clock::now();
        size_t freeMem, totalMem;
        rmmGetInfo(&freeMem, &totalMem, stream);
        theLog.record(event, device, ptr, start, end, freeMem, totalMem, size, stream); 
    }

private:
    rmm::Logger::MemEvent_t event;
    int device;
    void* ptr;
    size_t size;
    cudaStream_t stream;
    rmm::Logger::TimePt start;
};

#ifndef GETNAME
#define GETNAME(x) case x: return #x;
#endif

// Stringify RMM error code.
const char * rmmGetErrorString(rmmError_t errcode) {
  switch (errcode) {
    // There must be one entry per enum values in gdf_error.
    GETNAME(RMM_SUCCESS)
    GETNAME(RMM_ERROR_CUDA_ERROR)
    GETNAME(RMM_ERROR_INVALID_ARGUMENT)
    GETNAME(RMM_ERROR_NOT_INITIALIZED)
    GETNAME(RMM_ERROR_OUT_OF_MEMORY)
    GETNAME(RMM_ERROR_UNKNOWN)
    GETNAME(RMM_ERROR_IO)
    default:
        // This means we are missing an entry above for a rmmError_t value.
        return "Internal error. Unknown error code.";
  }
}

// Initialize memory manager state and storage.
rmmError_t rmmInitialize()
{
    theManager = new rmm::Manager;
#ifndef RMM_USE_CUDAMALLOC
    cnmemDevice_t dev;
    RMM_CHECK_CUDA( cudaGetDevice(&(dev.device)) );
    dev.size = 16800000000;
    dev.numStreams = 1;
    cudaStream_t streams[1]; streams[0] = 0;
    dev.streams = streams;
    dev.streamSizes = 0;
    RMM_CHECK_CNMEM( cnmemInit(1, &dev, 0) );    
#endif
    return RMM_SUCCESS;
}

// Shutdown memory manager.
rmmError_t rmmFinalize()
{
#ifndef RMM_USE_CUDAMALLOC
    RMM_CHECK_CNMEM( cnmemFinalize() );
#endif
    delete theManager;
    return RMM_SUCCESS;
}
 
// Allocate memory and return a pointer to device memory. 
rmmError_t rmmAlloc(void **ptr, size_t size, cudaStream_t stream)
{
    LogIt log(rmm::Logger::Alloc, size, stream);

    if (!ptr && !size) {
        return RMM_SUCCESS;
    }
    
    if (!ptr) 
        return RMM_ERROR_INVALID_ARGUMENT;

#ifdef RMM_USE_CUDAMALLOC
    RMM_CHECK_CUDA(cudaMalloc(ptr, size));
#else
    RMM_CHECK( theManager->registerStream(stream) );
    RMM_CHECK_CNMEM( cnmemMalloc(ptr, size, stream) );
#endif
    log.setPointer(*ptr);
    return RMM_SUCCESS;
}

// Reallocate device memory block to new size and recycle any remaining memory.
rmmError_t rmmRealloc(void **ptr, size_t new_size, cudaStream_t stream)
{
    LogIt log(rmm::Logger::Realloc, new_size, stream);

	if (!ptr && !new_size) {
        return RMM_SUCCESS;
    }

    if (!ptr) 
    	return RMM_ERROR_INVALID_ARGUMENT;

#ifdef RMM_USE_CUDAMALLOC
    RMM_CHECK_CUDA(cudaFree(*ptr));
	RMM_CHECK_CUDA(cudaMalloc(ptr, new_size));
#else
    RMM_CHECK( theManager->registerStream(stream) );
    RMM_CHECK_CNMEM( cnmemFree(*ptr, stream) );
    RMM_CHECK_CNMEM( cnmemMalloc(ptr, new_size, stream) );
#endif
    return RMM_SUCCESS;
}

// Release device memory and recycle the associated memory.
rmmError_t rmmFree(void *ptr, cudaStream_t stream)
{
    LogIt log(rmm::Logger::Free, ptr, 0, stream);
#ifdef RMM_USE_CUDAMALLOC
	RMM_CHECK_CUDA(cudaFree(ptr));
#else
    RMM_CHECK_CNMEM( cnmemFree(ptr, stream) );
#endif
	return RMM_SUCCESS;
}

// Get amounts of free and total memory managed by a manager associated with the stream.
rmmError_t rmmGetInfo(size_t *freeSize, size_t *totalSize, cudaStream_t stream)
{
#ifdef RMM_USE_CUDAMALLOC
    RMM_CHECK_CUDA(cudaMemGetInfo(freeSize, totalSize));
#else
    RMM_CHECK( theManager->registerStream(stream) );
    RMM_CHECK_CNMEM( cnmemMemGetInfo(freeSize, totalSize, stream) );
#endif
	return RMM_SUCCESS;
}

// Write the memory event stats log to specified path/filename
rmmError_t rmmWriteLog(const char* filename)
{
    try 
    {
        std::ofstream csv;
        csv.open(filename);
        theLog.to_csv(csv);
    }
    catch (const std::ofstream::failure& e) {
        return RMM_ERROR_IO;
    }
    return RMM_SUCCESS;
}

size_t rmmLogSize()
{
    std::ostringstream csv; 
    theLog.to_csv(csv);
    return csv.str().size();
}

rmmError_t rmmGetLog(char *buffer, size_t buffer_size)
{
    try 
    {
        std::ostringstream csv; 
        theLog.to_csv(csv);
        csv.str().copy(buffer, std::min(buffer_size, csv.str().size()));
    }
    catch (const std::ofstream::failure& e) {
        return RMM_ERROR_IO;
    }
    return RMM_SUCCESS;
}

