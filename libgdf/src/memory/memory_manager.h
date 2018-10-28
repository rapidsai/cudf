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

/** ---------------------------------------------------------------------------*
 * @brief Memory Manager class
 * 
 * Note: assumes at least C++11
 * ---------------------------------------------------------------------------**/

#pragma once

#include <vector>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <set>
#include <mutex>

#include "memory.h"
#include "cnmem.h"

/** ---------------------------------------------------------------------------*
 * @brief Macro wrapper for CNMEM API calls to return appropriate RMM errors.
 * ---------------------------------------------------------------------------**/
#define RMM_CHECK_CNMEM(call) do {            \
    cnmemStatus_t error = (call);             \
    switch (error) {                          \
    case CNMEM_STATUS_SUCCESS:                \
        break; /* don't return on success! */ \
    case CNMEM_STATUS_CUDA_ERROR:             \
        return RMM_ERROR_CUDA_ERROR;          \
    case CNMEM_STATUS_INVALID_ARGUMENT:       \
        return RMM_ERROR_INVALID_ARGUMENT;    \
    case CNMEM_STATUS_NOT_INITIALIZED:        \
        return RMM_ERROR_NOT_INITIALIZED;     \
    case CNMEM_STATUS_OUT_OF_MEMORY:          \
        return RMM_ERROR_OUT_OF_MEMORY;       \
    case CNMEM_STATUS_UNKNOWN_ERROR:          \
    default:                                  \
        return RMM_ERROR_UNKNOWN;             \
    }                                         \
} while(0)

typedef struct CUstream_st *cudaStream_t;

namespace rmm 
{
    class Logger
    {
    public:        
        Logger() { base_time = std::chrono::system_clock::now(); }

        typedef enum {
            Alloc = 0,
            Realloc,
            Free
        } MemEvent_t;

        using TimePt = std::chrono::system_clock::time_point;

        /// Record a memory manager event in the log.
        void record(MemEvent_t event, int deviceId, void* ptr,
                    TimePt start, TimePt end, 
                    size_t freeMem, size_t totalMem,
                    size_t size=0, cudaStream_t stream=0);

        void clear();
        
        /// Write the log to comma-separated value file
        void to_csv(std::ostream &csv);
    private:
        std::set<void*> current_allocations;

        struct MemoryEvent {
            MemEvent_t event;
            int deviceId;
            void* ptr;
            size_t size;
            cudaStream_t stream;
            size_t freeMem;
            size_t totalMem;
            size_t currentAllocations;
            TimePt start;
            TimePt end;
        };
        
        TimePt base_time;
        std::vector<MemoryEvent> events;
        std::mutex log_mutex;
    };

    class Manager
    {
    public:
        static Manager& getInstance(){
            // Myers' singleton. Thread safe and unique in C++11 -- Note, C++11 required!
            static Manager instance;
            return instance;
        }

        static Logger& getLogger() { return getInstance().logger; }

        static void setOptions(const rmmOptions_t &options) { getInstance().options = options; }
        static rmmOptions_t getOptions() { return getInstance().options; }

        void finalize() {
            std::lock_guard<std::mutex> guard(streams_mutex);
            registered_streams.clear();
            logger.clear();
        }

        /** ---------------------------------------------------------------------------*
         * @brief Register a new stream into the device memory manager.
         * 
         * Also returns success if the stream is already registered.
         * 
         * @param stream The stream to register
         * @return rmmError_t RMM_SUCCESS if all goes well, RMM_ERROR_INVALID_ARGUMENT
         *                    if the stream is invalid.
         * ---------------------------------------------------------------------------**/
        rmmError_t registerStream(cudaStream_t stream) { 
            std::lock_guard<std::mutex> guard(streams_mutex);
            if (registered_streams.empty() || 0 == registered_streams.count(stream)) {
                registered_streams.insert(stream);
                if (stream && PoolAllocation == options.allocation_mode) // don't register the null stream with CNMem
                    RMM_CHECK_CNMEM( cnmemRegisterStream(stream) );
            }
            return RMM_SUCCESS;
        }

    private:
        Manager() : options({ CudaDefaultAllocation, false, 0 }) {}
        ~Manager() = default;
        Manager(const Manager&) = delete;
        Manager& operator=(const Manager&) = delete;
  
        std::mutex streams_mutex;
        std::set<cudaStream_t> registered_streams;
        Logger logger;

        rmmOptions_t options;
    };    
}
