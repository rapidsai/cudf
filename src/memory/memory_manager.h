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
 * ---------------------------------------------------------------------------**/

#pragma once

#include <vector>
#include <chrono>
#include <string>
#include "memory.h"

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
                    TimePt start, TimePt finish, 
                    size_t size=0, cudaStream_t stream=0);
        
        /// Write the log to comma-separated value file
        void to_csv(const std::string &filename);
    private:
        struct MemoryEvent {
            MemEvent_t event;
            int deviceId;
            void* ptr;
            size_t size;
            cudaStream_t stream;
            TimePt start;
            TimePt end;
        };
        
        TimePt base_time;
        std::vector<MemoryEvent> events;
    };
}
