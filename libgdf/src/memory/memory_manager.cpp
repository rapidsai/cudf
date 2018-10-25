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

#include "memory_manager.h"

namespace rmm
{
    /** ---------------------------------------------------------------------------*
     * Record a memory manager event in the log.
     * 
     * @param[in] event The type of event (Alloc, Realloc, or Free)
     * @param[in] DeviceId The device to which this event applies.
     * @param[in] ptr The device pointer being allocated or freed.
     * @param[in] t The timestamp to record.
     * @param[in] size The size of allocation (only needed for Alloc/Realloc).
     * @param[in] stream The stream on which the allocation is happening 
     *                   (only needed for Alloc/Realloc).
     * ---------------------------------------------------------------------------**/
    void Logger::record(MemEvent_t event, int deviceId, void* ptr, 
                        TimePt start, TimePt end,
                        size_t freeMem, size_t totalMem,
                        size_t size, cudaStream_t stream)
                        
    {
        std::lock_guard<std::mutex> guard(log_mutex);
        if (Alloc == event)
            current_allocations.insert(ptr);
        else if (Free == event)
            current_allocations.erase(ptr);
        events.push_back({event, deviceId, ptr, size, stream, freeMem, totalMem, current_allocations.size(), start, end});
    }

    void Logger::to_csv(std::ostream &csv)
    {
        csv << "Event Type,Device ID,Address,Stream,Size (bytes),Free Memory,Total Memory,Current Allocs,Start,End,Elapsed\n";

        for (auto& e : events)
        {
            auto event_str = "Alloc";
            if (e.event == Realloc) event_str = "Realloc";
            if (e.event == Free) event_str = "Free";

            std::chrono::duration<double> elapsed = e.end-e.start;
            
            csv << event_str << "," << e.deviceId << "," << e.ptr << ","  << e.stream << ","
                << e.size << "," << e.freeMem << "," << e.totalMem << "," << e.currentAllocations << ","
                << std::chrono::duration<double>(e.start-base_time).count() << ","
                << std::chrono::duration<double>(e.end-base_time).count() << ","
                << elapsed.count() << std::endl;
        }
    }

    void Logger::clear()
    {
        std::lock_guard<std::mutex> guard(log_mutex);
        events.clear();
    }
}
