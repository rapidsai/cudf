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

#include "orc_util.hpp"
#include <stdio.h>
#include <stdlib.h>
#include "orc_debug.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel_orc.cuh"

#include <fstream>
#include <boost/date_time/local_time/local_time.hpp>

namespace cudf {
namespace orc {

CudaOrcError_t AllocateAndCopyToDevice(void** dest, const void* src, size_t size)
{
    // the cuda memory size must be multipled by 4 for orc reader for cuda
    size_t local_size = size;
    CudaFuncCall(cudaMalloc(dest, local_size)); // cudaMalloc is aligned by 256B
    CudaFuncCall(cudaMemcpy(*dest, src, size, cudaMemcpyHostToDevice));

    return GDF_ORC_SUCCESS;
}

CudaOrcError_t AllocateTemporaryBufffer(void** dest, size_t size)
{
    CudaFuncCall(cudaMalloc(dest, size));
    return GDF_ORC_SUCCESS;
}

CudaOrcError_t ReleaseTemporaryBufffer(void* dest, size_t size)
{
    CudaFuncCall(cudaFree(dest));
    return GDF_ORC_SUCCESS;
}

//* find "Posix time zone string(IEEE Std 1003.1)" from timezone region string
//* return empty string if the region is not found in the system
std::string findPosixTimezoneString(const char* region)
{
    std::string ret;
#ifdef __linux__
    std::string filename("/usr/share/zoneinfo/");
#else
    std::string filename("examples/zoneinfo/");
#endif
    filename += region;

    using std::ios_base;

    // get the tail line of the file
    std::ifstream fin;
    fin.open(filename, ios_base::in | ios_base::binary);
    if (fin) {
        fin.seekg(-2, ios_base::end);
        char ch;
        while ('\n' != (ch = fin.get())) {
            fin.seekg(-2, ios_base::cur);
        }
        std::getline(fin, ret);
    }
    else {
        PRINTF("file open failure: %d\n", filename.c_str());
    }

    return ret;
}


//* find the standard GMT offset for the given region
//* return false if the region is not found in the system
bool findGMToffsetFromRegion(int& gmtoffset, const char* region)
{
    // Use zoninfo file and boost::local_time::posix_time_zone
    // to find the default GMT offset for the given region
    // If the c++ compiler supports C++20, we may use std::chrono::tzdb instead.
    // The supported compiler feature is upto C++14 at this point.
    gmtoffset = 0;

    std::string posixTzString = findPosixTimezoneString(region);
    if (posixTzString.empty())return false;

    boost::local_time::posix_time_zone tz(posixTzString);
    gmtoffset = tz.base_utc_offset().total_seconds();

    return true;
}

}   // namespace orc
}   // namespace cudf
