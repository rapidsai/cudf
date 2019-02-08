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

#include "tests_common.h"


#ifdef WIN32
#ifdef _DEBUG
#error google tests is not designed for __Debug__ configulation
#error it will failed at run time
#pragma comment(lib, "libprotobufd.lib")    // required for bug: https://github.com/google/protobuf/issues/816
#pragma comment(lib, "orc_read_cudad.lib")
#else
#if 1   // 1: link protobuf full, 0: link protobuf-lite
#pragma comment(lib, "libprotobuf.lib")
#else
#pragma comment(lib, "libprotobuf-lite.lib")
#endif
#pragma comment(lib, "orc_read_cuda.lib")
#endif

#pragma comment(lib, "zlib.lib")

#pragma comment(lib, "gtest.lib")
#pragma comment(lib, "cuda.lib")
#endif


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}


