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
#include "gtest/gtest.h"
#include "tests/utilities/cudf_test_fixtures.h"
#include <rmm/rmm.h>

// Helper macros to simplify testing for success or failure
#define ASSERT_SUCCESS(res) ASSERT_EQ(RMM_SUCCESS, (res));
#define ASSERT_FAILURE(res) ASSERT_NE(RMM_SUCCESS, (res));

cudaStream_t stream;

struct MemoryManagerTest : 
    public ::testing::TestWithParam<rmmAllocationMode_t> 
{
    void SetUp() {
        ASSERT_EQ( cudaSuccess, cudaStreamCreate(&stream) );
        rmmOptions_t options {GetParam(), 0, false};
        ASSERT_SUCCESS( rmmInitialize(&options) );
    }

    void TearDown() {
        ASSERT_SUCCESS( rmmFinalize() );
        ASSERT_EQ( cudaSuccess, cudaStreamDestroy(stream) );
    }

    // some useful allocation sizes
    const size_t size_word = 4;
    const size_t size_kb = size_t{1}<<10;
    const size_t size_mb = size_t{1}<<20;
    const size_t size_gb = size_t{1}<<30;
    const size_t size_tb = size_t{1}<<40;
    const size_t size_pb = size_t{1}<<50;
};

// Init / Finalize tests

TEST_P(MemoryManagerTest, Initialize) {
    // Empty because handled in Fixture class.
}

TEST_P(MemoryManagerTest, Finalize) {
    // Empty because handled in Fixture class.
}

// zero size tests

TEST_P(MemoryManagerTest, AllocateZeroBytes) {
    char *a = 0;
    ASSERT_SUCCESS(RMM_ALLOC((void**)&a, 0, stream));
}

TEST_P(MemoryManagerTest, NullPtrAllocateZeroBytes) {
    ASSERT_SUCCESS(RMM_ALLOC(0, 0, stream));
}

// Bad argument tests

TEST_P(MemoryManagerTest, NullPtrInvalidArgument) {
    rmmError_t res = RMM_ALLOC(0, 4, stream);
    ASSERT_FAILURE(res);
    ASSERT_EQ(RMM_ERROR_INVALID_ARGUMENT, res);
}

// Simple allocation / free tests

TEST_P(MemoryManagerTest, AllocateWord) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC((void**)&a, size_word, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TEST_P(MemoryManagerTest, AllocateKB) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC((void**)&a, size_kb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TEST_P(MemoryManagerTest, AllocateMB) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC((void**)&a, size_mb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TEST_P(MemoryManagerTest, AllocateGB) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC((void**)&a, size_gb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TEST_P(MemoryManagerTest, AllocateTB) {
    char *a = 0;
    size_t freeBefore = 0, totalBefore = 0;
    ASSERT_SUCCESS( rmmGetInfo(&freeBefore, &totalBefore, stream) );
    
    if (size_tb > freeBefore) {
        ASSERT_FAILURE( RMM_ALLOC((void**)&a, size_tb, stream) );
    }
    else {
        ASSERT_SUCCESS( RMM_ALLOC((void**)&a, size_tb, stream) );
    }
    
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}


TEST_P(MemoryManagerTest, AllocateTooMuch) {
    char *a = 0;
    ASSERT_FAILURE( RMM_ALLOC((void**)&a, size_pb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TEST_P(MemoryManagerTest, FreeZero) {
    ASSERT_SUCCESS( RMM_FREE(0, stream) );
}

// Reallocation tests

TEST_P(MemoryManagerTest, ReallocateSmaller) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC((void**)&a, size_mb, stream) );
    ASSERT_SUCCESS( RMM_REALLOC((void**)&a, size_mb / 2, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TEST_P(MemoryManagerTest, ReallocateMuchSmaller) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC((void**)&a, size_gb, stream) );
    ASSERT_SUCCESS( RMM_REALLOC((void**)&a, size_kb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TEST_P(MemoryManagerTest, ReallocateLarger) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC((void**)&a, size_mb, stream) );
    ASSERT_SUCCESS( RMM_REALLOC((void**)&a, size_mb * 2, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TEST_P(MemoryManagerTest, ReallocateMuchLarger) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC((void**)&a, size_kb, stream) );
    ASSERT_SUCCESS( RMM_REALLOC((void**)&a, size_gb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TEST_P(MemoryManagerTest, GetInfo) {
    size_t freeBefore = 0, totalBefore = 0;
    ASSERT_SUCCESS( rmmGetInfo(&freeBefore, &totalBefore, stream) );
    ASSERT_GE(freeBefore, 0);
    ASSERT_GE(totalBefore, 0);

    char *a = 0;
    size_t sz = size_gb / 2;
    ASSERT_SUCCESS( RMM_ALLOC((void**)&a, sz, stream) );

    // make sure the available free memory goes down after an allocation
    size_t freeAfter = 0, totalAfter = 0;
    ASSERT_SUCCESS( rmmGetInfo(&freeAfter, &totalAfter, stream) );
    ASSERT_GE(totalAfter, totalBefore);
    ASSERT_LE(freeAfter, freeBefore);

    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TEST_P(MemoryManagerTest, AllocationOffset) {
    char *a = nullptr, *b = nullptr;
    ptrdiff_t offset = -1;
    ASSERT_SUCCESS( RMM_ALLOC((void**)&a, size_kb, stream) );
    ASSERT_SUCCESS( RMM_ALLOC((void**)&b, size_kb, stream) );

    ASSERT_SUCCESS( rmmGetAllocationOffset(&offset, a, stream) );
    ASSERT_GE(offset, 0);

    ASSERT_SUCCESS( rmmGetAllocationOffset(&offset, b, stream) );
    ASSERT_GE(offset, 0);

    ASSERT_SUCCESS( RMM_FREE(a, stream) );
    ASSERT_SUCCESS( RMM_FREE(b, stream) );
}

rmmAllocationMode_t modes[]= {CudaDefaultAllocation, 
                              PoolAllocation, 
                              static_cast<rmmAllocationMode_t>(PoolAllocation | CudaManagedMemory)};
INSTANTIATE_TEST_CASE_P(TestAllocationModes, MemoryManagerTest, ::testing::ValuesIn(modes));