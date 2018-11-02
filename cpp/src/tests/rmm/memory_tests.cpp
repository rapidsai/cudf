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
#include "gdf_test_fixtures.h"
#include "rmm.h"

// Helper macros to simplify testing for success or failure
#define ASSERT_SUCCESS(res) ASSERT_EQ(RMM_SUCCESS, (res));
#define ASSERT_FAILURE(res) ASSERT_NE(RMM_SUCCESS, (res));

cudaStream_t stream;

/// Helper class for similar tests
struct MemoryManagerTest : public GdfTest {

    static void SetUpTestCase() {
        ASSERT_EQ( cudaSuccess, cudaStreamCreate(&stream) );
        GdfTest::SetUpTestCase();
    }

    static void TearDownTestCase() {
        GdfTest::TearDownTestCase();
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

TEST_F(MemoryManagerTest, Initialize) {
    // Empty because handled in Fixture class.
}

TEST_F(MemoryManagerTest, Finalize) {
    // Empty because handled in Fixture class.
}

// zero size tests

TEST_F(MemoryManagerTest, AllocateZeroBytes) {
    char *a = 0;
    ASSERT_SUCCESS(rmmAlloc((void**)&a, 0, stream));
}

TEST_F(MemoryManagerTest, NullPtrAllocateZeroBytes) {
    ASSERT_SUCCESS(rmmAlloc(0, 0, stream));
}

// Bad argument tests

TEST_F(MemoryManagerTest, NullPtrInvalidArgument) {
    rmmError_t res = rmmAlloc(0, 4, stream);
    ASSERT_FAILURE(res);
    ASSERT_EQ(RMM_ERROR_INVALID_ARGUMENT, res);
}

// Simple allocation / free tests

TEST_F(MemoryManagerTest, AllocateWord) {
    char *a = 0;
    ASSERT_SUCCESS( rmmAlloc((void**)&a, size_word, stream) );
    ASSERT_SUCCESS( rmmFree(a, stream) );
}

TEST_F(MemoryManagerTest, AllocateKB) {
    char *a = 0;
    ASSERT_SUCCESS( rmmAlloc((void**)&a, size_kb, stream) );
    ASSERT_SUCCESS( rmmFree(a, stream) );
}

TEST_F(MemoryManagerTest, AllocateMB) {
    char *a = 0;
    ASSERT_SUCCESS( rmmAlloc((void**)&a, size_mb, stream) );
    ASSERT_SUCCESS( rmmFree(a, stream) );
}

TEST_F(MemoryManagerTest, AllocateGB) {
    char *a = 0;
    ASSERT_SUCCESS( rmmAlloc((void**)&a, size_gb, stream) );
    ASSERT_SUCCESS( rmmFree(a, stream) );
}

TEST_F(MemoryManagerTest, AllocateTB) {
    char *a = 0;
    size_t freeBefore = 0, totalBefore = 0;
    ASSERT_SUCCESS( rmmGetInfo(&freeBefore, &totalBefore, stream) );
    
    if (size_tb > freeBefore) {
        ASSERT_FAILURE( rmmAlloc((void**)&a, size_tb, stream) );
    }
    else {
        ASSERT_SUCCESS( rmmAlloc((void**)&a, size_tb, stream) );
    }
    
    ASSERT_SUCCESS( rmmFree(a, stream) );
}


TEST_F(MemoryManagerTest, AllocateTooMuch) {
    char *a = 0;
    ASSERT_FAILURE( rmmAlloc((void**)&a, size_pb, stream) );
    ASSERT_SUCCESS( rmmFree(a, stream) );
}

TEST_F(MemoryManagerTest, FreeZero) {
    ASSERT_SUCCESS( rmmFree(0, stream) );
}

// Reallocation tests

TEST_F(MemoryManagerTest, ReallocateSmaller) {
    char *a = 0;
    ASSERT_SUCCESS( rmmAlloc((void**)&a, size_mb, stream) );
    ASSERT_SUCCESS( rmmRealloc((void**)&a, size_mb / 2, stream) );
    ASSERT_SUCCESS( rmmFree(a, stream) );
}

TEST_F(MemoryManagerTest, ReallocateMuchSmaller) {
    char *a = 0;
    ASSERT_SUCCESS( rmmAlloc((void**)&a, size_gb, stream) );
    ASSERT_SUCCESS( rmmRealloc((void**)&a, size_kb, stream) );
    ASSERT_SUCCESS( rmmFree(a, stream) );
}

TEST_F(MemoryManagerTest, ReallocateLarger) {
    char *a = 0;
    ASSERT_SUCCESS( rmmAlloc((void**)&a, size_mb, stream) );
    ASSERT_SUCCESS( rmmRealloc((void**)&a, size_mb * 2, stream) );
    ASSERT_SUCCESS( rmmFree(a, stream) );
}

TEST_F(MemoryManagerTest, ReallocateMuchLarger) {
    char *a = 0;
    ASSERT_SUCCESS( rmmAlloc((void**)&a, size_kb, stream) );
    ASSERT_SUCCESS( rmmRealloc((void**)&a, size_gb, stream) );
    ASSERT_SUCCESS( rmmFree(a, stream) );
}

TEST_F(MemoryManagerTest, GetInfo) {
    size_t freeBefore = 0, totalBefore = 0;
    ASSERT_SUCCESS( rmmGetInfo(&freeBefore, &totalBefore, stream) );
    ASSERT_GE(freeBefore, 0);
    ASSERT_GE(totalBefore, 0);

    char *a = 0;
    size_t sz = size_gb / 2;
    ASSERT_SUCCESS( rmmAlloc((void**)&a, sz, stream) );

    // make sure the available free memory goes down after an allocation
    size_t freeAfter = 0, totalAfter = 0;
    ASSERT_SUCCESS( rmmGetInfo(&freeAfter, &totalAfter, stream) );
    ASSERT_GE(totalAfter, totalBefore);
    ASSERT_LE(freeAfter, freeBefore);

    ASSERT_SUCCESS( rmmFree(a, stream) );
}

TEST_F(MemoryManagerTest, AllocationOffset) {
    char *a = nullptr, *b = nullptr;
    ptrdiff_t offset = -1;
    ASSERT_SUCCESS( rmmAlloc((void**)&a, size_kb, stream) );
    ASSERT_SUCCESS( rmmAlloc((void**)&b, size_kb, stream) );

    ASSERT_SUCCESS( rmmGetAllocationOffset(&offset, a, stream) );
    ASSERT_GE(offset, 0);

    ASSERT_SUCCESS( rmmGetAllocationOffset(&offset, b, stream) );
    ASSERT_GE(offset, 0);

    ASSERT_SUCCESS( rmmFree(a, stream) );
    ASSERT_SUCCESS( rmmFree(b, stream) );
}
