#pragma once

#include "gtest/gtest.h"

#include <rmm/rmm.h>

// Base class fixture for GDF google tests that initializes / finalizes the memory manager
struct GdfTest : public ::testing::Test
{
    static void SetUpTestCase() {
        ASSERT_EQ( RMM_SUCCESS, rmmInitialize(nullptr) );
    }

    static void TearDownTestCase() {
        ASSERT_EQ( RMM_SUCCESS, rmmFinalize() );
    }
};

template <typename T>
struct cudfTest : public ::testing::Test
{
    static rmmAllocationMode_t allocationMode() { return T::alloc_mode; }

    static void SetUpTestCase() {
        rmmOptions_t options {allocationMode(), 0, false};
        ASSERT_EQ( RMM_SUCCESS, rmmInitialize(&options) );
    }

    static void TearDownTestCase() {
        ASSERT_EQ( RMM_SUCCESS, rmmFinalize() );
    }
};

template <rmmAllocationMode_t mode>
struct ModeType {
    static constexpr rmmAllocationMode_t alloc_mode{mode};
};

using allocation_modes =
    ::testing::Types< ModeType<CudaDefaultAllocation>,
                      ModeType<PoolAllocation>,
                      ModeType<CudaManagedMemory>,
                      ModeType<static_cast<rmmAllocationMode_t>(
                          PoolAllocation | CudaManagedMemory)>
                    >;
