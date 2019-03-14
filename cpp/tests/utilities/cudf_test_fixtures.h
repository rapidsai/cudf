#pragma once

#include <gtest/gtest.h>

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
