#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>

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

// GTest / GMock utilities

// Utility for testing the expectation that an expression x throws the specified
// exception whose what() message ends with the msg
#define EXPECT_THROW_MESSAGE(x, exception, startswith, endswith)     \
  EXPECT_THROW({                                                     \
    try { x; }                                                       \
    catch (const exception &e) {                                     \
    ASSERT_NE(nullptr, e.what());                                    \
    EXPECT_THAT(e.what(), testing::StartsWith((startswith)));        \
    EXPECT_THAT(e.what(), testing::EndsWith((endswith)));            \
    throw;                                                           \
  }}, exception);

#define CUDF_EXPECT_THROW_MESSAGE(x, msg) \
EXPECT_THROW_MESSAGE(x, cudf::logic_error, "cuDF failure at:", msg)

#define CUDA_EXPECT_THROW_MESSAGE(x, msg) \
EXPECT_THROW_MESSAGE(x, cudf::cuda_error, "CUDA error encountered at:", msg)
