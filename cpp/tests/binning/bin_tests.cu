/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/binning/bin.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <numeric>
#include <thrust/logical.h>
#include <cudf/column/column.hpp>
#include <cudf_test/type_lists.hpp>
#include <thrust/execution_policy.h>
#include <thrust/execution_policy.h>
#include <cudf/types.hpp>
#include <stdio.h>


namespace {

using namespace cudf::test;

template <typename T>
using fwc_wrapper = cudf::test::fixed_width_column_wrapper<T>;

// TODO: Maybe test more types.
using ValidBinTypes = FloatingPointTypes;

// =============================================================================
// ----- Define standard fixture for all bin tests -----------------------------

struct BinTestFixture : public BaseFixture {
};

template <typename T>
struct TypedBinTestFixture : public BinTestFixture {
};


// =============================================================================
// ----- helper functions ------------------------------------------------------

/// A simple struct to be used as a predicate for comparing a sequence to a given value encoded by this struct in algorithms.
struct equal_value
{
    equal_value(unsigned int value)
    {
        m_value = value;
    }

    __device__
    bool operator()(unsigned int x) const
    {
        return x == m_value;
    }

    unsigned int m_value; /// The value to compare with.
};


// =============================================================================
// ----- tests -----------------------------------------------------------------

// ----- test error cases ------------------------------------------------------

/// Left edges type check.

TEST(BinColumnTest, TestInvalidLeft)
{
    fwc_wrapper<double> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    fwc_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    fwc_wrapper<float> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

    EXPECT_THROW(cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO),
            cudf::logic_error);
};


/// Right edges type check.
TEST(BinColumnTest, TestInvalidRight)
{
    fwc_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    fwc_wrapper<double> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    fwc_wrapper<float> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

    EXPECT_THROW(cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO),
            cudf::logic_error);
};

/// Input type check.
TEST(BinColumnTest, TestInvalidInput)
{
    fwc_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    fwc_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    fwc_wrapper<double> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

    EXPECT_THROW(cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO),
            cudf::logic_error);
};

/// Number of left and right edges must match.
TEST(BinColumnTest, TestMismatchedEdges)
{
    fwc_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    fwc_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9};
    fwc_wrapper<float> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

    EXPECT_THROW(cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO),
            cudf::logic_error);
};

// If no edges are provided, the bin for all inputs is null.
TEST(BinColumnTest, TestEmptyEdges)
{
    fwc_wrapper<float> left_edges{};
    fwc_wrapper<float> right_edges{};
    fwc_wrapper<float> input{0.5, 0.5};

    std::unique_ptr<cudf::column> result = cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::YES);
    ASSERT_TRUE(result->size() == 2);
    ASSERT_TRUE(result->null_count() == 2);

    fwc_wrapper<cudf::size_type> expected{{0, 0}, {0, 0}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
};

// ----- test inclusion options -----------------------------------------------

template <typename T>
struct BoundaryExclusionBinTestFixture : public TypedBinTestFixture<T> {
    fwc_wrapper<T> left_edges{0, 1, 2, 3, 4};
    fwc_wrapper<T> right_edges{1, 2, 3, 4, 5};
    fwc_wrapper<T> input{1, 1};

    std::unique_ptr<cudf::column> bin(cudf::inclusive left_inc, cudf::inclusive right_inc)
    {
        return cudf::bin(input, left_edges, left_inc, right_edges, right_inc);
    }
};

TYPED_TEST_CASE(BoundaryExclusionBinTestFixture, ValidBinTypes);

// Boundary points when both bounds are excluded should be null.
TYPED_TEST(BoundaryExclusionBinTestFixture, TestNoIncludes)
{
    auto result = this->bin(cudf::inclusive::NO, cudf::inclusive::NO);
    ASSERT_TRUE(result->size() == 2);
    ASSERT_TRUE(result->null_count() == 2);

    fwc_wrapper<cudf::size_type> expected{{0, 0}, {0, 0}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
};

// Boundary point 1 should be in bin 1 [1, 2).
TYPED_TEST(BoundaryExclusionBinTestFixture, TestIncludeLeft)
{
    auto result = this->bin(cudf::inclusive::YES, cudf::inclusive::NO);
    ASSERT_TRUE(result->size() == 2);
    ASSERT_TRUE(result->null_count() == 0);

    fwc_wrapper<cudf::size_type> expected{{1, 1}, {1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
};

// Boundary point 1 should be in bin 0 (0, 1].
TYPED_TEST(BoundaryExclusionBinTestFixture, TestIncludeRight)
{
    auto result = this->bin(cudf::inclusive::NO, cudf::inclusive::YES);
    ASSERT_TRUE(result->size() == 2);
    ASSERT_TRUE(result->null_count() == 0);

    fwc_wrapper<cudf::size_type> expected{{0, 0}, {1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
};

// ----- test valid data ------------------------------------------------------

/// Empty input must return an empty output.
TEST(BinColumnTest, TestEmptyInput)
{
    fwc_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    fwc_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    fwc_wrapper<float> input{};

    std::unique_ptr<cudf::column> result = cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::YES);
    ASSERT_TRUE(result->size() == 0);
};

/// Null inputs must map to nulls.
TEST(BinColumnTest, TestInputWithNulls)
{
    fwc_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    fwc_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    fwc_wrapper<float> input{{1.5, 2.5, 3.5, 4.5}, {0, 1, 0, 1}};

    std::unique_ptr<cudf::column> result = cudf::bin(input, left_edges, cudf::inclusive::NO, right_edges, cudf::inclusive::NO);
    ASSERT_TRUE(result->size() == 4);
    ASSERT_TRUE(result->null_count() == 2);

    fwc_wrapper<cudf::size_type> expected{{0, 2, 0, 4}, {0, 1, 0, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
};

template <typename T>
struct FloatingPointBinTestFixture : public TypedBinTestFixture<T> {
    fwc_wrapper<T> left_edges{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    fwc_wrapper<T> right_edges{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    fwc_wrapper<T> input{2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5};
};

// TODO: Add parameterized/fuzzing tests if we have a consistent way to add those.
// TODO: Add tests for non-numeric types. Need to decide what types will be supported and how.
// TODO: Add tests for values outside the bounds.
// TODO: What happens if any bins contain nulls?

TYPED_TEST_CASE(FloatingPointBinTestFixture, FloatingPointTypes);

TYPED_TEST(FloatingPointBinTestFixture, TestFloatingPointData)
{
    // TODO: For some reason, auto doesn't work here. It _did_ work prior to my
    // turning this into a parameterized test, so my best gues is that some of
    // the template magic that Google Test is doing is making automatic type
    // detection fail.
    std::unique_ptr<cudf::column> result = cudf::bin(
            this->input,
            this->left_edges,
            cudf::inclusive::YES,
            this->right_edges,
            cudf::inclusive::YES);
    // Check that every element is placed in bin 2.
    auto begin = result->view().begin<const unsigned int>();
    auto end = result->view().end<const unsigned int>();
    ASSERT_TRUE(thrust::all_of(thrust::device, begin, end, equal_value(2)));
};

TEST(BinColumnTest, KnownFailure)
{
    fwc_wrapper<float> left_edges{3.5, 6.0};
    fwc_wrapper<float> right_edges{0.999, 3.5};
    fwc_wrapper<float> input{1, 2, 3, 4, 5, 6};

    std::unique_ptr<cudf::column> result = cudf::bin(input, left_edges, cudf::inclusive::NO, right_edges, cudf::inclusive::YES);

    unsigned int *tmp = (unsigned int *) malloc(6 * sizeof(unsigned int));
    cudaMemcpy(tmp, result->view().data<unsigned int>(), 6, cudaMemcpyDeviceToHost);
    printf("The values are %d, %d, %d, %d, %d, %d.\n", tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]);

};

}  // anonymous namespace

CUDF_TEST_PROGRAM_MAIN()
