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
#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

namespace {

using namespace cudf::test;

template <typename T>
using fwc_wrapper = cudf::test::fixed_width_column_wrapper<T>;

template <typename T>
using fpc_wrapper = cudf::test::fixed_point_column_wrapper<T>;

struct BinTestFixture : public BaseFixture {
};

/**
 * Test error cases.
 */

// Left edges type check.

TEST(BinColumnTest, TestInvalidLeft)
{
  fwc_wrapper<double> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fwc_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  fwc_wrapper<float> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  EXPECT_THROW(cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO),
               cudf::logic_error);
};

// Right edges type check.
TEST(BinColumnTest, TestInvalidRight)
{
  fwc_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fwc_wrapper<double> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  fwc_wrapper<float> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  EXPECT_THROW(cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO),
               cudf::logic_error);
};

// Input type check.
TEST(BinColumnTest, TestInvalidInput)
{
  fwc_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fwc_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  fwc_wrapper<double> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  EXPECT_THROW(cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO),
               cudf::logic_error);
};

// Left edges cannot contain nulls.
TEST(BinColumnTest, TestLeftEdgesWithNulls)
{
  fwc_wrapper<float> left_edges{{0, 1, 2}, {0, 1, 0}};
  fwc_wrapper<float> right_edges{1, 2, 3};
  fwc_wrapper<double> input{0.5, 0.5, 0.5, 0.5};

  EXPECT_THROW(cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO),
               cudf::logic_error);
};

// Right edges cannot contain nulls.
TEST(BinColumnTest, TestRightEdgesWithNulls)
{
  fwc_wrapper<float> left_edges{0, 1, 2};
  fwc_wrapper<float> right_edges{{1, 2, 3}, {0, 1, 0}};
  fwc_wrapper<double> input{0.5, 0.5, 0.5, 0.5};

  EXPECT_THROW(cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO),
               cudf::logic_error);
};

// Number of left and right edges must match.
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

  std::unique_ptr<cudf::column> result =
    cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO);
  ASSERT_TRUE(result->size() == 2);
  ASSERT_TRUE(result->null_count() == 2);

  fwc_wrapper<cudf::size_type> expected{{0, 0}, {0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
};

// Values outside the bounds should be labeled NULL.
TEST(BinColumnTest, TestOutOfBoundsInput)
{
  fwc_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fwc_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  fwc_wrapper<float> input{8.5, 9.5, 10.5, 11.5};

  std::unique_ptr<cudf::column> result =
    cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO);
  ASSERT_TRUE(result->size() == 4);
  ASSERT_TRUE(result->null_count() == 2);

  fwc_wrapper<cudf::size_type> expected{{8, 9, 0, 0}, {1, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
};

/**
 * Test inclusion options.
 */

template <typename T>
struct BoundaryExclusionBinTestFixture : public BinTestFixture {
  fwc_wrapper<T> left_edges{0, 1, 2, 3, 4};
  fwc_wrapper<T> right_edges{1, 2, 3, 4, 5};
  fwc_wrapper<T> input{1, 1};

  std::unique_ptr<cudf::column> bin(cudf::inclusive left_inc, cudf::inclusive right_inc)
  {
    return cudf::bin(input, left_edges, left_inc, right_edges, right_inc);
  }
};

TYPED_TEST_CASE(BoundaryExclusionBinTestFixture, FloatingPointTypes);

// Boundary points when both bounds are excluded should be labeled null.
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

/**
 * Test valid data.
 */

// Empty input must return an empty output.
TEST(BinColumnTest, TestEmptyInput)
{
  fwc_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fwc_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  fwc_wrapper<float> input{};

  std::unique_ptr<cudf::column> result =
    cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO);
  ASSERT_TRUE(result->size() == 0);
};

// Null inputs must map to nulls.
TEST(BinColumnTest, TestInputWithNulls)
{
  fwc_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fwc_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  fwc_wrapper<float> input{{1.5, 2.5, 3.5, 4.5}, {0, 1, 0, 1}};

  std::unique_ptr<cudf::column> result =
    cudf::bin(input, left_edges, cudf::inclusive::NO, right_edges, cudf::inclusive::NO);
  ASSERT_TRUE(result->size() == 4);
  ASSERT_TRUE(result->null_count() == 2);

  fwc_wrapper<cudf::size_type> expected{{0, 2, 0, 4}, {0, 1, 0, 1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
};

template <typename T>
struct FloatingPointBinTestFixture : public BinTestFixture {
  void test()
  {
    fwc_wrapper<T> left_edges{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    fwc_wrapper<T> right_edges{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    fwc_wrapper<T> input{2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5};

    auto result =
      cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO);

    // Check that every element is placed in bin 2.
    fwc_wrapper<cudf::size_type> expected{{2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                                          {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());

    // Test the first and last bins to catch any off-by-one errors.
    fwc_wrapper<T> input_endpoints{0.1, 9.9};
    std::unique_ptr<cudf::column> result2 = cudf::bin(
      input_endpoints, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO);

    fwc_wrapper<cudf::size_type> expected2{{0, 9}, {1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, result2->view());
  }
};

TYPED_TEST_CASE(FloatingPointBinTestFixture, FloatingPointTypes);

TYPED_TEST(FloatingPointBinTestFixture, TestFloatingPointData) { this->test(); };

template <typename T>
struct IntegerBinTestFixture : public BinTestFixture {
  void test (cudf::inclusive left_inc, cudf::inclusive right_inc, fwc_wrapper<cudf::size_type> expected)
  {
    fwc_wrapper<T> left_edges{0, 2, 4, 6, 8};
    fwc_wrapper<T> right_edges{2, 4, 6, 8, 10};
    fwc_wrapper<T> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto result =
      cudf::bin(input, left_edges, left_inc, right_edges, right_inc);

    // Check that every element is placed in bin 2.
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
};

TYPED_TEST_CASE(IntegerBinTestFixture, IntegralTypesNotBool);

TYPED_TEST(IntegerBinTestFixture, TestIntegerDataIncludeLeft)
{
    this->test(cudf::inclusive::YES, cudf::inclusive::NO, {{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0}});
};

TYPED_TEST(IntegerBinTestFixture, TestIntegerDataIncludeRight)
{
    this->test(cudf::inclusive::NO, cudf::inclusive::YES, {{0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4}, {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
};

TYPED_TEST(IntegerBinTestFixture, TestIntegerDataIncludeNeither)
{
    this->test(cudf::inclusive::NO, cudf::inclusive::NO, {{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5}, {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}});
};

template <typename T>
struct FixedPointBinTestFixture : public BinTestFixture {
};

TYPED_TEST_CASE(FixedPointBinTestFixture, FixedPointTypes);

TYPED_TEST(FixedPointBinTestFixture, TestFixedPointData)
{
    using fpc_type_wrapper = fpc_wrapper<cudf::device_storage_type_t<TypeParam> >;

    fpc_type_wrapper left_edges{{0, 10, 20, 30, 40, 50, 60, 70, 80, 90}, numeric::scale_type{0}};
    fpc_type_wrapper right_edges{{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}, numeric::scale_type{0}};
    fpc_type_wrapper input{{25, 25, 25, 25, 25, 25, 25, 25, 25, 25}, numeric::scale_type{0}};

    auto result =
      cudf::bin(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO);

    // Check that every element is placed in bin 2.
    fwc_wrapper<cudf::size_type> expected{{2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
};


}  // anonymous namespace

CUDF_TEST_PROGRAM_MAIN()
