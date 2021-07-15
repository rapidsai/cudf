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

#include <cudf/binaryop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <tests/binaryop/assert-binops.h>
#include <tests/binaryop/binop-fixture.hpp>
#include "cudf/utilities/error.hpp"

namespace cudf::test::binop {

template <typename T>
struct FixedPointCompiledTestBothReps : public cudf::test::BaseFixture {
};

template <typename T>
using wrapper = cudf::test::fixed_width_column_wrapper<T>;
TYPED_TEST_CASE(FixedPointCompiledTestBothReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpAdd)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto const sz = std::size_t{1000};

  auto begin      = cudf::detail::make_counting_transform_iterator(1, [](auto i) {
    return decimalXX{i, scale_type{0}};
  });
  auto const vec1 = std::vector<decimalXX>(begin, begin + sz);
  auto const vec2 = std::vector<decimalXX>(sz, decimalXX{2, scale_type{0}});
  auto expected   = std::vector<decimalXX>(sz);

  std::transform(std::cbegin(vec1),
                 std::cend(vec1),
                 std::cbegin(vec2),
                 std::begin(expected),
                 std::plus<decimalXX>());

  auto const lhs          = wrapper<decimalXX>(vec1.begin(), vec1.end());
  auto const rhs          = wrapper<decimalXX>(vec2.begin(), vec2.end());
  auto const expected_col = wrapper<decimalXX>(expected.begin(), expected.end());

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::ADD,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result =
    cudf::experimental::binary_operation(lhs, rhs, cudf::binary_operator::ADD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpMultiply)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto const sz = std::size_t{1000};

  auto begin      = cudf::detail::make_counting_transform_iterator(1, [](auto i) {
    return decimalXX{i, scale_type{0}};
  });
  auto const vec1 = std::vector<decimalXX>(begin, begin + sz);
  auto const vec2 = std::vector<decimalXX>(sz, decimalXX{2, scale_type{0}});
  auto expected   = std::vector<decimalXX>(sz);

  std::transform(std::cbegin(vec1),
                 std::cend(vec1),
                 std::cbegin(vec2),
                 std::begin(expected),
                 std::multiplies<decimalXX>());

  auto const lhs          = wrapper<decimalXX>(vec1.begin(), vec1.end());
  auto const rhs          = wrapper<decimalXX>(vec2.begin(), vec2.end());
  auto const expected_col = wrapper<decimalXX>(expected.begin(), expected.end());

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::MUL,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result =
    cudf::experimental::binary_operation(lhs, rhs, cudf::binary_operator::MUL, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col, result->view());
}

template <typename T>
using fp_wrapper = cudf::test::fixed_point_column_wrapper<T>;

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpMultiply2)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{11, 22, 33, 44, 55}, scale_type{-1}};
  auto const rhs      = fp_wrapper<RepType>{{10, 10, 10, 10, 10}, scale_type{0}};
  auto const expected = fp_wrapper<RepType>{{110, 220, 330, 440, 550}, scale_type{-1}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::MUL,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result =
    cudf::experimental::binary_operation(lhs, rhs, cudf::binary_operator::MUL, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpDiv)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{10, 30, 50, 70}, scale_type{-1}};
  auto const rhs      = fp_wrapper<RepType>{{4, 4, 4, 4}, scale_type{0}};
  auto const expected = fp_wrapper<RepType>{{2, 7, 12, 17}, scale_type{-1}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::DIV,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result =
    cudf::experimental::binary_operation(lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpDiv2)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{10, 30, 50, 70}, scale_type{-1}};
  auto const rhs      = fp_wrapper<RepType>{{4, 4, 4, 4}, scale_type{-2}};
  auto const expected = fp_wrapper<RepType>{{2, 7, 12, 17}, scale_type{1}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::DIV,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result =
    cudf::experimental::binary_operation(lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpDiv3)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{10, 30, 50, 70}, scale_type{-1}};
  auto const rhs      = make_fixed_point_scalar<decimalXX>(12, scale_type{-1});
  auto const expected = fp_wrapper<RepType>{{0, 2, 4, 5}, scale_type{0}};

  auto const type = cudf::binary_operation_fixed_point_output_type(
    cudf::binary_operator::DIV, static_cast<cudf::column_view>(lhs).type(), rhs->type());
  auto const result =
    cudf::experimental::binary_operation(lhs, *rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpDiv4)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto begin = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 11; });
  auto result_begin =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i * 11) / 12; });
  auto const lhs      = fp_wrapper<RepType>(begin, begin + 1000, scale_type{-1});
  auto const rhs      = make_fixed_point_scalar<decimalXX>(12, scale_type{-1});
  auto const expected = fp_wrapper<RepType>(result_begin, result_begin + 1000, scale_type{0});

  auto const type = cudf::binary_operation_fixed_point_output_type(
    cudf::binary_operator::DIV, static_cast<cudf::column_view>(lhs).type(), rhs->type());
  auto const result =
    cudf::experimental::binary_operation(lhs, *rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpAdd2)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{11, 22, 33, 44, 55}, scale_type{-1}};
  auto const rhs      = fp_wrapper<RepType>{{100, 200, 300, 400, 500}, scale_type{-2}};
  auto const expected = fp_wrapper<RepType>{{210, 420, 630, 840, 1050}, scale_type{-2}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::ADD,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result =
    cudf::experimental::binary_operation(lhs, rhs, cudf::binary_operator::ADD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpAdd3)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{1100, 2200, 3300, 4400, 5500}, scale_type{-3}};
  auto const rhs      = fp_wrapper<RepType>{{100, 200, 300, 400, 500}, scale_type{-2}};
  auto const expected = fp_wrapper<RepType>{{2100, 4200, 6300, 8400, 10500}, scale_type{-3}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::ADD,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result =
    cudf::experimental::binary_operation(lhs, rhs, cudf::binary_operator::ADD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpAdd4)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{11, 22, 33, 44, 55}, scale_type{-1}};
  auto const rhs      = make_fixed_point_scalar<decimalXX>(100, scale_type{-2});
  auto const expected = fp_wrapper<RepType>{{210, 320, 430, 540, 650}, scale_type{-2}};

  auto const type = cudf::binary_operation_fixed_point_output_type(
    cudf::binary_operator::ADD, static_cast<cudf::column_view>(lhs).type(), rhs->type());
  auto const result =
    cudf::experimental::binary_operation(lhs, *rhs, cudf::binary_operator::ADD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpAdd5)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = make_fixed_point_scalar<decimalXX>(100, scale_type{-2});
  auto const rhs      = fp_wrapper<RepType>{{11, 22, 33, 44, 55}, scale_type{-1}};
  auto const expected = fp_wrapper<RepType>{{210, 320, 430, 540, 650}, scale_type{-2}};

  auto const type = cudf::binary_operation_fixed_point_output_type(
    cudf::binary_operator::ADD, lhs->type(), static_cast<cudf::column_view>(rhs).type());
  auto const result =
    cudf::experimental::binary_operation(*lhs, rhs, cudf::binary_operator::ADD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpAdd6)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const col = fp_wrapper<RepType>{{30, 4, 5, 6, 7, 8}, scale_type{0}};

  auto const expected1 = fp_wrapper<RepType>{{60, 8, 10, 12, 14, 16}, scale_type{0}};
  auto const expected2 = fp_wrapper<RepType>{{6, 0, 1, 1, 1, 1}, scale_type{1}};
  auto const type1     = cudf::data_type{cudf::type_to_id<decimalXX>(), 0};
  auto const type2     = cudf::data_type{cudf::type_to_id<decimalXX>(), 1};
  auto const result1 =
    cudf::experimental::binary_operation(col, col, cudf::binary_operator::ADD, type1);
  auto const result2 =
    cudf::experimental::binary_operation(col, col, cudf::binary_operator::ADD, type2);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, result2->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, result1->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointCast)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const col      = fp_wrapper<RepType>{{6, 8, 10, 12, 14, 16}, scale_type{0}};
  auto const expected = fp_wrapper<RepType>{{0, 0, 1, 1, 1, 1}, scale_type{1}};
  auto const type     = cudf::data_type{cudf::type_to_id<decimalXX>(), 1};
  auto const result   = cudf::cast(col, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpMultiplyScalar)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{11, 22, 33, 44, 55}, scale_type{-1}};
  auto const rhs      = make_fixed_point_scalar<decimalXX>(100, scale_type{-1});
  auto const expected = fp_wrapper<RepType>{{1100, 2200, 3300, 4400, 5500}, scale_type{-2}};

  auto const type = cudf::binary_operation_fixed_point_output_type(
    cudf::binary_operator::MUL, static_cast<cudf::column_view>(lhs).type(), rhs->type());
  auto const result =
    cudf::experimental::binary_operation(lhs, *rhs, cudf::binary_operator::MUL, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpSimplePlus)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{150, 200}, scale_type{-2}};
  auto const rhs      = fp_wrapper<RepType>{{2250, 1005}, scale_type{-3}};
  auto const expected = fp_wrapper<RepType>{{3750, 3005}, scale_type{-3}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::ADD,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto const result =
    cudf::experimental::binary_operation(lhs, rhs, cudf::binary_operator::ADD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpEqualSimple)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const trues    = std::vector<bool>(4, true);
  auto const col1     = fp_wrapper<RepType>{{1, 2, 3, 4}, scale_type{0}};
  auto const col2     = fp_wrapper<RepType>{{100, 200, 300, 400}, scale_type{-2}};
  auto const expected = wrapper<bool>(trues.begin(), trues.end());

  auto const result = cudf::experimental::binary_operation(
    col1, col2, binary_operator::EQUAL, cudf::data_type{type_id::BOOL8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpEqualSimpleScale0)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const trues    = std::vector<bool>(4, true);
  auto const col      = fp_wrapper<RepType>{{1, 2, 3, 4}, scale_type{0}};
  auto const expected = wrapper<bool>(trues.begin(), trues.end());

  auto const result = cudf::experimental::binary_operation(
    col, col, binary_operator::EQUAL, cudf::data_type{type_id::BOOL8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpEqualSimpleScale0Null)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const col1     = fp_wrapper<RepType>{{1, 2, 3, 4}, {1, 1, 1, 1}, scale_type{0}};
  auto const col2     = fp_wrapper<RepType>{{1, 2, 3, 4}, {0, 0, 0, 0}, scale_type{0}};
  auto const expected = wrapper<bool>{{0, 1, 0, 1}, {0, 0, 0, 0}};

  auto const result = cudf::experimental::binary_operation(
    col1, col2, binary_operator::EQUAL, cudf::data_type{type_id::BOOL8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpEqualSimpleScale2Null)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const col1     = fp_wrapper<RepType>{{1, 2, 3, 4}, {1, 1, 1, 1}, scale_type{-2}};
  auto const col2     = fp_wrapper<RepType>{{1, 2, 3, 4}, {0, 0, 0, 0}, scale_type{0}};
  auto const expected = wrapper<bool>{{0, 1, 0, 1}, {0, 0, 0, 0}};

  auto const result = cudf::experimental::binary_operation(
    col1, col2, binary_operator::EQUAL, cudf::data_type{type_id::BOOL8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpEqualLessGreater)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const sz = std::size_t{1000};

  // TESTING binary op ADD

  auto begin = cudf::detail::make_counting_transform_iterator(1, [](auto e) { return e * 1000; });
  auto const vec1 = std::vector<RepType>(begin, begin + sz);
  auto const vec2 = std::vector<RepType>(sz, 0);

  auto const iota_3  = fp_wrapper<RepType>(vec1.begin(), vec1.end(), scale_type{-3});
  auto const zeros_3 = fp_wrapper<RepType>(vec2.begin(), vec2.end(), scale_type{-1});

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::ADD,
                                                   static_cast<cudf::column_view>(iota_3).type(),
                                                   static_cast<cudf::column_view>(zeros_3).type());
  auto const iota_3_after_add =
    cudf::experimental::binary_operation(zeros_3, iota_3, binary_operator::ADD, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(iota_3, iota_3_after_add->view());

  // TESTING binary op EQUAL, LESS, GREATER

  auto const trues    = std::vector<bool>(sz, true);
  auto const true_col = wrapper<bool>(trues.begin(), trues.end());

  auto const btype        = cudf::data_type{type_id::BOOL8};
  auto const equal_result = cudf::experimental::binary_operation(
    iota_3, iota_3_after_add->view(), binary_operator::EQUAL, btype);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(true_col, equal_result->view());

  auto const less_result = cudf::experimental::binary_operation(
    zeros_3, iota_3_after_add->view(), binary_operator::LESS, btype);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(true_col, less_result->view());

  auto const greater_result = cudf::experimental::binary_operation(
    iota_3_after_add->view(), zeros_3, binary_operator::GREATER, btype);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(true_col, greater_result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpNullMaxSimple)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const trues    = std::vector<bool>(4, true);
  auto const col1     = fp_wrapper<RepType>{{40, 30, 20, 10, 0}, {1, 0, 1, 1, 0}, scale_type{-2}};
  auto const col2     = fp_wrapper<RepType>{{10, 20, 30, 40, 0}, {1, 1, 1, 0, 0}, scale_type{-2}};
  auto const expected = fp_wrapper<RepType>{{40, 20, 30, 10, 0}, {1, 1, 1, 1, 0}, scale_type{-2}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::NULL_MAX,
                                                   static_cast<cudf::column_view>(col1).type(),
                                                   static_cast<cudf::column_view>(col2).type());
  auto const result =
    cudf::experimental::binary_operation(col1, col2, binary_operator::NULL_MAX, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpNullMinSimple)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const trues    = std::vector<bool>(4, true);
  auto const col1     = fp_wrapper<RepType>{{40, 30, 20, 10, 0}, {1, 1, 1, 0, 0}, scale_type{-1}};
  auto const col2     = fp_wrapper<RepType>{{10, 20, 30, 40, 0}, {1, 0, 1, 1, 0}, scale_type{-1}};
  auto const expected = fp_wrapper<RepType>{{10, 30, 20, 40, 0}, {1, 1, 1, 1, 0}, scale_type{-1}};

  auto const type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::NULL_MIN,
                                                   static_cast<cudf::column_view>(col1).type(),
                                                   static_cast<cudf::column_view>(col2).type());
  auto const result =
    cudf::experimental::binary_operation(col1, col2, binary_operator::NULL_MIN, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpNullEqualsSimple)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const trues    = std::vector<bool>(4, true);
  auto const col1     = fp_wrapper<RepType>{{400, 300, 300, 100}, {1, 1, 1, 0}, scale_type{-2}};
  auto const col2     = fp_wrapper<RepType>{{40, 200, 20, 400}, {1, 0, 1, 0}, scale_type{-1}};
  auto const expected = wrapper<bool>{{1, 0, 0, 1}, {1, 1, 1, 1}};

  auto const result = cudf::experimental::binary_operation(
    col1, col2, binary_operator::NULL_EQUALS, cudf::data_type{type_id::BOOL8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOp_Div)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{100, 300, 500, 700}, scale_type{-2}};
  auto const rhs      = fp_wrapper<RepType>{{4, 4, 4, 4}, scale_type{0}};
  auto const expected = fp_wrapper<RepType>{{25, 75, 125, 175}, scale_type{-2}};

  auto const type = data_type{type_to_id<decimalXX>(), -2};
  auto const result =
    cudf::experimental::binary_operation(lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOp_Div2)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{100000, 300000, 500000, 700000}, scale_type{-3}};
  auto const rhs      = fp_wrapper<RepType>{{20, 20, 20, 20}, scale_type{-1}};
  auto const expected = fp_wrapper<RepType>{{5000, 15000, 25000, 35000}, scale_type{-2}};

  auto const type = data_type{type_to_id<decimalXX>(), -2};
  auto const result =
    cudf::experimental::binary_operation(lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOp_Div3)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{10000, 30000, 50000, 70000}, scale_type{-2}};
  auto const rhs      = fp_wrapper<RepType>{{3, 9, 3, 3}, scale_type{0}};
  auto const expected = fp_wrapper<RepType>{{3333, 3333, 16666, 23333}, scale_type{-2}};

  auto const type = data_type{type_to_id<decimalXX>(), -2};
  auto const result =
    cudf::experimental::binary_operation(lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOp_Div4)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{10, 30, 50, 70}, scale_type{1}};
  auto const rhs      = make_fixed_point_scalar<decimalXX>(3, scale_type{0});
  auto const expected = fp_wrapper<RepType>{{3, 10, 16, 23}, scale_type{1}};

  auto const type = data_type{type_to_id<decimalXX>(), 1};
  auto const result =
    cudf::experimental::binary_operation(lhs, *rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOp_Div6)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs = make_fixed_point_scalar<decimalXX>(3000, scale_type{-3});
  auto const rhs = fp_wrapper<RepType>{{10, 30, 50, 70}, scale_type{-1}};

  auto const expected = fp_wrapper<RepType>{{300, 100, 60, 42}, scale_type{-2}};

  auto const type = data_type{type_to_id<decimalXX>(), -2};
  auto const result =
    cudf::experimental::binary_operation(*lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOp_Div7)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs = make_fixed_point_scalar<decimalXX>(1200, scale_type{0});
  auto const rhs = fp_wrapper<RepType>{{100, 200, 300, 500, 600, 800, 1200, 1300}, scale_type{-2}};

  auto const expected = fp_wrapper<RepType>{{12, 6, 4, 2, 2, 1, 1, 0}, scale_type{2}};

  auto const type = data_type{type_to_id<decimalXX>(), 2};
  auto const result =
    cudf::experimental::binary_operation(*lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOp_Div8)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{4000, 6000, 80000}, scale_type{-1}};
  auto const rhs      = make_fixed_point_scalar<decimalXX>(5000, scale_type{-3});
  auto const expected = fp_wrapper<RepType>{{0, 1, 16}, scale_type{2}};

  auto const type = data_type{type_to_id<decimalXX>(), 2};
  auto const result =
    cudf::experimental::binary_operation(lhs, *rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOp_Div9)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{10, 20, 30}, scale_type{2}};
  auto const rhs      = make_fixed_point_scalar<decimalXX>(7, scale_type{1});
  auto const expected = fp_wrapper<RepType>{{1, 2, 4}, scale_type{1}};

  auto const type = data_type{type_to_id<decimalXX>(), 1};
  auto const result =
    cudf::experimental::binary_operation(lhs, *rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOp_Div10)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{100, 200, 300}, scale_type{1}};
  auto const rhs      = make_fixed_point_scalar<decimalXX>(7, scale_type{0});
  auto const expected = fp_wrapper<RepType>{{14, 28, 42}, scale_type{1}};

  auto const type = data_type{type_to_id<decimalXX>(), 1};
  auto const result =
    cudf::experimental::binary_operation(lhs, *rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOp_Div11)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const lhs      = fp_wrapper<RepType>{{1000, 2000, 3000}, scale_type{1}};
  auto const rhs      = fp_wrapper<RepType>{{7, 7, 7}, scale_type{0}};
  auto const expected = fp_wrapper<RepType>{{142, 285, 428}, scale_type{1}};

  auto const type = data_type{type_to_id<decimalXX>(), 1};
  auto const result =
    cudf::experimental::binary_operation(lhs, rhs, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointCompiledTestBothReps, FixedPointBinaryOpThrows)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = device_storage_type_t<decimalXX>;

  auto const col           = fp_wrapper<RepType>{{100, 300, 500, 700}, scale_type{-2}};
  auto const non_bool_type = data_type{type_to_id<decimalXX>(), -2};
  auto const float_type    = data_type{type_id::FLOAT32};
  EXPECT_THROW(
    cudf::experimental::binary_operation(col, col, cudf::binary_operator::LESS, non_bool_type),
    cudf::logic_error);
  // Allowed now, but not allowed in jit.
  // EXPECT_THROW(cudf::experimental::binary_operation(col, col, cudf::binary_operator::MUL,
  // float_type),
  //              cudf::logic_error);
}

}  // namespace cudf::test::binop
