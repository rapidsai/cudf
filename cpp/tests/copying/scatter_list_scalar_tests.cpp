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

#include <cudf/scalar/scalar_factories.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

namespace cudf {
namespace test {

using M    = std::vector<valid_type>;
using SM_t = fixed_width_column_wrapper<size_type>;

class ScatterListScalarTests : public cudf::test::BaseFixture {
};

void test_single_scalar_scatter(column_view const& target,
                                scalar const& slr,
                                column_view const& scatter_map,
                                column_view expect)
{
  std::vector<std::reference_wrapper<const scalar>> slrs{slr};
  table_view targets{{target}};
  auto result = scatter(slrs, scatter_map, targets, true);
  cudf::test::print(result->view().column(0));
  cudf::test::print(expect);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view().column(0), expect);
}

template <typename T>
class ScatterListOfFixedWidthScalarTest : public ScatterListScalarTests {
};

TYPED_TEST_CASE(ScatterListOfFixedWidthScalarTest, FixedWidthTypesWithoutFixedPoint);

// Test grid
// Dim1 : {Fixed width, strings, lists, structs}
// Dim2 : {Null scalar, Non-null empty scalar, Non-null non-empty scalar}
// Dim3 : {Nullable target, non-nullable target row}

TYPED_TEST(ScatterListOfFixedWidthScalarTest, Basic)
{
  using LCW = lists_column_wrapper<TypeParam, int32_t>;
  using FCW = fixed_width_column_wrapper<TypeParam>;

  auto slr = std::make_unique<list_scalar>(FCW({2, 2, 2}, {1, 0, 1}), true);
  LCW col{LCW{1, 1, 1}, LCW{8, 8}, LCW{10, 10, 10, 10}, LCW{5}};
  SM_t scatter_map{3, 1, 0};

  LCW expected{LCW({2, 2, 2}, M{1, 0, 1}.begin()),
               LCW({2, 2, 2}, M{1, 0, 1}.begin()),
               LCW{10, 10, 10, 10},
               LCW({2, 2, 2}, M{1, 0, 1}.begin())};
  test_single_scalar_scatter(col, *slr, scatter_map, expected);
}

TYPED_TEST(ScatterListOfFixedWidthScalarTest, EmptyValidScalar)
{
  using LCW = lists_column_wrapper<TypeParam, int32_t>;
  using FCW = fixed_width_column_wrapper<TypeParam>;

  auto slr = std::make_unique<list_scalar>(FCW{}, true);
  LCW col{
    LCW{1, 1, 1}, LCW{8, 8}, LCW({10, 10, 10, 10}, M{1, 0, 1, 0}.begin()), LCW{5}, LCW{42, 42}};
  SM_t scatter_map{1, 0};

  LCW expected{LCW{}, LCW{}, LCW({10, 10, 10, 10}, M{1, 0, 1, 0}.begin()), LCW{5}, LCW{42, 42}};
  test_single_scalar_scatter(col, *slr, scatter_map, expected);
}

TYPED_TEST(ScatterListOfFixedWidthScalarTest, NullScalar)
{
  using LCW = lists_column_wrapper<TypeParam, int32_t>;
  using FCW = fixed_width_column_wrapper<TypeParam>;

  auto slr = std::make_unique<list_scalar>(FCW{}, false);
  LCW col{LCW({1, 1, 1}, M{0, 0, 1}.begin()), LCW{8, 8}, LCW{10, 10, 10, 10}, LCW{5}};
  SM_t scatter_map{3, 1};

  LCW expected({LCW({1, 1, 1}, M{0, 0, 1}.begin()), LCW{}, LCW{10, 10, 10, 10}, LCW{}},
               M{1, 0, 1, 0}.begin());
  test_single_scalar_scatter(col, *slr, scatter_map, expected);
}

TYPED_TEST(ScatterListOfFixedWidthScalarTest, NullableTargetRow)
{
  using LCW = lists_column_wrapper<TypeParam, int32_t>;
  using FCW = fixed_width_column_wrapper<TypeParam>;

  auto slr = std::make_unique<list_scalar>(FCW{9, 9}, true);
  LCW col({LCW{4, 4}, LCW{}, LCW{8, 8, 8}, LCW{}, LCW{9, 9, 9}}, M{1, 0, 1, 0, 1}.begin());
  SM_t scatter_map{0, 1};

  LCW expected({LCW{9, 9}, LCW{9, 9}, LCW{8, 8, 8}, LCW{}, LCW{9, 9, 9}}, M{1, 1, 1, 0, 1}.begin());
  test_single_scalar_scatter(col, *slr, scatter_map, expected);
}

class ScatterListOfStringScalarTest : public ScatterListScalarTests {
};

TEST_F(ScatterListOfStringScalarTest, Basic)
{
  using LCW      = lists_column_wrapper<string_view, int32_t>;
  using StringCW = strings_column_wrapper;

  auto slr = std::make_unique<list_scalar>(
    StringCW({"Hello!", "", "你好！", "صباح الخير!", "", "こんにちは！"},
             {true, false, true, true, false, true}),
    true);
  LCW col{LCW({"xx", "yy"}, M{0, 1}.begin()), LCW{""}, LCW{"a", "bab", "bacab"}};

  SM_t scatter_map{2, 1};

  LCW expected{
    LCW({"xx", "yy"}, M{0, 1}.begin()),
    LCW({"Hello!", "", "你好！", "صباح الخير!", "", "こんにちは！"}, M{1, 0, 1, 1, 0, 1}.begin()),
    LCW({"Hello!", "", "你好！", "صباح الخير!", "", "こんにちは！"}, M{1, 0, 1, 1, 0, 1}.begin())};

  test_single_scalar_scatter(col, *slr, scatter_map, expected);
}

TEST_F(ScatterListOfStringScalarTest, EmptyValidScalar)
{
  using LCW      = lists_column_wrapper<string_view, int32_t>;
  using StringCW = strings_column_wrapper;

  auto slr = std::make_unique<list_scalar>(StringCW{}, true);

  LCW col{LCW({"xx", "yy"}, M{0, 1}.begin()), LCW{""}, LCW{"a", "bab", "bacab"}, LCW{"888", "777"}};

  SM_t scatter_map{0, 3};

  LCW expected{LCW{}, LCW{""}, LCW{"a", "bab", "bacab"}, LCW{}};

  test_single_scalar_scatter(col, *slr, scatter_map, expected);
}

TEST_F(ScatterListOfStringScalarTest, NullScalar)
{
  using LCW      = lists_column_wrapper<string_view, int32_t>;
  using StringCW = strings_column_wrapper;

  auto slr = std::make_unique<list_scalar>(StringCW{}, false);
  LCW col{LCW{"xx", "yy"}, LCW({""}, M{0}.begin()), LCW{"a", "bab", "bacab"}, LCW{"888", "777"}};

  SM_t scatter_map{1, 2};

  LCW expected({LCW{"xx", "yy"}, LCW{}, LCW{}, LCW{"888", "777"}}, M{1, 0, 0, 1}.begin());

  test_single_scalar_scatter(col, *slr, scatter_map, expected);
}

TEST_F(ScatterListOfStringScalarTest, NullableTargetRow)
{
  using LCW      = lists_column_wrapper<string_view, int32_t>;
  using StringCW = strings_column_wrapper;

  auto slr = std::make_unique<list_scalar>(
    StringCW({"Hello!", "", "こんにちは！"}, {true, false, true}), true);
  LCW col({LCW{"xx", "yy"}, LCW({""}, M{0}.begin()), LCW{}, LCW{"888", "777"}},
          M{1, 1, 0, 1}.begin());

  SM_t scatter_map{3, 2};

  LCW expected({LCW{"xx", "yy"},
                LCW({""}, M{0}.begin()),
                LCW({"Hello!", "", "こんにちは！"}, M{1, 0, 1}.begin()),
                LCW({"Hello!", "", "こんにちは！"}, M{1, 0, 1}.begin())},
               M{1, 1, 1, 1}.begin());

  test_single_scalar_scatter(col, *slr, scatter_map, expected);
}

template <typename T>
class ScatterListOfListScalarTest : public ScatterListScalarTests {
};

TYPED_TEST_CASE(ScatterListOfListScalarTest, FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(ScatterListOfListScalarTest, Basic)
{
  using LCW = lists_column_wrapper<TypeParam, int32_t>;

  auto slr = std::make_unique<list_scalar>(
    LCW({LCW{1, 2, 3}, LCW{4}, LCW{}, LCW{5, 6}}, M{1, 1, 0, 1}.begin()), true);
  LCW col({LCW({LCW{88, 88}, LCW{}, LCW{9, 9, 9}}, M{1, 0, 1}.begin()),
           LCW{LCW{66}, LCW{}, LCW({77, 77, 77, 77}, M{1, 0, 0, 1}.begin())},
           LCW{LCW{55, 55}, LCW{}, LCW{10, 10, 10}},
           LCW{LCW{44, 44}}});

  SM_t scatter_map{1, 2, 3};

  LCW expected({LCW({LCW{88, 88}, LCW{}, LCW{9, 9, 9}}, M{1, 0, 1}.begin()),
                LCW({LCW{1, 2, 3}, LCW{4}, LCW{}, LCW{5, 6}}, M{1, 1, 0, 1}.begin()),
                LCW({LCW{1, 2, 3}, LCW{4}, LCW{}, LCW{5, 6}}, M{1, 1, 0, 1}.begin()),
                LCW({LCW{1, 2, 3}, LCW{4}, LCW{}, LCW{5, 6}}, M{1, 1, 0, 1}.begin())});

  test_single_scalar_scatter(col, *slr, scatter_map, expected);
}

TYPED_TEST(ScatterListOfListScalarTest, EmptyValidScalar)
{
  using LCW = lists_column_wrapper<TypeParam, int32_t>;

  auto slr = std::make_unique<list_scalar>(LCW{}, true);
  LCW col({LCW({LCW{88, 88}, LCW{}, LCW{9, 9, 9}}, M{1, 0, 1}.begin()),
           LCW{LCW{66}, LCW{}, LCW({77, 77, 77, 77}, M{1, 0, 0, 1}.begin())},
           LCW{LCW{55, 55}, LCW{}, LCW{10, 10, 10}},
           LCW{LCW{44, 44}}});

  SM_t scatter_map{3, 0};

  LCW expected({LCW{},
                LCW{LCW{66}, LCW{}, LCW({77, 77, 77, 77}, M{1, 0, 0, 1}.begin())},
                LCW{LCW{55, 55}, LCW{}, LCW{10, 10, 10}},
                LCW{}});

  test_single_scalar_scatter(col, *slr, scatter_map, expected);
}

TYPED_TEST(ScatterListOfListScalarTest, NullScalar)
{
  using LCW = lists_column_wrapper<TypeParam, int32_t>;

  auto slr = std::make_unique<list_scalar>(LCW{}, false);
  LCW col({LCW({LCW{88, 88}, LCW{}, LCW{9, 9, 9}}, M{1, 0, 1}.begin()),
           LCW{LCW{66}, LCW{}, LCW({77, 77, 77, 77}, M{1, 0, 0, 1}.begin())},
           LCW{LCW{44, 44}}});

  SM_t scatter_map{1, 0};

  LCW expected({LCW{}, LCW{}, LCW{LCW{44, 44}}}, M{0, 0, 1}.begin());

  test_single_scalar_scatter(col, *slr, scatter_map, expected);
}

TYPED_TEST(ScatterListOfListScalarTest, NullableTargetRows)
{
  using LCW = lists_column_wrapper<TypeParam, int32_t>;

  auto slr = std::make_unique<list_scalar>(
    LCW({LCW{1, 1, 1}, LCW{3, 3}, LCW{}, LCW{4}}, M{1, 1, 0, 1}.begin()), true);

  LCW col({LCW({LCW{88, 88}, LCW{}, LCW{9, 9, 9}}, M{1, 0, 1}.begin()),
           LCW{LCW{66}, LCW{}, LCW({77, 77, 77, 77}, M{1, 0, 0, 1}.begin())},
           LCW{LCW{44, 44}}},
          M{1, 0, 1}.begin());

  SM_t scatter_map{1};

  LCW expected({LCW({LCW{88, 88}, LCW{}, LCW{9, 9, 9}}, M{1, 0, 1}.begin()),
                LCW({LCW{1, 1, 1}, LCW{3, 3}, LCW{}, LCW{4}}, M{1, 1, 0, 1}.begin()),
                LCW{LCW{44, 44}}},
               M{1, 1, 1}.begin());

  test_single_scalar_scatter(col, *slr, scatter_map, expected);
}

}  // namespace test
}  // namespace cudf
