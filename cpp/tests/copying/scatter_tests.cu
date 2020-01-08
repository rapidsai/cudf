/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/copying.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>


class ScatterUntypedTests : public cudf::test::BaseFixture {};

// Throw logic error if scatter map is longer than source
TEST_F(ScatterUntypedTests, ScatterMapTooLong)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;

  fixed_width_column_wrapper<int32_t> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1, 0, 2, 4, 6});

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});

  EXPECT_THROW(cudf::experimental::scatter(source_table, scatter_map,
    target_table, true), cudf::logic_error);
}

// Throw logic error if scatter map has nulls
TEST_F(ScatterUntypedTests, ScatterMapNulls)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;

  fixed_width_column_wrapper<int32_t> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1}, {0, 1, 1, 1});

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});

  EXPECT_THROW(cudf::experimental::scatter(source_table, scatter_map,
    target_table, true), cudf::logic_error);
}

// Throw logic error if scatter map has nulls
TEST_F(ScatterUntypedTests, ScatterScalarMapNulls)
{
  using cudf::experimental::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using scalar_ptr = std::unique_ptr<cudf::scalar>;
  using scalar_vector = std::vector<scalar_ptr>;

  // Initializers lists can't take move-only types
  scalar_vector source_vector;
  auto source = scalar_ptr(new scalar_type_t<int32_t>(100));
  source_vector.push_back(std::move(source));

  fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1}, {0, 1, 1, 1});

  auto const target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true), cudf::logic_error);
}

// Throw logic error if source and target have different number of columns
TEST_F(ScatterUntypedTests, ScatterColumnNumberMismatch)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;

  fixed_width_column_wrapper<int32_t> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});

  auto const source_table = cudf::table_view({source});
  auto const target_table = cudf::table_view({target, target});

  EXPECT_THROW(cudf::experimental::scatter(source_table, scatter_map,
    target_table, true), cudf::logic_error);
}

// Throw logic error if number of scalars doesn't match number of columns
TEST_F(ScatterUntypedTests, ScatterScalarColumnNumberMismatch)
{
  using cudf::experimental::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using scalar_ptr = std::unique_ptr<cudf::scalar>;
  using scalar_vector = std::vector<scalar_ptr>;

  // Initializers lists can't take move-only types
  scalar_vector source_vector;
  auto source = scalar_ptr(new scalar_type_t<int32_t>(100));
  source_vector.push_back(std::move(source));

  fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});

  auto const target_table = cudf::table_view({target, target});

  EXPECT_THROW(cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true), cudf::logic_error);
}

// Throw logic error if source and target have different data types
TEST_F(ScatterUntypedTests, ScatterDataTypeMismatch)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;

  fixed_width_column_wrapper<int32_t> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<float> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});

  auto const source_table = cudf::table_view({source});
  auto const target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::experimental::scatter(source_table, scatter_map,
    target_table, true), cudf::logic_error);
}

// Throw logic error if source and target have different data types
TEST_F(ScatterUntypedTests, ScatterScalarDataTypeMismatch)
{
  using cudf::experimental::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using scalar_ptr = std::unique_ptr<cudf::scalar>;
  using scalar_vector = std::vector<scalar_ptr>;

  // Initializers lists can't take move-only types
  scalar_vector source_vector;
  auto source = scalar_ptr(new scalar_type_t<int32_t>(100));
  source_vector.push_back(std::move(source));

  fixed_width_column_wrapper<float> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});

  auto const target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true), cudf::logic_error);
}

template <typename T>
class ScatterIndexTypeTests : public cudf::test::BaseFixture {};

using IndexTypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_CASE(ScatterIndexTypeTests, IndexTypes);

// Throw logic error if check_bounds is set and index is out of bounds
TYPED_TEST(ScatterIndexTypeTests, ScatterOutOfBounds)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;

  fixed_width_column_wrapper<TypeParam> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<TypeParam> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<TypeParam> upper_bound({-3, 3, 1, 8});
  fixed_width_column_wrapper<TypeParam> lower_bound({-3, 3, 1, -9});

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});

  EXPECT_THROW(cudf::experimental::scatter(source_table, upper_bound,
    target_table, true), cudf::logic_error);
  EXPECT_THROW(cudf::experimental::scatter(source_table, lower_bound,
    target_table, true), cudf::logic_error);
}

// Throw logic error if check_bounds is set and index is out of bounds
TYPED_TEST(ScatterIndexTypeTests, ScatterScalarOutOfBounds)
{
  using cudf::experimental::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using scalar_ptr = std::unique_ptr<cudf::scalar>;
  using scalar_vector = std::vector<scalar_ptr>;

  // Initializers lists can't take move-only types
  scalar_vector source_vector;
  auto source = scalar_ptr(new scalar_type_t<TypeParam>(100));
  source_vector.push_back(std::move(source));

  fixed_width_column_wrapper<TypeParam> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<TypeParam> upper_bound({-3, 3, 1, 8});
  fixed_width_column_wrapper<TypeParam> lower_bound({-3, 3, 1, -9});

  auto const target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::experimental::scatter(source_vector, upper_bound,
    target_table, true), cudf::logic_error);
  EXPECT_THROW(cudf::experimental::scatter(source_vector, lower_bound,
    target_table, true), cudf::logic_error);
}

// Validate that each of the index types work
TYPED_TEST(ScatterIndexTypeTests, ScatterIndexType)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;

  fixed_width_column_wrapper<TypeParam> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<TypeParam> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<TypeParam> scatter_map({-3, 3, 1, -1});
  fixed_width_column_wrapper<TypeParam> expected({10, 3, 30, 2, 50, 1, 70, 4});

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::experimental::scatter(source_table, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

// Validate that each of the index types work
TYPED_TEST(ScatterIndexTypeTests, ScatterScalarIndexType)
{
  using cudf::experimental::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using scalar_ptr = std::unique_ptr<cudf::scalar>;
  using scalar_vector = std::vector<scalar_ptr>;

  // Initializers lists can't take move-only types
  scalar_vector source_vector;
  auto source = scalar_ptr(new scalar_type_t<TypeParam>(100));
  source_vector.push_back(std::move(source));

  fixed_width_column_wrapper<TypeParam> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<TypeParam> scatter_map({-3, 3, 1, -1});
  fixed_width_column_wrapper<TypeParam> expected({10, 100, 30, 100, 50, 100, 70, 100});

  auto const target_table = cudf::table_view({target});
  auto const expected_table = cudf::table_view({expected});

  auto const result = cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

template <typename T>
class ScatterInvalidIndexTypeTests : public cudf::test::BaseFixture {};

// NOTE string types hit static assert in fixed_width_column_wrapper
using InvalidIndexTypes = cudf::test::Concat<
    cudf::test::Types<float, double, cudf::experimental::bool8>,
    cudf::test::TimestampTypes>;
TYPED_TEST_CASE(ScatterInvalidIndexTypeTests, InvalidIndexTypes);

// Throw logic error if scatter map column has invalid data type
TYPED_TEST(ScatterInvalidIndexTypeTests, ScatterInvalidIndexType)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;

  fixed_width_column_wrapper<int32_t> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<TypeParam> scatter_map({-3, 3, 1, -1});

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});

  EXPECT_THROW(cudf::experimental::scatter(source_table, scatter_map,
    target_table, true), cudf::logic_error);
}

// Throw logic error if scatter map column has invalid data type
TYPED_TEST(ScatterInvalidIndexTypeTests, ScatterScalarInvalidIndexType)
{
  using cudf::experimental::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using scalar_ptr = std::unique_ptr<cudf::scalar>;
  using scalar_vector = std::vector<scalar_ptr>;
  using T = TypeParam;

  // Initializers lists can't take move-only types
  scalar_vector source_vector;
  auto source = scalar_ptr(new scalar_type_t<int32_t>(100));
  source_vector.push_back(std::move(source));

  auto const tmp = cudf::test::make_type_param_vector<T>({-3, 3, 1, -1});

  fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<T> scatter_map(tmp.begin(), tmp.end());

  auto const target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true), cudf::logic_error);
}

template <typename T>
class ScatterDataTypeTests : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(ScatterDataTypeTests, cudf::test::FixedWidthTypes);

// Empty scatter map returns copy of input
TYPED_TEST(ScatterDataTypeTests, EmptyScatterMap)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using T = TypeParam;

  auto const tmp1 = cudf::test::make_type_param_vector<T>({1, 2, 3, 4, 5, 6});
  auto const tmp2 = cudf::test::make_type_param_vector<T>({10, 20, 30, 40, 50, 60, 70, 80});

  fixed_width_column_wrapper<TypeParam> source(tmp1.begin(), tmp1.end());
  fixed_width_column_wrapper<TypeParam> target(tmp2.begin(), tmp2.end());
  fixed_width_column_wrapper<int32_t> scatter_map({});

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});

  auto const result = cudf::experimental::scatter(source_table, scatter_map,
    target_table, true);

  // Expect a copy of the input table
  expect_tables_equal(result->view(), target_table);
}

// Empty scatter map returns copy of input
TYPED_TEST(ScatterDataTypeTests, EmptyScalarScatterMap)
{
  using cudf::experimental::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using scalar_ptr = std::unique_ptr<cudf::scalar>;
  using scalar_vector = std::vector<scalar_ptr>;
  using T = TypeParam;

  // Initializers lists can't take move-only types
  scalar_vector source_vector;
  auto source = scalar_ptr(new scalar_type_t<T>(100));
  source_vector.push_back(std::move(source));

  auto const tmp = cudf::test::make_type_param_vector<T>({10, 20, 30, 40, 50, 60, 70, 80});

  fixed_width_column_wrapper<T> target(tmp.begin(), tmp.end());
  fixed_width_column_wrapper<int32_t> scatter_map({});

  auto const target_table = cudf::table_view({target});

  auto const result = cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true);

  // Expect a copy of the input table
  expect_tables_equal(result->view(), target_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterNoNulls)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using T = TypeParam;

  auto const tmp0 = cudf::test::make_type_param_vector<T>({1, 2, 3, 4, 5, 6});
  auto const tmp1 = cudf::test::make_type_param_vector<T>({10, 20, 30, 40, 50, 60, 70, 80});
  auto const tmp2 = cudf::test::make_type_param_vector<T>({10, 3, 30, 2, 50, 1, 70, 4});

  fixed_width_column_wrapper<TypeParam> source(tmp0.begin(), tmp0.end());
  fixed_width_column_wrapper<TypeParam> target(tmp1.begin(), tmp1.end());
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});
  fixed_width_column_wrapper<TypeParam> expected(tmp2.begin(), tmp2.end());

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::experimental::scatter(source_table, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterBothNulls)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using T = TypeParam;

  auto const tmp0 = cudf::test::make_type_param_vector<T>({2, 4, 6, 8});
  auto const tmp1 = cudf::test::make_type_param_vector<T>({10, 20, 30, 40, 50, 60, 70, 80});
  auto const tmp2 = cudf::test::make_type_param_vector<T>({10, 2, 30, 4, 50, 6, 70, 8});
  std::vector<bool> const v0{1, 1, 0, 0};
  std::vector<bool> const v1{0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<bool> const v2{0, 1, 0, 1, 1, 0, 1, 0};

  fixed_width_column_wrapper<T> source(tmp0.begin(), tmp0.end(), v0.begin());
  fixed_width_column_wrapper<T> target(tmp1.begin(), tmp1.end(), v1.begin());
  fixed_width_column_wrapper<int32_t> scatter_map({1, 3, -3, -1});
  fixed_width_column_wrapper<T> expected(tmp2.begin(), tmp2.end(), v2.end());

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::experimental::scatter(source_table, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterSourceNulls)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using T = TypeParam;

  auto const tmp0 = cudf::test::make_type_param_vector<T>({2, 4, 6, 8});
  auto const tmp1 = cudf::test::make_type_param_vector<T>({10, 20, 30, 40, 50, 60, 70, 80});
  auto const tmp2 = cudf::test::make_type_param_vector<T>({10, 2, 30, 4, 50, 6, 70, 8});
  std::vector<bool> const v0{1, 1, 0, 0};
  std::vector<bool> const v2{0, 1, 0, 1, 1, 1, 1, 1};

  fixed_width_column_wrapper<T> source(tmp0.begin(), tmp0.end(), v0.begin());
  fixed_width_column_wrapper<T> target(tmp1.begin(), tmp1.end());
  fixed_width_column_wrapper<int32_t> scatter_map({1, 3, -3, -1});
  fixed_width_column_wrapper<T> expected(tmp2.begin(), tmp2.end(), v2.end());

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::experimental::scatter(source_table, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterTargetNulls)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using T = TypeParam;

  auto const tmp0 = cudf::test::make_type_param_vector<T>({2, 4, 6, 8});
  auto const tmp1 = cudf::test::make_type_param_vector<T>({10, 20, 30, 40, 50, 60, 70, 80});
  auto const tmp2 = cudf::test::make_type_param_vector<T>({10, 2, 30, 4, 50, 6, 70, 8});
  std::vector<bool> const v1{0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<bool> const v2{0, 1, 0, 1, 1, 1, 1, 1};

  fixed_width_column_wrapper<T> source(tmp0.begin(), tmp0.end());
  fixed_width_column_wrapper<T> target(tmp1.begin(), tmp1.end(), v1.begin());
  fixed_width_column_wrapper<int32_t> scatter_map({1, 3, -3, -1});
  fixed_width_column_wrapper<T> expected(tmp2.begin(), tmp2.end(), v2.begin());

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::experimental::scatter(source_table, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterScalarNoNulls)
{
  using cudf::experimental::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using scalar_ptr = std::unique_ptr<cudf::scalar>;
  using scalar_vector = std::vector<scalar_ptr>;
  using T = TypeParam;

  // Initializers lists can't take move-only types
  scalar_vector source_vector;
  auto source = scalar_ptr(new scalar_type_t<TypeParam>(100));
  source_vector.push_back(std::move(source));

  auto const tmp1 = cudf::test::make_type_param_vector<T>({10, 20, 30, 40, 50, 60, 70, 80});
  auto const tmp2 = cudf::test::make_type_param_vector<T>({10, 100, 30, 100, 50, 100, 70, 100});

  fixed_width_column_wrapper<T> target(tmp1.begin(), tmp1.end());
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});
  fixed_width_column_wrapper<T> expected(tmp2.begin(), tmp2.end());

  auto const target_table = cudf::table_view({target});
  auto const expected_table = cudf::table_view({expected});

  auto const result = cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterScalarTargetNulls)
{
  using cudf::experimental::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using scalar_ptr = std::unique_ptr<cudf::scalar>;
  using scalar_vector = std::vector<scalar_ptr>;
  using T = TypeParam;

  // Initializers lists can't take move-only types
  scalar_vector source_vector;
  auto source = scalar_ptr(new scalar_type_t<TypeParam>(100));
  source_vector.push_back(std::move(source));

  auto const tmp1 = cudf::test::make_type_param_vector<T>({10, 20, 30, 40, 50, 60, 70, 80});
  auto const tmp2 = cudf::test::make_type_param_vector<T>({10, 100, 30, 100, 50, 100, 70, 100});
  std::vector<bool> const v1{0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<bool> const v2{0, 1, 0, 1, 1, 1, 1, 1};

  fixed_width_column_wrapper<T> target(tmp1.begin(), tmp1.end(), v1.begin());
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});
  fixed_width_column_wrapper<T> expected(tmp2.begin(), tmp2.end(), v2.begin());

  auto const target_table = cudf::table_view({target});
  auto const expected_table = cudf::table_view({expected});

  auto const result = cudf::experimental::scatter(source_vector, scatter_map, target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterScalarSourceNulls)
{
  using cudf::experimental::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using scalar_ptr = std::unique_ptr<cudf::scalar>;
  using scalar_vector = std::vector<scalar_ptr>;

  // Initializers lists can't take move-only types
  scalar_vector source_vector;
  auto source = scalar_ptr(new scalar_type_t<TypeParam>(100));
  source->set_valid(false);
  source_vector.push_back(std::move(source));

  auto const a = cudf::test::make_type_param_vector<TypeParam>({10, 20, 30, 40, 50, 60, 70, 80});
  auto const b = cudf::test::make_type_param_vector<TypeParam>({10, 100, 30, 100, 50, 100, 70, 100});
  std::vector<bool> v{0, 0, 0, 0, 1, 0, 1, 0};

  fixed_width_column_wrapper<TypeParam> target(a.begin(), a.end());
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});
  fixed_width_column_wrapper<TypeParam> expected(b.begin(), b.end(), v.begin());

  auto const target_table = cudf::table_view({target});
  auto const expected_table = cudf::table_view({expected});

  auto const result = cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterScalarBothNulls)
{
  using cudf::experimental::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using scalar_ptr = std::unique_ptr<cudf::scalar>;
  using scalar_vector = std::vector<scalar_ptr>;

  // Initializers lists can't take move-only types
  scalar_vector source_vector;
  auto source = scalar_ptr(new scalar_type_t<TypeParam>(100));
  source->set_valid(false);
  source_vector.push_back(std::move(source));

  auto const a = cudf::test::make_type_param_vector<TypeParam>({10, 20, 30, 40, 50, 60, 70, 80});
  std::vector<bool> b{0, 0, 0, 0, 1, 1, 1, 1};
  auto const c = cudf::test::make_type_param_vector<TypeParam>({10, 100, 30, 100, 50, 100, 70, 100});
  std::vector<bool> d{0, 0, 0, 0, 1, 0, 1, 0};

  fixed_width_column_wrapper<TypeParam> target(a.begin(), a.end(), b.begin());
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});
  fixed_width_column_wrapper<TypeParam> expected(c.begin(), c.end(), d.begin());

  auto const target_table = cudf::table_view({target});
  auto const expected_table = cudf::table_view({expected});

  auto const result = cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

class ScatterStringsTests : public cudf::test::BaseFixture {};

TEST_F(ScatterStringsTests, ScatterNoNulls)
{
  using cudf::test::strings_column_wrapper;
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;

  std::vector<const char*> h_source
    { "dog", "the", "jumps", "brown", "the" };
  strings_column_wrapper source(h_source.begin(), h_source.end());

  std::vector<const char*> h_target
    { "a", "quick", "fire", "fox", "browses", "over", "a", "lazy", "web" };
  strings_column_wrapper target(h_target.begin(), h_target.end());

  fixed_width_column_wrapper<int32_t> scatter_map({-1, -3, -5, 2, 0});

  std::vector<const char*> h_expected
    { "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog" };
  strings_column_wrapper expected(h_expected.begin(), h_expected.end());

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::experimental::scatter(source_table, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

TEST_F(ScatterStringsTests, ScatterScalarNoNulls)
{
  using cudf::string_scalar;
  using cudf::test::strings_column_wrapper;
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;
  using scalar_ptr = std::unique_ptr<cudf::scalar>;
  using scalar_vector = std::vector<scalar_ptr>;

  // Initializers lists can't take move-only types
  scalar_vector source_vector;
  auto source = scalar_ptr(new string_scalar{"buffalo"});
  source_vector.push_back(std::move(source));

  std::vector<const char*> h_target
    { "Buffalo", "bison", "Buffalo", "bison", "bully", "bully", "Buffalo", "bison" };
  strings_column_wrapper target(h_target.begin(), h_target.end());

  fixed_width_column_wrapper<int32_t> scatter_map({1, 3, -4, -3, -1});

  std::vector<const char*> h_expected
    { "Buffalo", "buffalo", "Buffalo", "buffalo", "buffalo", "buffalo", "Buffalo", "buffalo" };
  strings_column_wrapper expected(h_expected.begin(), h_expected.end());

  auto const target_table = cudf::table_view({target});
  auto const expected_table = cudf::table_view({expected});

  auto const result = cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}
