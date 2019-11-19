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

  auto const source = fixed_width_column_wrapper<int32_t>(
    {1, 2, 3, 4, 5, 6});
  auto const target = fixed_width_column_wrapper<int32_t>(
    {10, 20, 30, 40, 50, 60, 70, 80});
  auto const scatter_map = fixed_width_column_wrapper<int32_t>(
    {-3, 3, 1, -1, 0, 2, 4, 6});
  
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

  auto const source = fixed_width_column_wrapper<int32_t>(
    {1, 2, 3, 4, 5, 6});
  auto const target = fixed_width_column_wrapper<int32_t>(
    {10, 20, 30, 40, 50, 60, 70, 80});
  auto const scatter_map = fixed_width_column_wrapper<int32_t>(
    {-3, 3, 1, -1}, {0, 1, 1, 1});
  
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

  auto const target = fixed_width_column_wrapper<int32_t>(
    {10, 20, 30, 40, 50, 60, 70, 80});
  auto const scatter_map = fixed_width_column_wrapper<int32_t>(
    {-3, 3, 1, -1}, {0, 1, 1, 1});

  auto const target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true), cudf::logic_error);
}

// Throw logic error if source and target have different number of columns
TEST_F(ScatterUntypedTests, ScatterColumnNumberMismatch)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;

  auto const source = fixed_width_column_wrapper<int32_t>(
    {1, 2, 3, 4, 5, 6});
  auto const target = fixed_width_column_wrapper<int32_t>(
    {10, 20, 30, 40, 50, 60, 70, 80});
  auto const scatter_map = fixed_width_column_wrapper<int32_t>(
    {-3, 3, 1, -1});
  
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

  auto const target = fixed_width_column_wrapper<int32_t>(
    {10, 20, 30, 40, 50, 60, 70, 80});
  auto const scatter_map = fixed_width_column_wrapper<int32_t>(
    {-3, 3, 1, -1});

  auto const target_table = cudf::table_view({target, target});

  EXPECT_THROW(cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true), cudf::logic_error);
}

// Throw logic error if source and target have different data types
TEST_F(ScatterUntypedTests, ScatterDataTypeMismatch)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;

  auto const source = fixed_width_column_wrapper<int32_t>(
    {1, 2, 3, 4, 5, 6});
  auto const target = fixed_width_column_wrapper<float>(
    {10, 20, 30, 40, 50, 60, 70, 80});
  auto const scatter_map = fixed_width_column_wrapper<int32_t>(
    {-3, 3, 1, -1});
  
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

  auto const target = fixed_width_column_wrapper<float>(
    {10, 20, 30, 40, 50, 60, 70, 80});
  auto const scatter_map = fixed_width_column_wrapper<int32_t>(
    {-3, 3, 1, -1});

  auto const target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true), cudf::logic_error);
}

template <typename T>
class ScatterTests : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(ScatterTests, cudf::test::FixedWidthTypes);

/*TYPED_TEST(ScatterTests, ScatterMapOutOfBounds)
{
  // TODO
}*/

TYPED_TEST(ScatterTests, Basic)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;

  auto const source = fixed_width_column_wrapper<TypeParam>(
    {1, 2, 3, 4, 5, 6});
  auto const target = fixed_width_column_wrapper<TypeParam>(
    {10, 20, 30, 40, 50, 60, 70, 80});
  auto const scatter_map = fixed_width_column_wrapper<int32_t>(
    {-3, 3, 1, -1});
  auto const expected = fixed_width_column_wrapper<TypeParam>(
    {10, 3, 30, 2, 50, 1, 70, 4});
  
  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::experimental::scatter(source_table, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

TYPED_TEST(ScatterTests, BasicNulls)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;

  auto const source = fixed_width_column_wrapper<TypeParam>(
    {2, 4, 6, 8}, {1, 1, 0, 0});
  auto const target = fixed_width_column_wrapper<TypeParam>(
    {10, 20, 30, 40, 50, 60, 70, 80}, {0, 0, 0, 0, 1, 1, 1, 1});
  auto const scatter_map = fixed_width_column_wrapper<int32_t>(
    {1, 3, -3, -1});
  auto const expected = fixed_width_column_wrapper<TypeParam>(
    {10, 2, 30, 4, 50, 6, 70, 8}, {0, 1, 0, 1, 1, 0, 1, 0});
  
  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::experimental::scatter(source_table, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

TYPED_TEST(ScatterTests, BasicNullSource)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;

  auto const source = fixed_width_column_wrapper<TypeParam>(
    {2, 4, 6, 8}, {1, 1, 0, 0});
  auto const target = fixed_width_column_wrapper<TypeParam>(
    {10, 20, 30, 40, 50, 60, 70, 80});
  auto const scatter_map = fixed_width_column_wrapper<int32_t>(
    {1, 3, -3, -1});
  auto const expected = fixed_width_column_wrapper<TypeParam>(
    {10, 2, 30, 4, 50, 6, 70, 8}, {1, 1, 1, 1, 1, 0, 1, 0});
  
  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::experimental::scatter(source_table, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

TYPED_TEST(ScatterTests, BasicNullTarget)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::expect_tables_equal;

  auto const source = fixed_width_column_wrapper<TypeParam>(
    {2, 4, 6, 8});
  auto const target = fixed_width_column_wrapper<TypeParam>(
    {10, 20, 30, 40, 50, 60, 70, 80}, {0, 0, 0, 0, 1, 1, 1, 1});
  auto const scatter_map = fixed_width_column_wrapper<int32_t>(
    {1, 3, -3, -1});
  auto const expected = fixed_width_column_wrapper<TypeParam>(
    {10, 2, 30, 4, 50, 6, 70, 8}, {0, 1, 0, 1, 1, 1, 1, 1});
  
  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::experimental::scatter(source_table, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

TYPED_TEST(ScatterTests, BasicScalar)
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

  auto const target = fixed_width_column_wrapper<TypeParam>(
    {10, 20, 30, 40, 50, 60, 70, 80});
  auto const scatter_map = fixed_width_column_wrapper<int32_t>(
    {-3, 3, 1, -1});
  auto const expected = fixed_width_column_wrapper<TypeParam>(
    {10, 100, 30, 100, 50, 100, 70, 100});

  auto const target_table = cudf::table_view({target});
  auto const expected_table = cudf::table_view({expected});

  auto const result = cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

TYPED_TEST(ScatterTests, BasicScalarNulls)
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

  auto const target = fixed_width_column_wrapper<TypeParam>(
    {10, 20, 30, 40, 50, 60, 70, 80}, {0, 0, 0, 0, 1, 1, 1, 1});
  auto const scatter_map = fixed_width_column_wrapper<int32_t>(
    {-3, 3, 1, -1});
  auto const expected = fixed_width_column_wrapper<TypeParam>(
    {10, 100, 30, 100, 50, 100, 70, 100}, {0, 1, 0, 1, 1, 1, 1, 1});

  auto const target_table = cudf::table_view({target});
  auto const expected_table = cudf::table_view({expected});

  auto const result = cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}

TYPED_TEST(ScatterTests, BasicScalarNullSource)
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

  auto const target = fixed_width_column_wrapper<TypeParam>(
    {10, 20, 30, 40, 50, 60, 70, 80});
  auto const scatter_map = fixed_width_column_wrapper<int32_t>(
    {-3, 3, 1, -1});
  auto const expected = fixed_width_column_wrapper<TypeParam>(
    {10, 100, 30, 100, 50, 100, 70, 100}, {1, 0, 1, 0, 1, 0, 1, 0});

  auto const target_table = cudf::table_view({target});
  auto const expected_table = cudf::table_view({expected});

  auto const result = cudf::experimental::scatter(source_vector, scatter_map,
    target_table, true);

  expect_tables_equal(result->view(), expected_table);
}
