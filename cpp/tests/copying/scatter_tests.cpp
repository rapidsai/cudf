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
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

class ScatterUntypedTests : public cudf::test::BaseFixture {
};

// Throw logic error if scatter map is longer than source
TEST_F(ScatterUntypedTests, ScatterMapTooLong)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<int32_t> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1, 0, 2, 4, 6});

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});

  EXPECT_THROW(cudf::scatter(source_table, scatter_map, target_table, true), cudf::logic_error);
}

// Throw logic error if scatter map has nulls
TEST_F(ScatterUntypedTests, ScatterMapNulls)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<int32_t> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1}, {0, 1, 1, 1});

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});

  EXPECT_THROW(cudf::scatter(source_table, scatter_map, target_table, true), cudf::logic_error);
}

// Throw logic error if scatter map has nulls
TEST_F(ScatterUntypedTests, ScatterScalarMapNulls)
{
  using cudf::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;

  auto const source = scalar_type_t<int32_t>{100};
  std::reference_wrapper<const cudf::scalar> slr_ref{source};
  std::vector<std::reference_wrapper<const cudf::scalar>> source_vector{slr_ref};

  fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1}, {0, 1, 1, 1});

  auto const target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::scatter(source_vector, scatter_map, target_table, true), cudf::logic_error);
}

// Throw logic error if source and target have different number of columns
TEST_F(ScatterUntypedTests, ScatterColumnNumberMismatch)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<int32_t> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});

  auto const source_table = cudf::table_view({source});
  auto const target_table = cudf::table_view({target, target});

  EXPECT_THROW(cudf::scatter(source_table, scatter_map, target_table, true), cudf::logic_error);
}

// Throw logic error if number of scalars doesn't match number of columns
TEST_F(ScatterUntypedTests, ScatterScalarColumnNumberMismatch)
{
  using cudf::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;

  auto const source = scalar_type_t<int32_t>(100);
  std::reference_wrapper<const cudf::scalar> slr_ref{source};
  std::vector<std::reference_wrapper<const cudf::scalar>> source_vector{slr_ref};

  fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});

  auto const target_table = cudf::table_view({target, target});

  EXPECT_THROW(cudf::scatter(source_vector, scatter_map, target_table, true), cudf::logic_error);
}

// Throw logic error if source and target have different data types
TEST_F(ScatterUntypedTests, ScatterDataTypeMismatch)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<int32_t> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<float> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});

  auto const source_table = cudf::table_view({source});
  auto const target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::scatter(source_table, scatter_map, target_table, true), cudf::logic_error);
}

// Throw logic error if source and target have different data types
TEST_F(ScatterUntypedTests, ScatterScalarDataTypeMismatch)
{
  using cudf::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;

  auto const source = scalar_type_t<int32_t>(100);
  std::reference_wrapper<const cudf::scalar> slr_ref{source};
  std::vector<std::reference_wrapper<const cudf::scalar>> source_vector{slr_ref};

  fixed_width_column_wrapper<float> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});

  auto const target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::scatter(source_vector, scatter_map, target_table, true), cudf::logic_error);
}

template <typename T>
class ScatterIndexTypeTests : public cudf::test::BaseFixture {
};

using IndexTypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_CASE(ScatterIndexTypeTests, IndexTypes);

// Throw logic error if check_bounds is set and index is out of bounds
TYPED_TEST(ScatterIndexTypeTests, ScatterOutOfBounds)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<TypeParam> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<TypeParam> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<TypeParam> upper_bound({-3, 3, 1, 8});
  fixed_width_column_wrapper<TypeParam> lower_bound({-3, 3, 1, -9});

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});

  EXPECT_THROW(cudf::scatter(source_table, upper_bound, target_table, true), cudf::logic_error);
  EXPECT_THROW(cudf::scatter(source_table, lower_bound, target_table, true), cudf::logic_error);
}

// Throw logic error if check_bounds is set and index is out of bounds
TYPED_TEST(ScatterIndexTypeTests, ScatterScalarOutOfBounds)
{
  using cudf::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;

  auto const source = scalar_type_t<TypeParam>(100, true);
  std::reference_wrapper<const cudf::scalar> slr_ref{source};
  std::vector<std::reference_wrapper<const cudf::scalar>> source_vector{slr_ref};

  fixed_width_column_wrapper<TypeParam> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<TypeParam> upper_bound({-3, 3, 1, 8});
  fixed_width_column_wrapper<TypeParam> lower_bound({-3, 3, 1, -9});

  auto const target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::scatter(source_vector, upper_bound, target_table, true), cudf::logic_error);
  EXPECT_THROW(cudf::scatter(source_vector, lower_bound, target_table, true), cudf::logic_error);
}

// Validate that each of the index types work
TYPED_TEST(ScatterIndexTypeTests, ScatterIndexType)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<TypeParam> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<TypeParam> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<TypeParam> scatter_map({-3, 3, 1, -1});
  fixed_width_column_wrapper<TypeParam> expected({10, 3, 30, 2, 50, 1, 70, 4});

  auto const source_table   = cudf::table_view({source, source});
  auto const target_table   = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::scatter(source_table, scatter_map, target_table, true);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), expected_table);
}

// Validate that each of the index types work
TYPED_TEST(ScatterIndexTypeTests, ScatterScalarIndexType)
{
  using cudf::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;

  auto const source = scalar_type_t<TypeParam>(100, true);
  std::reference_wrapper<const cudf::scalar> slr_ref{source};
  std::vector<std::reference_wrapper<const cudf::scalar>> source_vector{slr_ref};

  fixed_width_column_wrapper<TypeParam> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<TypeParam> scatter_map({-3, 3, 1, -1});
  fixed_width_column_wrapper<TypeParam> expected({10, 100, 30, 100, 50, 100, 70, 100});

  auto const target_table   = cudf::table_view({target});
  auto const expected_table = cudf::table_view({expected});

  auto const result = cudf::scatter(source_vector, scatter_map, target_table, true);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), expected_table);
}

template <typename T>
class ScatterInvalidIndexTypeTests : public cudf::test::BaseFixture {
};

// NOTE string types hit static assert in fixed_width_column_wrapper
using InvalidIndexTypes = cudf::test::Concat<cudf::test::Types<float, double, bool>,
                                             cudf::test::ChronoTypes,
                                             cudf::test::FixedPointTypes>;
TYPED_TEST_CASE(ScatterInvalidIndexTypeTests, InvalidIndexTypes);

// Throw logic error if scatter map column has invalid data type
TYPED_TEST(ScatterInvalidIndexTypeTests, ScatterInvalidIndexType)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<int32_t> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<TypeParam, int32_t> scatter_map({-3, 3, 1, -1});

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});

  EXPECT_THROW(cudf::scatter(source_table, scatter_map, target_table, true), cudf::logic_error);
}

// Throw logic error if scatter map column has invalid data type
TYPED_TEST(ScatterInvalidIndexTypeTests, ScatterScalarInvalidIndexType)
{
  using cudf::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;

  auto const source = scalar_type_t<int32_t>(100, true);
  std::reference_wrapper<const cudf::scalar> slr_ref{source};
  std::vector<std::reference_wrapper<const cudf::scalar>> source_vector{slr_ref};

  fixed_width_column_wrapper<int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<TypeParam, int32_t> scatter_map({-3, 3, 1, -1});

  auto const target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::scatter(source_vector, scatter_map, target_table, true), cudf::logic_error);
}

template <typename T>
class ScatterDataTypeTests : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(ScatterDataTypeTests, cudf::test::FixedWidthTypes);

// Empty scatter map returns copy of input
TYPED_TEST(ScatterDataTypeTests, EmptyScatterMap)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<TypeParam, int32_t> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<TypeParam, int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({});

  auto const source_table = cudf::table_view({source, source});
  auto const target_table = cudf::table_view({target, target});

  auto const result = cudf::scatter(source_table, scatter_map, target_table, true);

  // Expect a copy of the input table
  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), target_table);
}

// Empty scatter map returns copy of input
TYPED_TEST(ScatterDataTypeTests, EmptyScalarScatterMap)
{
  using cudf::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;

  auto const source =
    scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<TypeParam>(100), true);
  std::reference_wrapper<const cudf::scalar> slr_ref{source};
  std::vector<std::reference_wrapper<const cudf::scalar>> source_vector{slr_ref};

  fixed_width_column_wrapper<TypeParam, int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({});

  auto const target_table = cudf::table_view({target});

  auto const result = cudf::scatter(source_vector, scatter_map, target_table, true);

  // Expect a copy of the input table
  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), target_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterNoNulls)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<TypeParam, int32_t> source({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<TypeParam, int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});
  fixed_width_column_wrapper<TypeParam, int32_t> expected({10, 3, 30, 2, 50, 1, 70, 4});

  auto const source_table   = cudf::table_view({source, source});
  auto const target_table   = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::scatter(source_table, scatter_map, target_table, true);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), expected_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterBothNulls)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<TypeParam, int32_t> source({2, 4, 6, 8}, {1, 1, 0, 0});
  fixed_width_column_wrapper<TypeParam, int32_t> target({10, 20, 30, 40, 50, 60, 70, 80},
                                                        {0, 0, 0, 0, 1, 1, 1, 1});
  fixed_width_column_wrapper<int32_t> scatter_map({1, 3, -3, -1});
  fixed_width_column_wrapper<TypeParam, int32_t> expected({10, 2, 30, 4, 50, 6, 70, 8},
                                                          {0, 1, 0, 1, 1, 0, 1, 0});

  auto const source_table   = cudf::table_view({source, source});
  auto const target_table   = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::scatter(source_table, scatter_map, target_table, true);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), expected_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterSourceNulls)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<TypeParam, int32_t> source({2, 4, 6, 8}, {1, 1, 0, 0});
  fixed_width_column_wrapper<TypeParam, int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({1, 3, -3, -1});
  fixed_width_column_wrapper<TypeParam, int32_t> expected({10, 2, 30, 4, 50, 6, 70, 8},
                                                          {1, 1, 1, 1, 1, 0, 1, 0});

  auto const source_table   = cudf::table_view({source, source});
  auto const target_table   = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::scatter(source_table, scatter_map, target_table, true);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), expected_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterTargetNulls)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<TypeParam, int32_t> source({2, 4, 6, 8});
  fixed_width_column_wrapper<TypeParam, int32_t> target({10, 20, 30, 40, 50, 60, 70, 80},
                                                        {0, 0, 0, 0, 1, 1, 1, 1});
  fixed_width_column_wrapper<int32_t> scatter_map({1, 3, -3, -1});
  fixed_width_column_wrapper<TypeParam, int32_t> expected({10, 2, 30, 4, 50, 6, 70, 8},
                                                          {0, 1, 0, 1, 1, 1, 1, 1});

  auto const source_table   = cudf::table_view({source, source});
  auto const target_table   = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::scatter(source_table, scatter_map, target_table, true);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), expected_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterScalarNoNulls)
{
  using cudf::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;
  using Type = cudf::device_storage_type_t<TypeParam>;

  auto const source = scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<Type>(100), true);
  std::reference_wrapper<const cudf::scalar> slr_ref{source};
  std::vector<std::reference_wrapper<const cudf::scalar>> source_vector{slr_ref};

  fixed_width_column_wrapper<TypeParam, int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});
  fixed_width_column_wrapper<TypeParam, int32_t> expected({10, 100, 30, 100, 50, 100, 70, 100});

  auto const target_table   = cudf::table_view({target});
  auto const expected_table = cudf::table_view({expected});

  auto const result = cudf::scatter(source_vector, scatter_map, target_table, true);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), expected_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterScalarTargetNulls)
{
  using cudf::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;
  using Type = cudf::device_storage_type_t<TypeParam>;

  auto const source = scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<Type>(100), true);
  std::reference_wrapper<const cudf::scalar> slr_ref{source};
  std::vector<std::reference_wrapper<const cudf::scalar>> source_vector{slr_ref};

  fixed_width_column_wrapper<TypeParam, int32_t> target({10, 20, 30, 40, 50, 60, 70, 80},
                                                        {0, 0, 0, 0, 1, 1, 1, 1});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});
  fixed_width_column_wrapper<TypeParam, int32_t> expected({10, 100, 30, 100, 50, 100, 70, 100},
                                                          {0, 1, 0, 1, 1, 1, 1, 1});

  auto const target_table   = cudf::table_view({target});
  auto const expected_table = cudf::table_view({expected});

  auto const result = cudf::scatter(source_vector, scatter_map, target_table, true);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), expected_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterScalarSourceNulls)
{
  using cudf::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;
  using Type = cudf::device_storage_type_t<TypeParam>;

  auto const source =
    scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<Type>(100), false);
  std::reference_wrapper<const cudf::scalar> slr_ref{source};
  std::vector<std::reference_wrapper<const cudf::scalar>> source_vector{slr_ref};

  fixed_width_column_wrapper<TypeParam, int32_t> target({10, 20, 30, 40, 50, 60, 70, 80});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});
  fixed_width_column_wrapper<TypeParam, int32_t> expected({10, 100, 30, 100, 50, 100, 70, 100},
                                                          {1, 0, 1, 0, 1, 0, 1, 0});

  auto const target_table   = cudf::table_view({target});
  auto const expected_table = cudf::table_view({expected});

  auto const result = cudf::scatter(source_vector, scatter_map, target_table, true);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), expected_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterScalarBothNulls)
{
  using cudf::scalar_type_t;
  using cudf::test::fixed_width_column_wrapper;
  using Type = cudf::device_storage_type_t<TypeParam>;

  auto const source =
    scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<Type>(100), false);
  std::reference_wrapper<const cudf::scalar> slr_ref{source};
  std::vector<std::reference_wrapper<const cudf::scalar>> source_vector{slr_ref};

  fixed_width_column_wrapper<TypeParam, int32_t> target({10, 20, 30, 40, 50, 60, 70, 80},
                                                        {0, 0, 0, 0, 1, 1, 1, 1});
  fixed_width_column_wrapper<int32_t> scatter_map({-3, 3, 1, -1});
  fixed_width_column_wrapper<TypeParam, int32_t> expected({10, 100, 30, 100, 50, 100, 70, 100},
                                                          {0, 0, 0, 0, 1, 0, 1, 0});

  auto const target_table   = cudf::table_view({target});
  auto const expected_table = cudf::table_view({expected});

  auto const result = cudf::scatter(source_vector, scatter_map, target_table, true);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), expected_table);
}

TYPED_TEST(ScatterDataTypeTests, ScatterSourceNullsLarge)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::make_counting_transform_iterator;

  constexpr cudf::size_type N{513};

  fixed_width_column_wrapper<TypeParam, int32_t> source({0, 0, 0, 0}, {0, 0, 0, 0});
  fixed_width_column_wrapper<int32_t> scatter_map({0, 1, 2, 3});
  auto target_data = make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(target_data)::value_type>
    target(target_data, target_data + N);

  auto expect_data  = make_counting_transform_iterator(0, [](auto i) { return i; });
  auto expect_valid = make_counting_transform_iterator(0, [](auto i) { return i > 3; });
  fixed_width_column_wrapper<TypeParam, typename decltype(expect_data)::value_type> expected(
    expect_data, expect_data + N, expect_valid);

  auto const source_table   = cudf::table_view({source, source});
  auto const target_table   = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::scatter(source_table, scatter_map, target_table, true);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), expected_table);
}

class ScatterStringsTests : public cudf::test::BaseFixture {
};

TEST_F(ScatterStringsTests, ScatterNoNulls)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::strings_column_wrapper;

  std::vector<const char*> h_source{"dog", "the", "jumps", "brown", "the"};
  strings_column_wrapper source(h_source.begin(), h_source.end());

  std::vector<const char*> h_target{
    "a", "quick", "fire", "fox", "browses", "over", "a", "lazy", "web"};
  strings_column_wrapper target(h_target.begin(), h_target.end());

  fixed_width_column_wrapper<int32_t> scatter_map({-1, -3, -5, 2, 0});

  std::vector<const char*> h_expected{
    "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"};
  strings_column_wrapper expected(h_expected.begin(), h_expected.end());

  auto const source_table   = cudf::table_view({source, source});
  auto const target_table   = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::scatter(source_table, scatter_map, target_table, true);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), expected_table);
}

TEST_F(ScatterStringsTests, ScatterScalarNoNulls)
{
  using cudf::string_scalar;
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::strings_column_wrapper;

  auto const source = string_scalar("buffalo");
  std::reference_wrapper<const cudf::scalar> slr_ref{source};
  std::vector<std::reference_wrapper<const cudf::scalar>> source_vector{slr_ref};

  std::vector<const char*> h_target{
    "Buffalo", "bison", "Buffalo", "bison", "bully", "bully", "Buffalo", "bison"};
  strings_column_wrapper target(h_target.begin(), h_target.end());

  fixed_width_column_wrapper<int32_t> scatter_map({1, 3, -4, -3, -1});

  std::vector<const char*> h_expected{
    "Buffalo", "buffalo", "Buffalo", "buffalo", "buffalo", "buffalo", "Buffalo", "buffalo"};
  strings_column_wrapper expected(h_expected.begin(), h_expected.end());

  auto const target_table   = cudf::table_view({target});
  auto const expected_table = cudf::table_view({expected});

  auto const result = cudf::scatter(source_vector, scatter_map, target_table, true);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), expected_table);
}

template <typename T>
class BooleanMaskScatter : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(BooleanMaskScatter, cudf::test::FixedWidthTypes);

TYPED_TEST(BooleanMaskScatter, WithNoNullElementsInTarget)
{
  using T = TypeParam;
  cudf::test::fixed_width_column_wrapper<T, int32_t> source({1, 5, 6, 8, 9});
  cudf::test::fixed_width_column_wrapper<T, int32_t> target({2, 2, 3, 4, 11, 12, 7, 7, 10, 10});
  cudf::test::fixed_width_column_wrapper<bool> mask(
    {true, false, false, false, true, true, false, true, true, false});

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto source_table   = cudf::table_view({source});
  auto target_table   = cudf::table_view({target});
  auto expected_table = cudf::table_view({expected});

  auto got = cudf::boolean_mask_scatter(source_table, target_table, mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table, got->view());
}

TYPED_TEST(BooleanMaskScatter, WithNull)
{
  using T = TypeParam;
  cudf::test::fixed_width_column_wrapper<T, int32_t> source_col1({1, 5, 6, 8, 9}, {1, 0, 1, 0, 1});
  cudf::test::strings_column_wrapper source_col2({"This", "is", "cudf", "test", "column"},
                                                 {1, 0, 0, 1, 0});
  cudf::test::fixed_width_column_wrapper<T, int32_t> target_col1({2, 2, 3, 4, 11, 12, 7, 7, 10, 10},
                                                                 {1, 1, 0, 1, 1, 1, 1, 1, 1, 0});
  cudf::test::strings_column_wrapper target_col2(
    {"a", "bc", "cd", "ef", "gh", "ij", "jk", "lm", "no", "pq"}, {1, 1, 0, 1, 1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> mask(
    {true, false, false, false, true, true, false, true, true, false});

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected_col1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                                                   {1, 1, 0, 1, 0, 1, 1, 0, 1, 0});
  cudf::test::strings_column_wrapper expected_col2(
    {"This", "bc", "cd", "ef", "is", "cudf", "jk", "test", "column", "pq"},
    {1, 1, 0, 1, 0, 0, 1, 1, 0, 0});
  auto source_table   = cudf::table_view({source_col1, source_col2});
  auto target_table   = cudf::table_view({target_col1, target_col2});
  auto expected_table = cudf::table_view({expected_col1, expected_col2});

  auto got = cudf::boolean_mask_scatter(source_table, target_table, mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table, got->view());
}

class BooleanMaskScatterString : public cudf::test::BaseFixture {
};

TEST_F(BooleanMaskScatterString, NoNUll)
{
  cudf::test::strings_column_wrapper source({"This", "cudf"});
  cudf::test::strings_column_wrapper target({"is", "is", "a", "udf", "api"});
  cudf::test::fixed_width_column_wrapper<bool> mask({true, false, false, true, false});

  cudf::test::strings_column_wrapper expected({"This", "is", "a", "cudf", "api"});
  auto source_table   = cudf::table_view({source});
  auto target_table   = cudf::table_view({target});
  auto expected_table = cudf::table_view({expected});

  auto got = cudf::boolean_mask_scatter(source_table, target_table, mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table, got->view());
}

TEST_F(BooleanMaskScatterString, WithNUll)
{
  cudf::test::strings_column_wrapper source({"This", "cudf"}, {0, 1});
  cudf::test::strings_column_wrapper target({"is", "is", "a", "udf", "api"}, {1, 0, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<bool> mask({true, false, false, true, false});

  cudf::test::strings_column_wrapper expected({"This", "is", "a", "cudf", "api"}, {0, 0, 0, 1, 1});
  auto source_table   = cudf::table_view({source});
  auto target_table   = cudf::table_view({target});
  auto expected_table = cudf::table_view({expected});

  auto got = cudf::boolean_mask_scatter(source_table, target_table, mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table, got->view());
}

class BooleanMaskScatterFails : public cudf::test::BaseFixture {
};

TEST_F(BooleanMaskScatterFails, SourceAndTargetTypeMismatch)
{
  cudf::test::fixed_width_column_wrapper<int32_t> source({1, 5, 6, 8, 9});
  cudf::test::fixed_width_column_wrapper<int64_t> target({2, 2, 3, 4, 11, 12, 7, 7, 10, 10});
  cudf::test::fixed_width_column_wrapper<bool> mask(
    {true, false, false, false, true, true, false, true, true, false});
  auto source_table = cudf::table_view({source});
  auto target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::boolean_mask_scatter(source_table, target_table, mask), cudf::logic_error);
}

TEST_F(BooleanMaskScatterFails, BooleanMaskTypeMismatch)
{
  cudf::test::fixed_width_column_wrapper<int32_t> source({1, 5, 6, 8, 9});
  cudf::test::fixed_width_column_wrapper<int32_t> target({2, 2, 3, 4, 11, 12, 7, 7, 10, 10});
  cudf::test::fixed_width_column_wrapper<int8_t> mask(
    {true, false, false, false, true, true, false, true, true, false});
  auto source_table = cudf::table_view({source});
  auto target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::boolean_mask_scatter(source_table, target_table, mask), cudf::logic_error);
}

TEST_F(BooleanMaskScatterFails, BooleanMaskTargetSizeMismatch)
{
  cudf::test::fixed_width_column_wrapper<int32_t> source({1, 5, 6, 8, 9});
  cudf::test::fixed_width_column_wrapper<int32_t> target({2, 2, 3, 4, 11, 12, 7, 7, 10, 10});
  cudf::test::fixed_width_column_wrapper<bool> mask(
    {true, false, false, false, true, true, false, true, true});
  auto source_table = cudf::table_view({source});
  auto target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::boolean_mask_scatter(source_table, target_table, mask), cudf::logic_error);
}

TEST_F(BooleanMaskScatterFails, NumberOfColumnMismatch)
{
  cudf::test::fixed_width_column_wrapper<int32_t> source({1, 5, 6, 8, 9});
  cudf::test::fixed_width_column_wrapper<int32_t> target({2, 2, 3, 4, 11, 12, 7, 7, 10, 10});
  cudf::test::fixed_width_column_wrapper<bool> mask(
    {true, false, false, false, true, true, false, true, true});
  auto source_table = cudf::table_view({source, source});
  auto target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::boolean_mask_scatter(source_table, target_table, mask), cudf::logic_error);
}

TEST_F(BooleanMaskScatterFails, MoreTruesInMaskThanSourceSize)
{
  cudf::test::fixed_width_column_wrapper<int32_t> source({1, 5, 6, 8, 9});
  cudf::test::fixed_width_column_wrapper<int32_t> target({2, 2, 3, 4, 11, 12, 7, 7, 10, 10});
  cudf::test::fixed_width_column_wrapper<bool> mask(
    {true, false, true, false, true, true, false, true, true});
  auto source_table = cudf::table_view({source, source});
  auto target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::boolean_mask_scatter(source_table, target_table, mask), cudf::logic_error);
}

template <typename T>
struct BooleanMaskScalarScatter : public cudf::test::BaseFixture {
  std::unique_ptr<cudf::scalar> form_scalar(T value, bool validity = true)
  {
    using ScalarType = cudf::scalar_type_t<T>;
    std::unique_ptr<cudf::scalar> scalar{nullptr};

    if (cudf::is_numeric<T>()) {
      scalar = cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
    } else if (cudf::is_timestamp<T>()) {
      scalar = cudf::make_timestamp_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
    } else if (cudf::is_duration<T>()) {
      scalar = cudf::make_duration_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
    }

    static_cast<ScalarType*>(scalar.get())->set_value(value);
    static_cast<ScalarType*>(scalar.get())->set_valid(validity);

    return scalar;
  }
};

TYPED_TEST_CASE(BooleanMaskScalarScatter, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(BooleanMaskScalarScatter, WithNoNullElementsInTarget)
{
  using T       = TypeParam;
  T source      = cudf::test::make_type_param_scalar<T>(11);
  bool validity = true;
  auto scalar   = this->form_scalar(source, validity);
  std::vector<std::reference_wrapper<const cudf::scalar>> scalar_vect;
  scalar_vect.push_back(*scalar);
  cudf::test::fixed_width_column_wrapper<T, int32_t> target({2, 2, 3, 4, 11, 12, 7, 7, 10, 10});
  cudf::test::fixed_width_column_wrapper<bool> mask(
    {true, false, false, false, true, true, false, true, true, false});

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected({11, 2, 3, 4, 11, 11, 7, 11, 11, 10});
  auto target_table   = cudf::table_view({target});
  auto expected_table = cudf::table_view({expected});

  auto got = cudf::boolean_mask_scatter(scalar_vect, target_table, mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table, got->view());
}

TYPED_TEST(BooleanMaskScalarScatter, WithNull)
{
  using T       = TypeParam;
  T source      = cudf::test::make_type_param_scalar<T>(11);
  bool validity = false;
  auto scalar_1 = this->form_scalar(source, validity);
  auto scalar_2 = cudf::make_string_scalar("cudf");
  scalar_2->set_valid(true);
  std::vector<std::reference_wrapper<const cudf::scalar>> scalar_vect;
  scalar_vect.push_back(*scalar_1);
  scalar_vect.push_back(*scalar_2);
  cudf::test::fixed_width_column_wrapper<T, int32_t> target_col1({2, 2, 3, 4, 11, 12, 7, 7, 10, 10},
                                                                 {1, 1, 0, 1, 1, 1, 1, 1, 1, 0});
  cudf::test::strings_column_wrapper target_col2(
    {"a", "bc", "cd", "ef", "gh", "ij", "jk", "lm", "no", "pq"}, {1, 1, 0, 1, 1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> mask(
    {true, false, false, false, true, true, false, true, true, false});

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected_col1(
    {11, 2, 3, 4, 11, 11, 7, 11, 11, 10}, {0, 1, 0, 1, 0, 0, 1, 0, 0, 0});
  cudf::test::strings_column_wrapper expected_col2(
    {"cudf", "bc", "cd", "ef", "cudf", "cudf", "jk", "cudf", "cudf", "pq"},
    {1, 1, 0, 1, 1, 1, 1, 1, 1, 0});
  auto target_table   = cudf::table_view({target_col1, target_col2});
  auto expected_table = cudf::table_view({expected_col1, expected_col2});

  auto got = cudf::boolean_mask_scatter(scalar_vect, target_table, mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table, got->view());
}

class BooleanMaskScatterScalarString : public cudf::test::BaseFixture {
};

TEST_F(BooleanMaskScatterScalarString, NoNUll)
{
  auto scalar = cudf::make_string_scalar("cudf");
  scalar->set_valid(true);
  std::vector<std::reference_wrapper<const cudf::scalar>> scalar_vect;
  scalar_vect.push_back(*scalar);

  cudf::test::strings_column_wrapper target({"is", "is", "a", "udf", "api"});
  cudf::test::fixed_width_column_wrapper<bool> mask({true, false, false, true, false});

  cudf::test::strings_column_wrapper expected({"cudf", "is", "a", "cudf", "api"});
  auto target_table   = cudf::table_view({target});
  auto expected_table = cudf::table_view({expected});

  auto got = cudf::boolean_mask_scatter(scalar_vect, target_table, mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table, got->view());
}

TEST_F(BooleanMaskScatterScalarString, WithNUll)
{
  auto scalar = cudf::make_string_scalar("cudf");
  scalar->set_valid(true);
  std::vector<std::reference_wrapper<const cudf::scalar>> scalar_vect;
  scalar_vect.push_back(*scalar);
  cudf::test::strings_column_wrapper target({"is", "is", "a", "udf", "api"}, {1, 0, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<bool> mask({true, false, true, true, false});

  cudf::test::strings_column_wrapper expected({"cudf", "is", "cudf", "cudf", "api"},
                                              {1, 0, 1, 1, 1});
  auto target_table   = cudf::table_view({target});
  auto expected_table = cudf::table_view({expected});
  auto got            = cudf::boolean_mask_scatter(scalar_vect, target_table, mask);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table, got->view());
}

class BooleanMaskScatterScalarFails : public cudf::test::BaseFixture {
};

TEST_F(BooleanMaskScatterScalarFails, SourceAndTargetTypeMismatch)
{
  auto scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<int32_t>()}));
  std::vector<std::reference_wrapper<const cudf::scalar>> scalar_vect;
  scalar_vect.push_back(*scalar);
  cudf::test::fixed_width_column_wrapper<int64_t> target({2, 2, 3, 4, 11, 12, 7, 7, 10, 10});
  cudf::test::fixed_width_column_wrapper<bool> mask(
    {true, false, false, false, true, true, false, true, true, false});
  auto target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::boolean_mask_scatter(scalar_vect, target_table, mask), cudf::logic_error);
}

TEST_F(BooleanMaskScatterScalarFails, BooleanMaskTypeMismatch)
{
  auto scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<int32_t>()}));
  std::vector<std::reference_wrapper<const cudf::scalar>> scalar_vect;
  scalar_vect.push_back(*scalar);
  cudf::test::fixed_width_column_wrapper<int32_t> target({2, 2, 3, 4, 11, 12, 7, 7, 10, 10});
  cudf::test::fixed_width_column_wrapper<int8_t> mask(
    {true, false, false, false, true, true, false, true, true, false});
  auto target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::boolean_mask_scatter(scalar_vect, target_table, mask), cudf::logic_error);
}

TEST_F(BooleanMaskScatterScalarFails, BooleanMaskTargetSizeMismatch)
{
  auto scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<int32_t>()}));
  std::vector<std::reference_wrapper<const cudf::scalar>> scalar_vect;
  scalar_vect.push_back(*scalar);
  cudf::test::fixed_width_column_wrapper<int32_t> target({2, 2, 3, 4, 11, 12, 7, 7, 10, 10});
  cudf::test::fixed_width_column_wrapper<bool> mask(
    {true, false, false, false, true, true, false, true, true});
  auto target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::boolean_mask_scatter(scalar_vect, target_table, mask), cudf::logic_error);
}

TEST_F(BooleanMaskScatterScalarFails, NumberOfColumnAndScalarMismatch)
{
  auto scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<int32_t>()}));
  std::vector<std::reference_wrapper<const cudf::scalar>> scalar_vect;
  scalar_vect.push_back(*scalar);
  scalar_vect.push_back(*scalar);
  cudf::test::fixed_width_column_wrapper<int32_t> target({2, 2, 3, 4, 11, 12, 7, 7, 10, 10});
  cudf::test::fixed_width_column_wrapper<bool> mask(
    {true, false, false, false, true, true, false, true, true});
  auto target_table = cudf::table_view({target});

  EXPECT_THROW(cudf::boolean_mask_scatter(scalar_vect, target_table, mask), cudf::logic_error);
}

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {
};

template <typename T>
using wrapper = cudf::test::fixed_width_column_wrapper<T>;
TYPED_TEST_CASE(FixedPointTestBothReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestBothReps, FixedPointScatter)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto const ONE   = decimalXX{1, scale_type{0}};
  auto const TWO   = decimalXX{2, scale_type{0}};
  auto const THREE = decimalXX{3, scale_type{0}};
  auto const FOUR  = decimalXX{4, scale_type{0}};
  auto const FIVE  = decimalXX{5, scale_type{0}};

  auto const source      = wrapper<decimalXX>({ONE, TWO, THREE, FOUR, FIVE});
  auto const target      = wrapper<decimalXX>({ONE, TWO, THREE, FOUR, FIVE, FOUR, THREE, TWO, ONE});
  auto const scatter_map = wrapper<int32_t>({1, 2, -1, -3, -4});
  auto const expected    = wrapper<decimalXX>({ONE, ONE, TWO, FOUR, FIVE, FIVE, FOUR, TWO, THREE});

  auto const source_table   = cudf::table_view({source, source});
  auto const target_table   = cudf::table_view({target, target});
  auto const expected_table = cudf::table_view({expected, expected});

  auto const result = cudf::scatter(source_table, scatter_map, target_table, true);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table, result->view());
}
