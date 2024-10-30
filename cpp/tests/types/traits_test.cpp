/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/utilities/traits.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <tuple>

template <typename Tuple, typename F, std::size_t... Indices>
void tuple_for_each_impl(Tuple&& tuple, F&& f, std::index_sequence<Indices...>)
{
  (void)std::initializer_list<int>{
    ((void)(f(std::get<Indices>(std::forward<Tuple>(tuple)))), int{})...};
}

template <typename F, typename... Args>
void tuple_for_each(std::tuple<Args...> const& tuple, F&& f)
{
  tuple_for_each_impl(tuple, std::forward<F>(f), std::index_sequence_for<Args...>{});
}

class TraitsTest : public ::testing::Test {};

template <typename T>
class TypedTraitsTest : public TraitsTest {};

TYPED_TEST_SUITE(TypedTraitsTest, cudf::test::AllTypes);

TEST_F(TraitsTest, NumericDataTypesAreNumeric)
{
  EXPECT_TRUE(
    std::all_of(cudf::test::numeric_type_ids.begin(),
                cudf::test::numeric_type_ids.end(),
                [](cudf::type_id type) { return cudf::is_numeric(cudf::data_type{type}); }));
}

TEST_F(TraitsTest, TimestampDataTypesAreNotNumeric)
{
  EXPECT_TRUE(
    std::none_of(cudf::test::timestamp_type_ids.begin(),
                 cudf::test::timestamp_type_ids.end(),
                 [](cudf::type_id type) { return cudf::is_numeric(cudf::data_type{type}); }));
}

TEST_F(TraitsTest, NumericDataTypesAreNotTimestamps)
{
  EXPECT_TRUE(
    std::none_of(cudf::test::numeric_type_ids.begin(),
                 cudf::test::numeric_type_ids.end(),
                 [](cudf::type_id type) { return cudf::is_timestamp(cudf::data_type{type}); }));
}

TEST_F(TraitsTest, TimestampDataTypesAreTimestamps)
{
  EXPECT_TRUE(
    std::all_of(cudf::test::timestamp_type_ids.begin(),
                cudf::test::timestamp_type_ids.end(),
                [](cudf::type_id type) { return cudf::is_timestamp(cudf::data_type{type}); }));
}

TYPED_TEST(TypedTraitsTest, RelationallyComparable)
{
  // All the test types should be comparable with themselves
  bool comparable = cudf::is_relationally_comparable<TypeParam, TypeParam>();
  EXPECT_TRUE(comparable);
}

TYPED_TEST(TypedTraitsTest, NotRelationallyComparable)
{
  // No type should be comparable with an empty dummy type
  struct foo {};
  bool comparable = cudf::is_relationally_comparable<foo, TypeParam>();
  EXPECT_FALSE(comparable);

  comparable = cudf::is_relationally_comparable<TypeParam, foo>();
  EXPECT_FALSE(comparable);
}

TYPED_TEST(TypedTraitsTest, NotRelationallyComparableWithList)
{
  bool comparable = cudf::is_relationally_comparable<TypeParam, cudf::list_view>();
  EXPECT_FALSE(comparable);

  comparable = cudf::is_relationally_comparable<cudf::list_view, cudf::list_view>();
  EXPECT_FALSE(comparable);
}

TYPED_TEST(TypedTraitsTest, EqualityComparable)
{
  // All the test types should be comparable with themselves
  bool comparable = cudf::is_equality_comparable<TypeParam, TypeParam>();
  EXPECT_TRUE(comparable);
}

TYPED_TEST(TypedTraitsTest, NotEqualityComparable)
{
  // No type should be comparable with an empty dummy type
  struct foo {};
  bool comparable = cudf::is_equality_comparable<foo, TypeParam>();
  EXPECT_FALSE(comparable);

  comparable = cudf::is_equality_comparable<TypeParam, foo>();
  EXPECT_FALSE(comparable);
}

TYPED_TEST(TypedTraitsTest, NotEqualityComparableWithList)
{
  bool comparable = cudf::is_equality_comparable<TypeParam, cudf::list_view>();
  EXPECT_FALSE(comparable);

  cudf::is_equality_comparable<cudf::list_view, cudf::list_view>();
  EXPECT_FALSE(comparable);
}

// TODO: Tests for is_compound, is_fixed_width

CUDF_TEST_PROGRAM_MAIN()
