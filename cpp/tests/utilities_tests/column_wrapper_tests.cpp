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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>

template <typename T>
struct FixedWidthColumnWrapperTest : public cudf::test::BaseFixture,
                                     cudf::test::UniformRandomGenerator<cudf::size_type> {
  FixedWidthColumnWrapperTest() : cudf::test::UniformRandomGenerator<cudf::size_type>{1000, 5000} {}

  auto size() { return this->generate(); }

  auto data_type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

TYPED_TEST_SUITE(FixedWidthColumnWrapperTest, cudf::test::FixedWidthTypes);

TYPED_TEST(FixedWidthColumnWrapperTest, EmptyIterator)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.head(), nullptr);
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}
TYPED_TEST(FixedWidthColumnWrapperTest, EmptyList)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> col{};
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.head(), nullptr);
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NonNullableIteratorConstructor)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });

  auto size = this->size();

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + size);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), size);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NonNullableListConstructor)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4, 5});

  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableIteratorConstructorAllValid)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });

  auto all_valid = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  auto size = this->size();

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + size, all_valid);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), size);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableListConstructorAllValid)
{
  auto all_valid = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4, 5}, all_valid);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableIteratorConstructorAllNull)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });

  auto all_null = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return false; });

  auto size = this->size();

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + size, all_null);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), size);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), size);
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableListConstructorAllNull)
{
  auto all_null = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return false; });

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4, 5}, all_null);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullablePairListConstructorAllNull)
{
  using p = std::pair<int32_t, bool>;
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col(
    {p{1, false}, p{2, false}, p{3, false}, p{4, false}, p{5, false}});
  cudf::column_view view = col;

  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullablePairListConstructorAllNullMatch)
{
  auto odd_valid =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 != 0; });

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> match_col({1, 2, 3, 4, 5}, odd_valid);
  cudf::column_view match_view = match_col;

  using p = std::pair<int32_t, bool>;
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({p{1, odd_valid[0]},
                                                                  p{2, odd_valid[1]},
                                                                  p{3, odd_valid[2]},
                                                                  p{4, odd_valid[3]},
                                                                  p{5, odd_valid[4]}});
  cudf::column_view view = col;

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view, match_view);
}

TYPED_TEST(FixedWidthColumnWrapperTest, ReleaseWrapperAllValid)
{
  auto all_valid = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4, 5}, all_valid);
  auto colPtr            = col.release();
  cudf::column_view view = *colPtr;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, ReleaseWrapperAllNull)
{
  auto all_null = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return false; });

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4, 5}, all_null);
  auto colPtr            = col.release();
  cudf::column_view view = *colPtr;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
  EXPECT_EQ(view.offset(), 0);
}

template <typename T>
struct StringsColumnWrapperTest : public cudf::test::BaseFixture,
                                  cudf::test::UniformRandomGenerator<cudf::size_type> {
  auto data_type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

TYPED_TEST_SUITE(StringsColumnWrapperTest, cudf::test::StringTypes);

TYPED_TEST(StringsColumnWrapperTest, EmptyList)
{
  cudf::test::strings_column_wrapper col;
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.head(), nullptr);
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(StringsColumnWrapperTest, NullablePairListConstructorAllNull)
{
  using p = std::pair<std::string, bool>;
  cudf::test::strings_column_wrapper col(
    {p{"a", false}, p{"string", false}, p{"test", false}, p{"for", false}, p{"nulls", false}});
  cudf::strings_column_view view = cudf::column_view(col);

  constexpr auto count = 5;
  EXPECT_EQ(view.size(), count);
  EXPECT_EQ(view.offsets().size(), count + 1);
  // all null entries results in no data allocated to chars
  EXPECT_EQ(nullptr, view.parent().head());
  EXPECT_NE(nullptr, view.offsets().head());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
}

TYPED_TEST(StringsColumnWrapperTest, NullablePairListConstructorAllNullMatch)
{
  auto odd_valid =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 != 0; });

  cudf::test::strings_column_wrapper match_col({"a", "string", "", "test", "for", "nulls"},
                                               odd_valid);
  cudf::column_view match_view = match_col;

  using p = std::pair<std::string, bool>;
  cudf::test::strings_column_wrapper col({p{"a", odd_valid[0]},
                                          p{"string", odd_valid[1]},
                                          p{"", odd_valid[2]},
                                          p{"test", odd_valid[3]},
                                          p{"for", odd_valid[4]},
                                          p{"nulls", odd_valid[5]}});
  cudf::column_view view = col;

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view, match_view);
}
