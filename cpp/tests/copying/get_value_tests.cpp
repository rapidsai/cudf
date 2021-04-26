/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

namespace cudf {
namespace test {

template <typename T>
struct FixedWidthGetValueTest : public BaseFixture {
};

TYPED_TEST_CASE(FixedWidthGetValueTest, FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(FixedWidthGetValueTest, BasicGet)
{
  fixed_width_column_wrapper<TypeParam, int32_t> col({9, 8, 7, 6});
  auto s = get_element(col, 0);

  using ScalarType = scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(cudf::test::make_type_param_scalar<TypeParam>(9), typed_s->value());
}

TYPED_TEST(FixedWidthGetValueTest, GetFromNullable)
{
  fixed_width_column_wrapper<TypeParam, int32_t> col({9, 8, 7, 6}, {0, 1, 0, 1});
  auto s = get_element(col, 1);

  using ScalarType = scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(cudf::test::make_type_param_scalar<TypeParam>(8), typed_s->value());
}

TYPED_TEST(FixedWidthGetValueTest, GetNull)
{
  fixed_width_column_wrapper<TypeParam, int32_t> col({9, 8, 7, 6}, {0, 1, 0, 1});
  auto s = get_element(col, 2);

  EXPECT_FALSE(s->is_valid());
}

TYPED_TEST(FixedWidthGetValueTest, IndexOutOfBounds)
{
  fixed_width_column_wrapper<TypeParam, int32_t> col({9, 8, 7, 6}, {0, 1, 0, 1});

  CUDF_EXPECT_THROW_MESSAGE(get_element(col, -1);, "Index out of bounds");
  CUDF_EXPECT_THROW_MESSAGE(get_element(col, 4);, "Index out of bounds");
}

struct StringGetValueTest : public BaseFixture {
};

TEST_F(StringGetValueTest, BasicGet)
{
  strings_column_wrapper col{"this", "is", "a", "test"};
  auto s = get_element(col, 3);

  auto typed_s = static_cast<string_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ("test", typed_s->to_string());
}

TEST_F(StringGetValueTest, GetEmpty)
{
  strings_column_wrapper col{"this", "is", "", "test"};
  auto s = get_element(col, 2);

  auto typed_s = static_cast<string_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ("", typed_s->to_string());
}

TEST_F(StringGetValueTest, GetFromNullable)
{
  strings_column_wrapper col({"this", "is", "a", "test"}, {0, 1, 0, 1});
  auto s = get_element(col, 1);

  auto typed_s = static_cast<string_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ("is", typed_s->to_string());
}

TEST_F(StringGetValueTest, GetNull)
{
  strings_column_wrapper col({"this", "is", "a", "test"}, {0, 1, 0, 1});
  auto s = get_element(col, 2);

  EXPECT_FALSE(s->is_valid());
}

template <typename T>
struct DictionaryGetValueTest : public BaseFixture {
};

TYPED_TEST_CASE(DictionaryGetValueTest, FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(DictionaryGetValueTest, BasicGet)
{
  fixed_width_column_wrapper<TypeParam, int32_t> keys({6, 7, 8, 9});
  fixed_width_column_wrapper<uint32_t> indices{0, 0, 1, 2, 1, 3, 3, 2};
  auto col = make_dictionary_column(keys, indices);

  auto s = get_element(*col, 2);

  using ScalarType = scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(cudf::test::make_type_param_scalar<TypeParam>(7), typed_s->value());
}

TYPED_TEST(DictionaryGetValueTest, GetFromNullable)
{
  fixed_width_column_wrapper<TypeParam, int32_t> keys({6, 7, 8, 9});
  fixed_width_column_wrapper<uint32_t> indices({0, 0, 1, 2, 1, 3, 3, 2}, {0, 1, 0, 1, 1, 1, 0, 0});
  auto col = make_dictionary_column(keys, indices);

  auto s = get_element(*col, 3);

  using ScalarType = scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(cudf::test::make_type_param_scalar<TypeParam>(8), typed_s->value());
}

TYPED_TEST(DictionaryGetValueTest, GetNull)
{
  fixed_width_column_wrapper<TypeParam, int32_t> keys({6, 7, 8, 9});
  fixed_width_column_wrapper<uint32_t> indices({0, 0, 1, 2, 1, 3, 3, 2}, {0, 1, 0, 1, 1, 1, 0, 0});
  auto col = make_dictionary_column(keys, indices);

  auto s = get_element(*col, 2);

  EXPECT_FALSE(s->is_valid());
}

/*
 * Lists test grid:
 * Dim1 nestedness:          {Nested, Non-nested}
 * Dim2 validity, emptiness: {Null element, Non-null non-empty list, Non-null empty list}
 * Dim3 leaf data type:      {Fixed-width, string}
 */

template <typename T>
struct ListGetFixedWidthValueTest : public BaseFixture {
};

TYPED_TEST_CASE(ListGetFixedWidthValueTest, FixedWidthTypes);

TYPED_TEST(ListGetFixedWidthValueTest, NonNestedGetNonNullNonEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  LCW col{LCW{1, 2, 34}, LCW{}, LCW{1}, LCW{}};
  fixed_width_column_wrapper<TypeParam> expected_data{1, 2, 34};
  size_type index = 0;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TYPED_TEST(ListGetFixedWidthValueTest, NonNestedGetNonNullEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  LCW col{LCW{1, 2, 34}, LCW{}, LCW{1}, LCW{}};
  fixed_width_column_wrapper<TypeParam> expected_data{};
  size_type index = 1;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TYPED_TEST(ListGetFixedWidthValueTest, NonNestedGetNull)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  std::vector<valid_type> valid{0, 1, 0, 1};
  LCW col({LCW{1, 2, 34}, LCW{}, LCW{1}, LCW{}}, valid.begin());
  size_type index = 2;

  auto s = get_element(col, index);

  EXPECT_FALSE(s->is_valid());
}

TYPED_TEST(ListGetFixedWidthValueTest, NestedGetNonNullNonEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // clang-format off
  LCW col{
    LCW{LCW{1, 2}, LCW{34}},
    LCW{},
    LCW{LCW{1}},
    LCW{LCW{42}, LCW{10}}
  };
  // clang-format on
  LCW expected_data{LCW{42}, LCW{10}};

  size_type index = 3;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TYPED_TEST(ListGetFixedWidthValueTest, NestedGetNonNullNonEmptyPreserveNull)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  std::vector<valid_type> valid{0, 1};
  // clang-format off
  LCW col{
    LCW{LCW{1, 2}, LCW{34}},
    LCW{},
    LCW{LCW{1}},
    LCW({LCW{42}, LCW{10}}, valid.begin())
  };
  // clang-format on
  LCW expected_data({LCW{42}, LCW{10}}, valid.begin());
  size_type index = 3;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TYPED_TEST(ListGetFixedWidthValueTest, NestedGetNonNullEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // clang-format off
  LCW col{
    LCW{LCW{1, 2}, LCW{34}},
    LCW{},
    LCW{LCW{1}},
    LCW{LCW{42}, LCW{10}}
  };
  // clang-format on
  LCW expected_data{};
  size_type index = 1;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TYPED_TEST(ListGetFixedWidthValueTest, NestedGetNull)
{
  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  std::vector<valid_type> valid{1, 0, 1, 0};
  // clang-format off
  LCW col(
    {
      LCW{LCW{1, 2}, LCW{34}},
      LCW{},
      LCW{LCW{1}},
      LCW{LCW{42}, LCW{10}}
    }, valid.begin());
  // clang-format on
  size_type index = 1;

  auto s = get_element(col, index);

  EXPECT_FALSE(s->is_valid());
}

struct ListGetStringValueTest : public BaseFixture {
};

TEST_F(ListGetStringValueTest, NonNestedGetNonNullNonEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<string_view>;

  LCW col{LCW{"aaa", "Héllo"}, LCW{}, LCW{""}, LCW{"42"}};
  strings_column_wrapper expected_data{"aaa", "Héllo"};
  size_type index = 0;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TEST_F(ListGetStringValueTest, NonNestedGetNonNullEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<string_view>;

  LCW col{LCW{"aaa", "Héllo"}, LCW{}, LCW{""}, LCW{"42"}};
  strings_column_wrapper expected_data{};
  size_type index = 1;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TEST_F(ListGetStringValueTest, NonNestedGetNull)
{
  using LCW = cudf::test::lists_column_wrapper<string_view>;

  std::vector<valid_type> valid{1, 0, 0, 1};
  LCW col({LCW{"aaa", "Héllo"}, LCW{}, LCW{""}, LCW{"42"}}, valid.begin());
  size_type index = 2;

  auto s = get_element(col, index);

  EXPECT_FALSE(s->is_valid());
}

TEST_F(ListGetStringValueTest, NestedGetNonNullNonEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<string_view>;

  // clang-format off
  LCW col{
    LCW{LCW{"aaa", "Héllo"}},
    {LCW{}},
    LCW{LCW{""}},
    LCW{LCW{"42"}, LCW{"21"}}
  };
  // clang-format on
  LCW expected_data{LCW{""}};
  size_type index = 2;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TEST_F(ListGetStringValueTest, NestedGetNonNullNonEmptyPreserveNull)
{
  using LCW = cudf::test::lists_column_wrapper<string_view>;

  std::vector<valid_type> valid{0, 1};
  // clang-format off
  LCW col{
    LCW{LCW{"aaa", "Héllo"}},
    {LCW{}},
    LCW({LCW{""}, LCW{"cc"}}, valid.begin()),
    LCW{LCW{"42"}, LCW{"21"}}
  };
  // clang-format on
  LCW expected_data({LCW{""}, LCW{"cc"}}, valid.begin());
  size_type index = 2;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TEST_F(ListGetStringValueTest, NestedGetNonNullEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<string_view>;

  // clang-format off
  LCW col{
    LCW{LCW{"aaa", "Héllo"}},
    LCW{LCW{""}},
    LCW{LCW{"42"}, LCW{"21"}},
    {LCW{}}
  };
  // clang-format on
  LCW expected_data{LCW{}};  // a list column with 1 row of an empty string list
  size_type index = 3;

  auto s       = get_element(col, index);
  auto typed_s = static_cast<list_scalar const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_data, typed_s->view());
}

TEST_F(ListGetStringValueTest, NestedGetNull)
{
  using LCW = cudf::test::lists_column_wrapper<string_view>;

  std::vector<valid_type> valid{0, 0, 1, 1};
  // clang-format off
  LCW col(
    {
      LCW{LCW{"aaa", "Héllo"}},
      LCW{LCW{""}},
      LCW{LCW{"42"}, LCW{"21"}}, 
      {LCW{}}
    }, valid.begin());
  // clang-format on
  LCW expected_data{};
  size_type index = 0;

  auto s = get_element(col, index);
  EXPECT_FALSE(s->is_valid());
}

}  // namespace test
}  // namespace cudf
