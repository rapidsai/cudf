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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_list_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

namespace cudf {
namespace test {

template <typename T>
struct FixedWidthGetValueTest : public BaseFixture {
};

TYPED_TEST_CASE(FixedWidthGetValueTest, FixedWidthTypes);

TYPED_TEST(FixedWidthGetValueTest, BasicGet)
{
  fixed_width_column_wrapper<TypeParam> col{9, 8, 7, 6};
  auto s = get_element(col, 0);

  using ScalarType = scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(TypeParam(9), typed_s->value());
}

TYPED_TEST(FixedWidthGetValueTest, GetFromNullable)
{
  fixed_width_column_wrapper<TypeParam> col({9, 8, 7, 6}, {0, 1, 0, 1});
  auto s = get_element(col, 1);

  using ScalarType = scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(TypeParam(8), typed_s->value());
}

TYPED_TEST(FixedWidthGetValueTest, GetNull)
{
  fixed_width_column_wrapper<TypeParam> col({9, 8, 7, 6}, {0, 1, 0, 1});
  auto s = get_element(col, 2);

  EXPECT_FALSE(s->is_valid());
}

TYPED_TEST(FixedWidthGetValueTest, IndexOutOfBounds)
{
  fixed_width_column_wrapper<TypeParam> col({9, 8, 7, 6}, {0, 1, 0, 1});

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

TYPED_TEST_CASE(DictionaryGetValueTest, FixedWidthTypes);

TYPED_TEST(DictionaryGetValueTest, BasicGet)
{
  fixed_width_column_wrapper<TypeParam> keys{6, 7, 8, 9};
  fixed_width_column_wrapper<int32_t> indices{0, 0, 1, 2, 1, 3, 3, 2};
  auto col = make_dictionary_column(keys, indices);

  auto s = get_element(*col, 2);

  using ScalarType = scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(TypeParam(7), typed_s->value());
}

TYPED_TEST(DictionaryGetValueTest, GetFromNullable)
{
  fixed_width_column_wrapper<TypeParam> keys{6, 7, 8, 9};
  fixed_width_column_wrapper<int32_t> indices({0, 0, 1, 2, 1, 3, 3, 2}, {0, 1, 0, 1, 1, 1, 0, 0});
  auto col = make_dictionary_column(keys, indices);

  auto s = get_element(*col, 3);

  using ScalarType = scalar_type_t<TypeParam>;
  auto typed_s     = static_cast<ScalarType const*>(s.get());

  EXPECT_TRUE(s->is_valid());
  EXPECT_EQ(TypeParam(8), typed_s->value());
}

TYPED_TEST(DictionaryGetValueTest, GetNull)
{
  fixed_width_column_wrapper<TypeParam> keys{6, 7, 8, 9};
  fixed_width_column_wrapper<int32_t> indices({0, 0, 1, 2, 1, 3, 3, 2}, {0, 1, 0, 1, 1, 1, 0, 0});
  auto col = make_dictionary_column(keys, indices);

  auto s = get_element(*col, 2);

  EXPECT_FALSE(s->is_valid());
}

}  // namespace test
}  // namespace cudf
