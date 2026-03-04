/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>

struct DictionaryEncodeTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryEncodeTest, EncodeStringColumn)
{
  cudf::test::strings_column_wrapper input(
    {"eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"});

  auto dictionary = cudf::dictionary::encode(input);
  auto view       = cudf::dictionary_column_view(dictionary->view());

  auto decoded = cudf::dictionary::decode(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), input);
}

template <typename T>
class DictionaryEncodeNumericTest : public DictionaryEncodeTest {};
using NumericTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(DictionaryEncodeNumericTest, NumericTypes);

TYPED_TEST(DictionaryEncodeNumericTest, Encode)
{
  auto input = cudf::test::fixed_width_column_wrapper<TypeParam, int>{4, 7, 0, -11, 7, 0};

  auto dictionary = cudf::dictionary::encode(input);
  auto view       = cudf::dictionary_column_view(dictionary->view());

  auto decoded = cudf::dictionary::decode(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), input);
}

TEST_F(DictionaryEncodeTest, EncodeWithNull)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{
    {444, 0, 333, 111, 222, 222, 222, 444, 000},
    {true, true, true, true, true, false, true, true, true}};

  auto dictionary = cudf::dictionary::encode(input);
  auto view       = cudf::dictionary_column_view(dictionary->view());

  auto decoded = cudf::dictionary::decode(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), input);
}

template <typename T>
class DictionaryEncodeIndicesTest : public DictionaryEncodeTest {};
using IndexTypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_SUITE(DictionaryEncodeIndicesTest, IndexTypes);

TYPED_TEST(DictionaryEncodeIndicesTest, IndexType)
{
  auto input      = cudf::test::strings_column_wrapper({"aaa", "bbb", "bbb", "cccc"});
  auto data_type  = cudf::data_type{cudf::type_to_id<TypeParam>()};
  auto dictionary = cudf::dictionary::encode(input, data_type);
  auto view       = cudf::dictionary_column_view(dictionary->view());
  EXPECT_EQ(view.indices().type(), data_type);
}

TEST_F(DictionaryEncodeTest, Errors)
{
  cudf::test::fixed_width_column_wrapper<int16_t> input{0, 1, 2, 3, -1, -2, -3};

  EXPECT_THROW(cudf::dictionary::encode(input, cudf::data_type{cudf::type_id::UINT16}),
               cudf::data_type_error);

  auto encoded = cudf::dictionary::encode(input);
  EXPECT_THROW(cudf::dictionary::encode(encoded->view()), std::invalid_argument);
}
