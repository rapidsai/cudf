/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/convert/convert_cp932.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct StringsCp932Test : public cudf::test::BaseFixture {};

TEST_F(StringsCp932Test, Utf8ToCp932Ascii)
{
  // ASCII characters should pass through unchanged
  std::vector<char const*> h_strings{"Hello", "World", "0123456789", nullptr, ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.cbegin(),
    h_strings.cend(),
    thrust::make_transform_iterator(h_strings.cbegin(),
                                    [](auto const str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::utf8_to_cp932(strings_view);

  // ASCII is the same in UTF-8 and CP932
  cudf::test::strings_column_wrapper expected(
    h_strings.cbegin(),
    h_strings.cend(),
    thrust::make_transform_iterator(h_strings.cbegin(),
                                    [](auto const str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCp932Test, Utf8ToCp932Hiragana)
{
  // UTF-8: あ = E3 81 82, い = E3 81 84
  // CP932: あ = 82 A0, い = 82 A2
  std::vector<char const*> h_strings{"\xE3\x81\x82\xE3\x81\x84"};  // "あい"
  cudf::test::strings_column_wrapper strings(h_strings.cbegin(), h_strings.cend());

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::utf8_to_cp932(strings_view);

  std::vector<char const*> h_expected{"\x82\xA0\x82\xA2"};  // "あい" in CP932
  cudf::test::strings_column_wrapper expected(h_expected.cbegin(), h_expected.cend());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCp932Test, Utf8ToCp932Katakana)
{
  // UTF-8: ア = E3 82 A2, イ = E3 82 A4
  // CP932: ア = 83 41, イ = 83 43
  std::vector<char const*> h_strings{"\xE3\x82\xA2\xE3\x82\xA4"};  // "アイ"
  cudf::test::strings_column_wrapper strings(h_strings.cbegin(), h_strings.cend());

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::utf8_to_cp932(strings_view);

  std::vector<char const*> h_expected{"\x83\x41\x83\x43"};  // "アイ" in CP932
  cudf::test::strings_column_wrapper expected(h_expected.cbegin(), h_expected.cend());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCp932Test, Utf8ToCp932HalfWidthKatakana)
{
  // Half-width katakana (U+FF61-U+FF9F) maps to single-byte 0xA1-0xDF in CP932
  // UTF-8: ｱ = EF BD B1 (U+FF71), ｲ = EF BD B2 (U+FF72)
  // CP932: ｱ = B1, ｲ = B2 (single byte)
  std::vector<char const*> h_strings{"\xEF\xBD\xB1\xEF\xBD\xB2"};  // "ｱｲ"
  cudf::test::strings_column_wrapper strings(h_strings.cbegin(), h_strings.cend());

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::utf8_to_cp932(strings_view);

  std::vector<char const*> h_expected{"\xB1\xB2"};  // "ｱｲ" in CP932 (single-byte)
  cudf::test::strings_column_wrapper expected(h_expected.cbegin(), h_expected.cend());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCp932Test, Utf8ToCp932Kanji)
{
  // UTF-8: 日 = E6 97 A5, 本 = E6 9C AC
  // CP932: 日 = 93 FA, 本 = 96 7B
  std::vector<char const*> h_strings{"\xE6\x97\xA5\xE6\x9C\xAC"};  // "日本"
  cudf::test::strings_column_wrapper strings(h_strings.cbegin(), h_strings.cend());

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::utf8_to_cp932(strings_view);

  std::vector<char const*> h_expected{"\x93\xFA\x96\x7B"};  // "日本" in CP932
  cudf::test::strings_column_wrapper expected(h_expected.cbegin(), h_expected.cend());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCp932Test, Utf8ToCp932Mixed)
{
  // Mixed ASCII and Japanese
  // UTF-8: "Hello日本" = 48 65 6C 6C 6F E6 97 A5 E6 9C AC
  // CP932: "Hello日本" = 48 65 6C 6C 6F 93 FA 96 7B
  std::vector<char const*> h_strings{"Hello\xE6\x97\xA5\xE6\x9C\xAC"};  // "Hello日本"
  cudf::test::strings_column_wrapper strings(h_strings.cbegin(), h_strings.cend());

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::utf8_to_cp932(strings_view);

  std::vector<char const*> h_expected{"Hello\x93\xFA\x96\x7B"};  // "Hello日本" in CP932
  cudf::test::strings_column_wrapper expected(h_expected.cbegin(), h_expected.cend());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCp932Test, Utf8ToCp932NullAndEmpty)
{
  std::vector<char const*> h_strings{nullptr, "", "test", nullptr, ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.cbegin(),
    h_strings.cend(),
    thrust::make_transform_iterator(h_strings.cbegin(),
                                    [](auto const str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::utf8_to_cp932(strings_view);

  cudf::test::strings_column_wrapper expected(
    h_strings.cbegin(),
    h_strings.cend(),
    thrust::make_transform_iterator(h_strings.cbegin(),
                                    [](auto const str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCp932Test, Utf8ToCp932ErrorOnEmoji)
{
  // Emoji characters cannot be represented in CP932
  // UTF-8: 😀 = F0 9F 98 80 (U+1F600)
  std::vector<char const*> h_strings{"Hello \xF0\x9F\x98\x80"};  // "Hello 😀"
  cudf::test::strings_column_wrapper strings(h_strings.cbegin(), h_strings.cend());

  auto strings_view = cudf::strings_column_view(strings);

  // Should throw an exception because emoji cannot be converted to CP932
  EXPECT_THROW(cudf::strings::utf8_to_cp932(strings_view), cudf::logic_error);
}

TEST_F(StringsCp932Test, Utf8ToCp932EmptyColumn)
{
  cudf::test::strings_column_wrapper strings{};
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::utf8_to_cp932(strings_view);
  EXPECT_EQ(results->size(), 0);
}
