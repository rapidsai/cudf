/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <nvtext/unicode_normalize.hpp>

#include <vector>

struct TextUnicodeNormalizeTest : public cudf::test::BaseFixture {};

// Helper to build a minimal unicode_data table with three columns:
//   col[0]: STRING  - hex codepoint strings (e.g. "00E9")
//   col[1]: INT32   - CCC (canonical combining class) values
//   col[2]: STRING  - decomposition mapping strings (e.g. "0065 0301")
static cudf::table_view make_unicode_table(
  cudf::test::strings_column_wrapper& codepoints,
  cudf::test::fixed_width_column_wrapper<int32_t>& ccc_values,
  cudf::test::strings_column_wrapper& decomp_mappings)
{
  return cudf::table_view({codepoints, ccc_values, decomp_mappings});
}

TEST_F(TextUnicodeNormalizeTest, EmptyInput)
{
  auto empty = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  cudf::strings_column_view input(empty->view());

  cudf::test::strings_column_wrapper codepoints(std::initializer_list<std::string>{});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values(std::initializer_list<int32_t>{});
  cudf::test::strings_column_wrapper decomp_mappings(std::initializer_list<std::string>{});
  auto unicode_data = make_unicode_table(codepoints, ccc_values, decomp_mappings);

  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);
  auto result = nvtext::normalize_unicode(input, *normalizer);
  EXPECT_EQ(result->size(), 0);

  normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFD);
  result = nvtext::normalize_unicode(input, *normalizer);
  EXPECT_EQ(result->size(), 0);
}

TEST_F(TextUnicodeNormalizeTest, NullStrings)
{
  cudf::test::strings_column_wrapper strings({"", "", ""}, {false, false, false});
  cudf::strings_column_view input(strings);

  cudf::test::strings_column_wrapper codepoints(std::initializer_list<std::string>{});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values(std::initializer_list<int32_t>{});
  cudf::test::strings_column_wrapper decomp_mappings(std::initializer_list<std::string>{});
  auto unicode_data = make_unicode_table(codepoints, ccc_values, decomp_mappings);

  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);
  auto result = nvtext::normalize_unicode(input, *normalizer);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, strings);
}

TEST_F(TextUnicodeNormalizeTest, AsciiPassthrough)
{
  // ASCII-only input should be unchanged for all four normalization forms
  cudf::test::strings_column_wrapper strings({"hello", "world", "abc 123", ""});
  cudf::strings_column_view input(strings);

  // No codepoints needed for pure ASCII
  cudf::test::strings_column_wrapper codepoints(std::initializer_list<std::string>{});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values(std::initializer_list<int32_t>{});
  cudf::test::strings_column_wrapper decomp_mappings(std::initializer_list<std::string>{});
  auto unicode_data = make_unicode_table(codepoints, ccc_values, decomp_mappings);

  for (auto form : {nvtext::unicode_normalization_form::NFD,
                    nvtext::unicode_normalization_form::NFC,
                    nvtext::unicode_normalization_form::NFKD,
                    nvtext::unicode_normalization_form::NFKC}) {
    auto normalizer = nvtext::create_unicode_normalizer(unicode_data, form);
    auto result     = nvtext::normalize_unicode(input, *normalizer);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, strings);
  }
}

TEST_F(TextUnicodeNormalizeTest, NFD_BasicDecomp)
{
  // U+00E9 "é" decomposes to U+0065 "e" + U+0301 combining acute accent
  // Table entry: codepoint="00E9", CCC=0, decomp="0065 0301"
  cudf::test::strings_column_wrapper input_strings({"\xC3\xA9"});  // é in UTF-8
  cudf::strings_column_view input(input_strings);

  // U+0301 (CCC=230) must be in the table so the compose kernel treats it
  // as a combining mark (CCC>0) rather than a new starter.
  cudf::test::strings_column_wrapper codepoints({"00E9", "0301"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 230});
  cudf::test::strings_column_wrapper decomp_mappings({"0065 0301", ""});
  auto unicode_data = make_unicode_table(codepoints, ccc_values, decomp_mappings);

  // NFD: é → e + combining acute (U+0065 U+0301)
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFD);
  auto result = nvtext::normalize_unicode(input, *normalizer);

  // Expected: "e" + U+0301 combining acute accent
  cudf::test::strings_column_wrapper expected({"e\xCC\x81"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, NFD_Hangul)
{
  // U+AC00 "가" algorithmically decomposes to U+1100 + U+1161 under NFD
  // No table entry needed for algorithmic Hangul decomposition
  cudf::test::strings_column_wrapper input_strings({"\xEA\xB0\x80"});  // 가 in UTF-8
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints(std::initializer_list<std::string>{});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values(std::initializer_list<int32_t>{});
  cudf::test::strings_column_wrapper decomp_mappings(std::initializer_list<std::string>{});
  auto unicode_data = make_unicode_table(codepoints, ccc_values, decomp_mappings);

  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFD);
  auto result = nvtext::normalize_unicode(input, *normalizer);

  // Expected: U+1100 (ᄀ) + U+1161 (ᅡ) in UTF-8
  cudf::test::strings_column_wrapper expected({"\xE1\x84\x80\xE1\x85\xA1"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, NFC_Compose)
{
  // NFC: e + combining acute → é (U+00E9)
  // Input is already decomposed: U+0065 + U+0301
  // Table needs U+00E9 so the composition lookup can find it
  cudf::test::strings_column_wrapper input_strings({"e\xCC\x81"});  // e + combining acute
  cudf::strings_column_view input(input_strings);

  // U+0301 (CCC=230) must be in the table so the compose kernel treats it
  // as a combining mark (CCC>0) rather than a new starter.
  cudf::test::strings_column_wrapper codepoints({"00E9", "0301"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 230});
  cudf::test::strings_column_wrapper decomp_mappings({"0065 0301", ""});
  auto unicode_data = make_unicode_table(codepoints, ccc_values, decomp_mappings);

  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);
  auto result = nvtext::normalize_unicode(input, *normalizer);

  // Expected: é (U+00E9) in UTF-8
  cudf::test::strings_column_wrapper expected({"\xC3\xA9"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, NFKD_CompatDecomp)
{
  // U+FB01 "ﬁ" (fi ligature) has compatibility decomposition: "0066 0069" (fi)
  // Under NFD it is unchanged (no canonical decomposition)
  // Under NFKD it expands to "fi"
  cudf::test::strings_column_wrapper input_strings({"\xEF\xAC\x81"});  // ﬁ in UTF-8
  cudf::strings_column_view input(input_strings);

  // Compat decomp is indicated by "<compat>" prefix in the decomp mapping
  cudf::test::strings_column_wrapper codepoints({"FB01"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0});
  cudf::test::strings_column_wrapper decomp_mappings({"<compat> 0066 0069"});
  auto unicode_data = make_unicode_table(codepoints, ccc_values, decomp_mappings);

  // NFD: ﬁ is unchanged (compatibility decomp not applied)
  auto normalizer_nfd =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFD);
  auto result_nfd = nvtext::normalize_unicode(input, *normalizer_nfd);
  cudf::test::strings_column_wrapper expected_nfd({"\xEF\xAC\x81"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result_nfd, expected_nfd);

  // NFKD: ﬁ → "fi"
  auto normalizer_nfkd =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFKD);
  auto result_nfkd = nvtext::normalize_unicode(input, *normalizer_nfkd);
  cudf::test::strings_column_wrapper expected_nfkd({"fi"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result_nfkd, expected_nfkd);
}

TEST_F(TextUnicodeNormalizeTest, NFKC_CompatThenCompose)
{
  // U+FB01 "ﬁ" → NFKC → "fi"
  // compat decomp gives "fi"; f+i has no canonical composition so stays "fi"
  cudf::test::strings_column_wrapper input_strings({"\xEF\xAC\x81"});  // ﬁ in UTF-8
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"FB01"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0});
  cudf::test::strings_column_wrapper decomp_mappings({"<compat> 0066 0069"});
  auto unicode_data = make_unicode_table(codepoints, ccc_values, decomp_mappings);

  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFKC);
  auto result = nvtext::normalize_unicode(input, *normalizer);

  cudf::test::strings_column_wrapper expected({"fi"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, CanonicalReorder)
{
  // Test that combining marks are reordered by CCC after NFD decomposition.
  // Construct a string with two combining marks in the wrong order:
  //   U+0041 'A' + U+0316 (CCC=220) + U+0300 (CCC=230)
  // After canonical decomposition and reorder the CCC=220 mark stays before CCC=230.
  // Here we use a string that already has a base + two combining characters
  // where the higher CCC comes first — reorder should swap them.
  //
  // U+0300: combining grave accent      CCC=230
  // U+0316: combining grave accent below CCC=220
  //
  // Input: 'A' + U+0300 (CCC=230) + U+0316 (CCC=220)  — wrong order
  // After NFD reorder: 'A' + U+0316 (CCC=220) + U+0300 (CCC=230) — correct order

  // UTF-8: A=0x41, U+0300=0xCC 0x80, U+0316=0xCC 0x96
  cudf::test::strings_column_wrapper input_strings({"A\xCC\x80\xCC\x96"});
  cudf::strings_column_view input(input_strings);

  // Table entries for the two combining marks (no decomposition, just CCC values)
  cudf::test::strings_column_wrapper codepoints({"0300", "0316"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({230, 220});
  cudf::test::strings_column_wrapper decomp_mappings({"", ""});
  auto unicode_data = make_unicode_table(codepoints, ccc_values, decomp_mappings);

  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFD);
  auto result = nvtext::normalize_unicode(input, *normalizer);

  // Expected: 'A' + U+0316 (CCC=220) + U+0300 (CCC=230)
  cudf::test::strings_column_wrapper expected({"A\xCC\x96\xCC\x80"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, MultiStringBatch)
{
  // A batch of 5 strings with mixed content; verify per-row correctness under NFC.
  // Rows:
  //   0: "hello"      → "hello"   (ASCII passthrough)
  //   1: ""           → ""        (empty)
  //   2: nullptr      → nullptr   (null)
  //   3: "e\xCC\x81" → "é"       (compose e + acute → U+00E9)
  //   4: "café"       → "café"    (already NFC)

  std::vector<char const*> h_input{"hello", "", nullptr, "e\xCC\x81", "caf\xC3\xA9"};
  cudf::test::strings_column_wrapper input_strings(
    h_input.begin(), h_input.end(), std::vector<bool>{true, true, false, true, true}.begin());
  cudf::strings_column_view input(input_strings);

  // Table: U+00E9 with canonical decomp "0065 0301"
  // U+0301 (CCC=230) must be in the table so the compose kernel treats it
  // as a combining mark (CCC>0) rather than a new starter.
  cudf::test::strings_column_wrapper codepoints({"00E9", "0301"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 230});
  cudf::test::strings_column_wrapper decomp_mappings({"0065 0301", ""});
  auto unicode_data = make_unicode_table(codepoints, ccc_values, decomp_mappings);

  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);
  auto result = nvtext::normalize_unicode(input, *normalizer);

  std::vector<char const*> h_expected{"hello", "", nullptr, "\xC3\xA9", "caf\xC3\xA9"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), std::vector<bool>{true, true, false, true, true}.begin());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, ErrorWrongColumnCount)
{
  cudf::test::strings_column_wrapper col_a({"00E9"});
  cudf::test::fixed_width_column_wrapper<int32_t> col_b({0});
  cudf::table_view t({col_a, col_b});
  EXPECT_THROW(nvtext::create_unicode_normalizer(t, nvtext::unicode_normalization_form::NFC),
               std::invalid_argument);
}

TEST_F(TextUnicodeNormalizeTest, ErrorWrongColumnType)
{
  auto const form = nvtext::unicode_normalization_form::NFC;
  cudf::test::fixed_width_column_wrapper<int32_t> intcol({0x00E9});
  cudf::test::strings_column_wrapper strcol({"0065 0301"});

  EXPECT_THROW(nvtext::create_unicode_normalizer(cudf::table_view({intcol, intcol, strcol}), form),
               std::invalid_argument);
  EXPECT_THROW(nvtext::create_unicode_normalizer(cudf::table_view({strcol, strcol, strcol}), form),
               std::invalid_argument);
  EXPECT_THROW(nvtext::create_unicode_normalizer(cudf::table_view({strcol, intcol, intcol}), form),
               std::invalid_argument);
}

TEST_F(TextUnicodeNormalizeTest, ErrorNullsInColumns)
{
  auto const form = nvtext::unicode_normalization_form::NFC;
  cudf::test::strings_column_wrapper col0({"00E9"}, {false});
  cudf::test::fixed_width_column_wrapper<int32_t> col1({0});
  cudf::test::strings_column_wrapper col2({"0065 0301"});
  EXPECT_THROW(nvtext::create_unicode_normalizer(cudf::table_view({col0, col1, col2}), form),
               std::invalid_argument);
}
