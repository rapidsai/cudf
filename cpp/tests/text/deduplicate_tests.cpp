/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/deduplicate.hpp>

#include <vector>

struct TextDeduplicateTest : public cudf::test::BaseFixture {};

namespace {
cudf::test::strings_column_wrapper build_input()
{
  // https://loremipsum.io/generator?n=25&t=p
  // clang-format off
  auto input = cudf::test::strings_column_wrapper({
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ", //  90
    "01234567890123456789 magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation     ", // 180
    "laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit   ", // 270
    "voluptate velit esse cillum dolore eu fugiat nulla pariatur. 01234567890123456789         ", // 360
    "cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.    ", // 450
    "Ea esse numquam et recusandae quia et voluptatem sint quo explicabo repudiandae. At nihil ", // 540
    "sunt non architecto doloremque eos dolorem consequuntur. Vel adipisci quod et voluptatum  ", // 630
    "quis est fuga tempore qui dignissimos aliquam et sint repellendus ut autem voluptas quo   ", // 720
    "deleniti earum? Qui ipsam ipsum hic ratione mollitia aut nobis laboriosam. Eum aspernatur ", // 810
    "dolorem sit voluptatum numquam in iure placeat vel laudantium molestiae? Ad reprehenderit ", // 900
    "quia aut minima deleniti id consequatur sapiente est dolores cupiditate. 012345678901234  ", // 990
  });
  // clang-format on
  return input;
}
}  // namespace

TEST_F(TextDeduplicateTest, SuffixArray)
{
  auto const input = cudf::test::strings_column_wrapper({
    "cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. ",
    "quis est fuga tempore qui dignissimos aliquam et sint repellendus ut autem voluptas quo",
  });

  auto const sv = cudf::strings_column_view(input);

  auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    {124, 65,  155, 31,  49,  112, 91,  73,  132, 95,  70,  28,  77,  58,  9,   41,  13,  108, 37,
     86,  140, 135, 23,  100, 152, 161, 22,  85,  48,  36,  99,  79,  125, 130, 66,  7,   5,   156,
     80,  46,  32,  0,   72,  4,   18,  50,  113, 149, 107, 144, 159, 102, 147, 19,  142, 53,  51,
     92,  74,  133, 43,  44,  96,  98,  115, 111, 40,  47,  45,  71,  3,   17,  114, 68,  120, 29,
     137, 127, 89,  117, 63,  78,  146, 126, 62,  145, 61,  34,  164, 131, 69,  160, 84,  59,  121,
     103, 30,  12,  148, 67,  116, 10,  26,  56,  138, 20,  42,  16,  60,  163, 11,  105, 81,  122,
     35,  143, 2,   104, 14,  166, 128, 109, 38,  87,  106, 141, 15,  82,  54,  123, 90,  151, 52,
     119, 136, 118, 93,  75,  24,  64,  154, 94,  27,  76,  57,  8,   139, 134, 21,  6,   158, 101,
     129, 97,  110, 39,  88,  33,  83,  25,  55,  1,   165, 150, 153, 157, 162});

  auto const results = nvtext::build_suffix_array(sv, 8);
  auto const col_view =
    cudf::column_view(cudf::device_span<cudf::size_type const>(results->data(), results->size()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, col_view);
}

TEST_F(TextDeduplicateTest, ResolveDuplicates)
{
  auto input = build_input();
  auto sv    = cudf::strings_column_view(input);

  auto sa       = nvtext::build_suffix_array(sv, 0);
  auto results  = nvtext::resolve_duplicates(sv, *sa, 20);
  auto expected = cudf::test::strings_column_wrapper({" 01234567890123456789 "});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  results  = nvtext::resolve_duplicates(sv, *sa, 15);
  expected = cudf::test::strings_column_wrapper(
    {" 01234567890123456789 ", ". 012345678901234", " reprehenderit "});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  // Test with sliced input
  auto sliced = cudf::slice(input, {1, 10}).front();

  sv       = cudf::strings_column_view(sliced);
  sa       = nvtext::build_suffix_array(sv, 0);
  results  = nvtext::resolve_duplicates(sv, *sa, 15);
  expected = cudf::test::strings_column_wrapper({"01234567890123456789 ", " reprehenderit "});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
}

TEST_F(TextDeduplicateTest, ResolvePair)
{
  auto const input = build_input();
  auto const sv    = cudf::strings_column_view(input);
  auto const split = cudf::split(input, {3});
  auto const sv1   = split.front();
  auto const sv2   = split.back();

  auto sa1 = nvtext::build_suffix_array(sv1, 0);
  auto sa2 = nvtext::build_suffix_array(sv2, 0);

  auto results  = nvtext::resolve_duplicates_pair(sv1, *sa1, sv2, *sa2, 15);
  auto expected = cudf::test::strings_column_wrapper(
    {" 01234567890123456789 ", " 012345678901234", " reprehenderit "});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  auto results2 = nvtext::resolve_duplicates_pair(sv2, *sa2, sv1, *sa1, 15);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), results2->view());
}

TEST_F(TextDeduplicateTest, Errors)
{
  auto const input = cudf::test::strings_column_wrapper({"0123456789"});
  auto const sv    = cudf::strings_column_view(input);

  EXPECT_THROW(nvtext::build_suffix_array(sv, 50), std::invalid_argument);

  auto sa = nvtext::build_suffix_array(sv, 8);
  EXPECT_THROW(nvtext::resolve_duplicates(sv, *sa, 5), std::invalid_argument);
  EXPECT_THROW(nvtext::resolve_duplicates(sv, *sa, 50), std::invalid_argument);
  EXPECT_THROW(nvtext::resolve_duplicates_pair(sv, *sa, sv, *sa, 5), std::invalid_argument);
  EXPECT_THROW(nvtext::resolve_duplicates_pair(sv, *sa, sv, *sa, 50), std::invalid_argument);
}
