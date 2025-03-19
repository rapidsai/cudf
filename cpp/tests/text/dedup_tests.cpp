/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/dedup.hpp>

#include <vector>

struct TextDedupTest : public cudf::test::BaseFixture {};

TEST_F(TextDedupTest, StringDedup)
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

  auto sv = cudf::strings_column_view(input);

  auto results  = nvtext::substring_duplicates(sv, 20);
  auto expected = cudf::test::strings_column_wrapper({" 01234567890123456789 "});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  results  = nvtext::substring_duplicates(sv, 15);
  expected = cudf::test::strings_column_wrapper(
    {" 01234567890123456789 ", ". 012345678901234", " reprehenderit "});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  // Test with sliced input
  auto const sliced_input = cudf::slice(input, {1, 10}).front();

  sv       = cudf::strings_column_view(sliced_input);
  results  = nvtext::substring_duplicates(sv, 15);
  expected = cudf::test::strings_column_wrapper({"01234567890123456789 ", " reprehenderit "});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
}

TEST_F(TextDedupTest, SuffixArray)
{
  auto input = cudf::test::strings_column_wrapper({
    "cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. ",
    "quis est fuga tempore qui dignissimos aliquam et sint repellendus ut autem voluptas quo",
  });

  auto sv = cudf::strings_column_view(input);

  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>(
    {124, 65,  155, 31,  49,  112, 91,  73,  132, 95,  70,  28,  77,  58,  9,   41,  13,  108, 37,
     86,  140, 135, 23,  100, 152, 161, 22,  85,  48,  36,  99,  79,  125, 130, 66,  7,   5,   156,
     80,  46,  32,  0,   72,  4,   18,  50,  113, 149, 107, 144, 159, 102, 147, 19,  142, 53,  51,
     92,  74,  133, 43,  44,  96,  98,  115, 111, 40,  47,  45,  71,  3,   17,  114, 68,  120, 29,
     137, 127, 89,  117, 63,  78,  146, 126, 62,  145, 61,  34,  164, 131, 69,  160, 84,  59,  121,
     103, 30,  12,  148, 67,  116, 10,  26,  56,  138, 20,  42,  16,  60,  163, 11,  105, 81,  122,
     35,  143, 2,   104, 14,  166, 128, 109, 38,  87,  106, 141, 15,  82,  54,  123, 90,  151, 52,
     119, 136, 118, 93,  75,  24,  64,  154, 94,  27,  76,  57,  8,   139, 134, 21,  6,   158, 101,
     129, 97,  110, 39,  88,  33,  83,  25,  55,  1,   165, 150, 153, 157, 162});

  auto results = nvtext::build_suffix_array(sv, 8);
  auto results_bitonic =
    std::make_unique<cudf::column>(std::move(*(results.release())), rmm::device_buffer{}, 0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results_bitonic->view());
}

TEST_F(TextDedupTest, StringDedupPair)
{
  // clang-format off
  auto input = cudf::test::strings_column_wrapper({
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ", //  90
    "01234567890123456789 magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation     ", // 180
    "laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit   ", // 270

    "voluptate velit esse cillum dolore eu fugiat nulla pariatur. 01234567890123456789         ", //  90
    "cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.    ", // 180
    "Ea esse numquam et recusandae quia et voluptatem sint quo explicabo repudiandae. At nihil ", // 270
    "sunt non architecto doloremque eos dolorem consequuntur. Vel adipisci quod et voluptatum  ", // 360
    "quis est fuga tempore qui dignissimos aliquam et sint repellendus ut autem voluptas quo   ", // 450
    "deleniti earum? Qui ipsam ipsum hic ratione mollitia aut nobis laboriosam. Eum aspernatur ", // 540
    "dolorem sit voluptatum numquam in iure placeat vel laudantium molestiae? Ad reprehenderit ", // 180
    "quia aut minima deleniti id consequatur sapiente est dolores cupiditate. 012345678901234  ", // 270
  });
  // clang-format on
  auto const sv          = cudf::strings_column_view(input);
  auto const split_input = cudf::split(input, {3});
  auto const sv1         = split_input.front();
  auto const sv2         = split_input.back();
  // clang-format off
  auto sa1 = nvtext::build_suffix_array(sv1, 0);
  // 270, 269, 268, 267, 175, 176, 177, 178,  89, 228, 124, 132,  39, 116, 195,  21, 233, 209,  27, 217,
  //  60, 244,  11, 206,  63,  50, 127, 203, 162, 250,  78,   5, 238, 179, 110, 135, 187, 154, 149, 253,
  //  56,  17,  71, 192, 141,  26, 148,  55, 227, 123, 100,  90, 101,  91, 102,  92, 103,  93, 104,  94,
  // 105,  95, 106,  96, 107,  97, 108,  98, 109,  99, 229,   0, 125, 115, 208, 122, 181, 133,  40, 112,
  // 117, 196, 146,  22, 225, 170, 234, 182,  81,  46, 167, 210,  28, 218,  33,  59, 161, 134,  70, 262,
  //  83,  41, 215,  61, 245,  12,  85, 243, 237, 207,  32,  58, 258,  64,  51,   3,  73, 260, 143, 128,
  // 255, 222, 165, 263,  24,  35, 204, 163,  49, 113, 259, 191, 145,  82,  84, 130, 139, 251,  79,  47,
  // 137, 172, 201,  42,   6, 119, 198, 239, 231, 185, 152,  44, 189, 265,  19,  53, 168,  65, 180, 118,
  // 197,  52, 247,  14, 131,  10,   4, 140, 147, 111,  23, 136, 212,  68, 213,  74, 174, 252, 114,  80,
  // 261,  48, 144, 129, 138, 188, 155,  30, 220,  87, 216,  62,  69, 214, 246,  13, 211, 173,  29, 219,
  // 248,  76,  15,   1, 183, 156, 202,  43,  75, 256,   7, 120, 223, 199, 150,  38, 249,  77,  16, 166,
  // 242, 257,   2, 254, 184, 264, 159, 240, 232, 186, 153,  45,  31,  57, 221, 190,  18,  67, 157,   8,
  // 266,  88, 194,  20, 126,  25,  54, 226, 169, 236,  72,  34, 171, 158,  36, 121, 224, 160, 200, 230,
  // 151,   9,  86,  37, 241,  66, 193, 235, 142, 205, 164
  auto sa2 = nvtext::build_suffix_array(sv2, 0);
  // 270, 269, 268,  81,  82,  83,  84,  85,  86,  87,  88, 252,  60, 162, 184,  20, 207, 240, 195,  27,
  //  89, 232,  15, 228,  34,  37, 204, 120, 123, 140, 188, 151,  44, 112,  50, 128, 179, 165, 219,  97,
  // 136,   9, 101, 251,  59, 263,  71, 253,  61, 264,  72, 254,  62, 265,  73, 255,  63, 266,  74, 256,
  //  64, 267,  75, 257,  65,  76, 258,  66,  77, 259,  67,  78, 260,  68,  79, 261,  69,  80, 262,  70,
  // 161, 163, 183, 194,  49, 131, 159, 118, 145, 221,  52,  42, 134,   6, 248, 108, 215,  55, 142, 185,
  // 132,  21, 208, 241, 206, 164, 144, 196, 174, 245,  28,  90, 233,  19, 227,  33, 127,   8, 250, 160,
  // 133, 170, 138, 197,  11,  95, 172, 199, 224, 167, 212, 175, 238,  16, 229, 155,  35,  38,  40, 171,
  // 203, 182, 158,  41,  54, 205, 244, 223,  22, 192, 121, 190,  13, 177,  99, 246, 201, 148, 124, 139,
  //  48, 130, 141, 198, 154,  12,  47,  23,  30,  92, 235,  24,   2, 104,  26, 119, 150, 111,  96, 193,
  // 189, 152, 115, 122, 173, 191, 200, 210, 225, 146,  45, 113, 153,  29,  91, 234,   1, 103, 209,  31,
  //  93, 236,  51, 243, 222, 129, 168,   4, 106, 116, 213, 180, 218,  58,  32, 126, 169,  94, 166, 237,
  //  53, 176, 239, 220,  18, 211,  98,  17, 230, 156, 231,  14, 187,  43, 178, 135, 100,   5, 247, 107,
  // 226,   7, 249, 202, 157, 147, 109, 216,  56,  36, 117, 214, 143,  39, 181,  46,  25, 149, 110, 114,
  // 242,   3, 105, 217,  57, 125, 186, 137,  10,   0, 102
  // clang-format on

  // cudf::test::print(cudf::column_view(cudf::device_span<int32_t const>(*sa1)));
  // cudf::test::print(cudf::column_view(cudf::device_span<int32_t const>(*sa2)));

  auto results  = nvtext::resolve_duplicates_pair(sv1, *sa1, sv2, *sa2, 15);
  auto expected = cudf::test::strings_column_wrapper(
    {" 01234567890123456789 ", " 012345678901234", " reprehenderit "});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  auto results2 = nvtext::resolve_duplicates_pair(sv2, *sa2, sv1, *sa1, 15);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), results2->view());
}
