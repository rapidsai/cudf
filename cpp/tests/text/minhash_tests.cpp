/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <nvtext/minhash.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <vector>

struct MinHashTest : public cudf::test::BaseFixture {};

TEST_F(MinHashTest, Basic)
{
  auto validity = cudf::test::iterators::null_at(1);
  auto input =
    cudf::test::strings_column_wrapper({"doc 1",
                                        "",
                                        "this is doc 2",
                                        "",
                                        "doc 3",
                                        "d",
                                        "The quick brown fox jumpéd over the lazy brown dog."},
                                       validity);

  auto view = cudf::strings_column_view(input);

  auto results = nvtext::minhash(view);

  auto expected = cudf::test::fixed_width_column_wrapper<uint32_t>(
    {1207251914u, 0u, 21141582u, 0u, 1207251914u, 655955059u, 86520422u}, validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto results64  = nvtext::minhash64(view);
  auto expected64 = cudf::test::fixed_width_column_wrapper<uint64_t>({774489391575805754ul,
                                                                      0ul,
                                                                      3232308021562742685ul,
                                                                      0ul,
                                                                      13145552576991307582ul,
                                                                      14660046701545912182ul,
                                                                      398062025280761388ul},
                                                                     validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, LengthEqualsWidth)
{
  auto input   = cudf::test::strings_column_wrapper({"abcdé", "fghjk", "lmnop", "qrstu", "vwxyz"});
  auto view    = cudf::strings_column_view(input);
  auto results = nvtext::minhash(view, 0, 5);
  auto expected = cudf::test::fixed_width_column_wrapper<uint32_t>(
    {3825281041u, 2728681928u, 1984332911u, 3965004915u, 192452857u});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(MinHashTest, MultiSeed)
{
  auto input =
    cudf::test::strings_column_wrapper({"doc 1",
                                        "this is doc 2",
                                        "doc 3",
                                        "d",
                                        "The quick brown fox jumpéd over the lazy brown dog."});

  auto view = cudf::strings_column_view(input);

  auto seeds   = cudf::test::fixed_width_column_wrapper<uint32_t>({0, 1, 2});
  auto results = nvtext::minhash(view, cudf::column_view(seeds));

  using LCW = cudf::test::lists_column_wrapper<uint32_t>;
  // clang-format off
  LCW expected({LCW{1207251914u, 1677652962u, 1061355987u},
                LCW{  21141582u,  580916568u, 1258052021u},
                LCW{1207251914u,  943567174u, 1109272887u},
                LCW{ 655955059u,  488346356u, 2394664816u},
                LCW{  86520422u,  236622901u,  102546228u}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto seeds64   = cudf::test::fixed_width_column_wrapper<uint64_t>({0, 1, 2});
  auto results64 = nvtext::minhash64(view, cudf::column_view(seeds64));

  using LCW64 = cudf::test::lists_column_wrapper<uint64_t>;
  // clang-format off
  LCW64 expected64({LCW64{  774489391575805754ul, 10435654231793485448ul, 1188598072697676120ul},
                    LCW64{ 3232308021562742685ul,  4445611509348165860ul, 1188598072697676120ul},
                    LCW64{13145552576991307582ul,  6846192680998069919ul, 1188598072697676120ul},
                    LCW64{14660046701545912182ul, 17106501326045553694ul, 17713478494106035784ul},
                    LCW64{  398062025280761388ul,   377720198157450084ul,  984941365662009329ul}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, MultiSeedWithNullInputRow)
{
  auto validity = cudf::test::iterators::null_at(1);
  auto input    = cudf::test::strings_column_wrapper({"abcdéfgh", "", "", "stuvwxyz"}, validity);
  auto view     = cudf::strings_column_view(input);

  auto seeds   = cudf::test::fixed_width_column_wrapper<uint32_t>({1, 2});
  auto results = nvtext::minhash(view, cudf::column_view(seeds));

  using LCW = cudf::test::lists_column_wrapper<uint32_t>;
  LCW expected({LCW{484984072u, 1074168784u}, LCW{}, LCW{0u, 0u}, LCW{571652169u, 173528385u}},
               validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto seeds64   = cudf::test::fixed_width_column_wrapper<uint64_t>({11, 22});
  auto results64 = nvtext::minhash64(view, cudf::column_view(seeds64));

  using LCW64 = cudf::test::lists_column_wrapper<uint64_t>;
  LCW64 expected64({LCW64{2597399324547032480ul, 4461410998582111052ul},
                    LCW64{},
                    LCW64{0ul, 0ul},
                    LCW64{2717781266371273264ul, 6977325820868387259ul}},
                   validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, EmptyTest)
{
  auto input   = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  auto view    = cudf::strings_column_view(input->view());
  auto results = nvtext::minhash(view);
  EXPECT_EQ(results->size(), 0);
  results = nvtext::minhash64(view);
  EXPECT_EQ(results->size(), 0);
}

TEST_F(MinHashTest, ErrorsTest)
{
  auto input = cudf::test::strings_column_wrapper({"this string intentionally left blank"});
  auto view  = cudf::strings_column_view(input);
  EXPECT_THROW(nvtext::minhash(view, 0, 0), std::invalid_argument);
  EXPECT_THROW(nvtext::minhash64(view, 0, 0), std::invalid_argument);
  auto seeds = cudf::test::fixed_width_column_wrapper<uint32_t>();
  EXPECT_THROW(nvtext::minhash(view, cudf::column_view(seeds)), std::invalid_argument);
  auto seeds64 = cudf::test::fixed_width_column_wrapper<uint64_t>();
  EXPECT_THROW(nvtext::minhash64(view, cudf::column_view(seeds64)), std::invalid_argument);

  std::vector<std::string> h_input(50000, "");
  input = cudf::test::strings_column_wrapper(h_input.begin(), h_input.end());
  view  = cudf::strings_column_view(input);

  auto const zeroes = thrust::constant_iterator<uint32_t>(0);
  seeds             = cudf::test::fixed_width_column_wrapper<uint32_t>(zeroes, zeroes + 50000);
  EXPECT_THROW(nvtext::minhash(view, cudf::column_view(seeds)), std::overflow_error);
  seeds64 = cudf::test::fixed_width_column_wrapper<uint64_t>(zeroes, zeroes + 50000);
  EXPECT_THROW(nvtext::minhash64(view, cudf::column_view(seeds64)), std::overflow_error);
}
