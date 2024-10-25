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

TEST_F(MinHashTest, Permuted)
{
  auto input =
    cudf::test::strings_column_wrapper({"doc 1",
                                        "this is doc 2",
                                        "doc 3",
                                        "d",
                                        "The quick brown fox jumpéd over the lazy brown dog.",
                                        "line six",
                                        "line seven",
                                        "line eight",
                                        "line nine",
                                        "line ten"});

  auto view = cudf::strings_column_view(input);

  auto first  = thrust::counting_iterator<uint32_t>(10);
  auto params = cudf::test::fixed_width_column_wrapper<uint32_t>(first, first + 3);
  auto results =
    nvtext::minhash_permuted(view, 0, cudf::column_view(params), cudf::column_view(params), 4);

  using LCW32 = cudf::test::lists_column_wrapper<uint32_t>;
  // clang-format off
  LCW32 expected({
    LCW32{1392101586u,  394869177u,  811528444u},
    LCW32{ 211415830u,  187088503u,  130291444u},
    LCW32{2098117052u,  394869177u,  799753544u},
    LCW32{2264583304u, 2920538364u, 3576493424u},
    LCW32{ 253327882u,   41747273u,  302030804u},
    LCW32{2109809594u, 1017470651u,  326988172u},
    LCW32{1303819864u,  850676747u,  147107852u},
    LCW32{ 736021564u,  720812292u, 1405158760u},
    LCW32{ 902780242u,  134064807u, 1613944636u},
    LCW32{ 547084870u, 1748895564u,  656501844u}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto params64  = cudf::test::fixed_width_column_wrapper<uint64_t, uint32_t>(first, first + 3);
  auto results64 = nvtext::minhash64_permuted(
    view, 0, cudf::column_view(params64), cudf::column_view(params64), 4);

  using LCW64 = cudf::test::lists_column_wrapper<uint64_t>;
  // clang-format off
  LCW64 expected64({
    LCW64{ 827364888116975697ul, 1601854279692781452ul,  70500662054893256ul},
    LCW64{  18312093741021833ul,  133793446674258329ul,  21974512489226198ul},
    LCW64{  22474244732520567ul, 1638811775655358395ul, 949306297364502264ul},
    LCW64{1332357434996402861ul, 2157346081260151330ul, 676491718310205848ul},
    LCW64{  65816830624808020ul,   43323600380520789ul,  63511816333816345ul},
    LCW64{ 629657184954525200ul,   49741036507643002ul,  97466271004074331ul},
    LCW64{ 301611977846331113ul,  101188874709594830ul,  97466271004074331ul},
    LCW64{ 121498891461700668ul,  171065800427907402ul,  97466271004074331ul},
    LCW64{  54617739511834072ul,  231454301607238929ul,  97466271004074331ul},
    LCW64{ 576418665851990314ul,  231454301607238929ul,  97466271004074331ul}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, PermutedWide)
{
  std::string const small(2 << 10, 'x');  // below wide_string_threshold
  std::string const wide(2 << 19, 'y');   // above wide_string_threshold
  auto input = cudf::test::strings_column_wrapper({small, wide});
  auto view  = cudf::strings_column_view(input);

  auto first  = thrust::counting_iterator<uint32_t>(20);
  auto params = cudf::test::fixed_width_column_wrapper<uint32_t>(first, first + 3);
  auto results =
    nvtext::minhash_permuted(view, 0, cudf::column_view(params), cudf::column_view(params), 4);

  using LCW32 = cudf::test::lists_column_wrapper<uint32_t>;
  // clang-format off
  LCW32 expected({
    LCW32{1731998032u,  315359380u, 3193688024u},
    LCW32{1293098788u, 2860992281u,  133918478u}
  });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto params64  = cudf::test::fixed_width_column_wrapper<uint64_t, uint32_t>(first, first + 3);
  auto results64 = nvtext::minhash64_permuted(
    view, 0, cudf::column_view(params64), cudf::column_view(params64), 4);

  using LCW64 = cudf::test::lists_column_wrapper<uint64_t>;
  // clang-format off
   LCW64 expected64({
     LCW64{1818322427062143853ul, 641024893347719371ul, 1769570368846988848ul},
     LCW64{1389920339306667795ul, 421787002125838902ul, 1759496674158703968ul}
   });
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results64, expected64);
}

TEST_F(MinHashTest, EmptyTest)
{
  auto input  = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  auto view   = cudf::strings_column_view(input->view());
  auto params = cudf::test::fixed_width_column_wrapper<uint32_t>({1, 2, 3});
  auto results =
    nvtext::minhash_permuted(view, 0, cudf::column_view(params), cudf::column_view(params), 4);
  EXPECT_EQ(results->size(), 0);
  auto params64 = cudf::test::fixed_width_column_wrapper<uint64_t>({1, 2, 3});
  results       = nvtext::minhash64_permuted(
    view, 0, cudf::column_view(params64), cudf::column_view(params64), 4);
  EXPECT_EQ(results->size(), 0);
}

TEST_F(MinHashTest, ErrorsTest)
{
  auto input = cudf::test::strings_column_wrapper({"this string intentionally left blank"});
  auto view  = cudf::strings_column_view(input);
  auto empty = cudf::test::fixed_width_column_wrapper<uint32_t>();
  EXPECT_THROW(
    nvtext::minhash_permuted(view, 0, cudf::column_view(empty), cudf::column_view(empty), 0),
    std::invalid_argument);
  auto empty64 = cudf::test::fixed_width_column_wrapper<uint64_t>();
  EXPECT_THROW(
    nvtext::minhash64_permuted(view, 0, cudf::column_view(empty64), cudf::column_view(empty64), 0),
    std::invalid_argument);
  EXPECT_THROW(
    nvtext::minhash_permuted(view, 0, cudf::column_view(empty), cudf::column_view(empty), 4),
    std::invalid_argument);
  EXPECT_THROW(
    nvtext::minhash64_permuted(view, 0, cudf::column_view(empty64), cudf::column_view(empty64), 4),
    std::invalid_argument);

  std::vector<std::string> h_input(50000, "");
  input = cudf::test::strings_column_wrapper(h_input.begin(), h_input.end());
  view  = cudf::strings_column_view(input);

  auto const zeroes = thrust::constant_iterator<uint32_t>(0);
  auto params       = cudf::test::fixed_width_column_wrapper<uint32_t>(zeroes, zeroes + 50000);
  EXPECT_THROW(
    nvtext::minhash_permuted(view, 0, cudf::column_view(params), cudf::column_view(params), 4),
    std::overflow_error);
  auto params64 = cudf::test::fixed_width_column_wrapper<uint64_t>(zeroes, zeroes + 50000);
  EXPECT_THROW(nvtext::minhash64_permuted(
                 view, 0, cudf::column_view(params64), cudf::column_view(params64), 4),
               std::overflow_error);

  EXPECT_THROW(
    nvtext::minhash_permuted(view, 0, cudf::column_view(params), cudf::column_view(empty), 4),
    std::invalid_argument);
  EXPECT_THROW(
    nvtext::minhash64_permuted(view, 0, cudf::column_view(params64), cudf::column_view(empty64), 4),
    std::invalid_argument);
}
