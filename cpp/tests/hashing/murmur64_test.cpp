/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <cudf/detail/iterator.cuh>
#include <cudf/hashing.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

using NumericTypesNoBools =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

template <typename T>
class HashMurmur64TestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(HashMurmur64TestTyped, NumericTypesNoBools);

TYPED_TEST(HashMurmur64TestTyped, TestNumeric)
{
  using T   = TypeParam;
  auto col1 = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    {-1, -1, 0, 2, 22, 0, 11, 12, 116, 32, 0, 42, 7, 62, 1, -22, 0, 0},
    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}};
  auto col2 = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    {-1, -1, 0, 2, 22, 1, 11, 12, 116, 32, 0, 42, 7, 62, 1, -22, 1, -22},
    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}};

  auto const output1 = cudf::hashing::murmur_hash3_64_128(cudf::table_view({col1}));
  auto const output2 = cudf::hashing::murmur_hash3_64_128(cudf::table_view({col2}));

  CUDF_TEST_EXPECT_TABLES_EQUAL(output1->view(), output2->view());
}

class HashMurmur64Test : public cudf::test::BaseFixture {};

TEST_F(HashMurmur64Test, MultiType)
{
  auto col1 = cudf::test::strings_column_wrapper(
    {"The",
     "quick",
     "brown fox",
     "jumps over the lazy dog.",
     "I am Jack's complete lack of null value",
     "A very long (greater than 128 bytes/char string) to test a a very long string",
     "Some multi-byte characters here: ééé",
     "ééé",
     "ééé ééé",
     "ééé ééé ééé ééé",
     "",
     "!@#$%^&*(())",
     "0123456789",
     "{}|:<>?,./;[]=-"});

  auto output = cudf::hashing::murmur_hash3_64_128(cudf::table_view({col1}));
  // these were generated using the CPU compiled
  // https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
  auto expected = cudf::test::fixed_width_column_wrapper<uint64_t>({3481043174314896794ul,
                                                                    1981901315483788749ul,
                                                                    1418748153263580713ul,
                                                                    11224732510765974842ul,
                                                                    10813495276579975748ul,
                                                                    3654904410285196488ul,
                                                                    7289234017606107350ul,
                                                                    225672801045596944ul,
                                                                    14927688838032769435ul,
                                                                    7513581995808204968ul,
                                                                    0ul,
                                                                    14163495587303857889ul,
                                                                    4581940570640870180ul,
                                                                    18164432652839101653ul});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view().column(0), expected);

  using ts = cudf::timestamp_s;
  auto col2 =
    cudf::test::fixed_width_column_wrapper<ts, ts::duration>({ts::duration::zero(),
                                                              static_cast<ts::duration>(100),
                                                              static_cast<ts::duration>(-100),
                                                              ts::duration::min(),
                                                              static_cast<ts::duration>(1000),
                                                              static_cast<ts::duration>(1111),
                                                              static_cast<ts::duration>(100000),
                                                              static_cast<ts::duration>(-1000),
                                                              static_cast<ts::duration>(2222),
                                                              static_cast<ts::duration>(3333),
                                                              static_cast<ts::duration>(44444),
                                                              static_cast<ts::duration>(-100),
                                                              static_cast<ts::duration>(100),
                                                              ts::duration::max()});

  output = cudf::hashing::murmur_hash3_64_128(cudf::table_view({col1, col2}));

  expected = cudf::test::fixed_width_column_wrapper<uint64_t>({9466414547226101804ul,
                                                               8228850782181844430ul,
                                                               785745530268169861ul,
                                                               4419198613737028765ul,
                                                               8688495967673047490ul,
                                                               16639004178923553997ul,
                                                               5537069187226926761ul,
                                                               1827028865561218073ul,
                                                               11416255570436251115ul,
                                                               17794658852294150415ul,
                                                               5518440516384677761ul,
                                                               18392488402293221125ul,
                                                               14843016968729228293ul,
                                                               12478631074270786392ul});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view().column(0), expected);
}
