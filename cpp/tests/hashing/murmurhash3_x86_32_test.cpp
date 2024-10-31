/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/hashing.hpp>

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

class MurmurHashTest : public cudf::test::BaseFixture {};

TEST_F(MurmurHashTest, MultiValue)
{
  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "All work and no play makes Jack a dull boy",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)"});

  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, limits::min(), limits::max()});

  // Different truth values should be equal
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0});

  using ts = cudf::timestamp_s;
  cudf::test::fixed_width_column_wrapper<ts, ts::duration> const secs_col(
    {ts::duration::zero(),
     static_cast<ts::duration>(100),
     static_cast<ts::duration>(-100),
     ts::duration::min(),
     ts::duration::max()});

  auto const input1 = cudf::table_view({strings_col, ints_col, bools_col1, secs_col});
  auto const input2 = cudf::table_view({strings_col, ints_col, bools_col2, secs_col});

  auto const output1 = cudf::hashing::murmurhash3_x86_32(input1);
  auto const output2 = cudf::hashing::murmurhash3_x86_32(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TEST_F(MurmurHashTest, MultiValueNulls)
{
  // Nulls with different values should be equal
  cudf::test::strings_column_wrapper const strings_col1(
    {"",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "All work and no play makes Jack a dull boy",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)"},
    {false, true, true, false, true});
  cudf::test::strings_column_wrapper const strings_col2(
    {"different but null",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "I am Jack's complete lack of null value",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)"},
    {false, true, true, false, true});

  // Nulls with different values should be equal
  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col1(
    {0, 100, -100, limits::min(), limits::max()}, {true, false, false, true, true});
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col2(
    {0, -200, 200, limits::min(), limits::max()}, {true, false, false, true, true});

  // Nulls with different values should be equal
  // Different truth values should be equal
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1},
                                                                {true, true, false, false, true});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255},
                                                                {true, true, false, false, true});

  // Nulls with different values should be equal
  using ts = cudf::timestamp_s;
  cudf::test::fixed_width_column_wrapper<ts, ts::duration> const secs_col1(
    {ts::duration::zero(),
     static_cast<ts::duration>(100),
     static_cast<ts::duration>(-100),
     ts::duration::min(),
     ts::duration::max()},
    {true, false, false, true, true});
  cudf::test::fixed_width_column_wrapper<ts, ts::duration> const secs_col2(
    {ts::duration::zero(),
     static_cast<ts::duration>(-200),
     static_cast<ts::duration>(200),
     ts::duration::min(),
     ts::duration::max()},
    {true, false, false, true, true});

  auto const input1 = cudf::table_view({strings_col1, ints_col1, bools_col1, secs_col1});
  auto const input2 = cudf::table_view({strings_col2, ints_col2, bools_col2, secs_col2});

  auto const output1 = cudf::hashing::murmurhash3_x86_32(input1);
  auto const output2 = cudf::hashing::murmurhash3_x86_32(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TEST_F(MurmurHashTest, BasicList)
{
  using LCW = cudf::test::lists_column_wrapper<uint64_t>;
  using ICW = cudf::test::fixed_width_column_wrapper<uint32_t>;

  auto const col = LCW{{}, {}, {1}, {1, 1}, {1}, {1, 2}, {2, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}};
  auto const input  = cudf::table_view({col});
  auto const expect = ICW{1607593296,
                          1607593296,
                          -636010097,
                          -132459357,
                          -636010097,
                          -2008850957,
                          -1023787369,
                          761197503,
                          761197503,
                          1340177511,
                          -1023787369,
                          -1023787369};

  auto const output = cudf::hashing::murmurhash3_x86_32(input);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);

  auto const expect_seeded = ICW{1607594268u,
                                 1607594268u,
                                 1576790066u,
                                 1203671017u,
                                 1576790066u,
                                 2107478077u,
                                 1756855002u,
                                 2228938758u,
                                 2228938758u,
                                 3491134126u,
                                 1756855002u,
                                 1756855002u};

  auto const seeded_output = cudf::hashing::murmurhash3_x86_32(input, 15);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_seeded, seeded_output->view(), verbosity);
}

TEST_F(MurmurHashTest, NullableList)
{
  using LCW = cudf::test::lists_column_wrapper<uint64_t>;
  using ICW = cudf::test::fixed_width_column_wrapper<uint32_t>;

  auto const valids =
    std::vector<bool>{true, true, true, true, true, true, true, false, true, true, false};
  auto const col =
    LCW{{{}, {}, {1}, {1}, {2, 2}, {2}, {2}, {}, {2, 2}, {2, 2}, {}}, valids.begin()};
  auto expect = ICW{-2023148619,
                    -2023148619,
                    -31671896,
                    -31671896,
                    -1205248335,
                    1865773848,
                    1865773848,
                    -2023148682,
                    -1205248335,
                    -1205248335,
                    -2023148682};

  auto const output = cudf::hashing::murmurhash3_x86_32(cudf::table_view({col}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);

  auto const expect_seeded = ICW{2271820643u,
                                 2271820643u,
                                 1038318696u,
                                 1038318696u,
                                 595138041u,
                                 3027840870u,
                                 3027840870u,
                                 2271820578u,
                                 595138041u,
                                 595138041u,
                                 2271820578u};

  auto const seeded_output = cudf::hashing::murmurhash3_x86_32(cudf::table_view({col}), 31);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_seeded, seeded_output->view(), verbosity);
}

TEST_F(MurmurHashTest, ListOfStruct)
{
  auto col1 = cudf::test::fixed_width_column_wrapper<int32_t>{
    {-1, -1, 0, 2, 2, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 0, 1, 2},
    {true,
     true,
     true,
     true,
     true,
     false,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     false,
     false}};
  auto col2 = cudf::test::strings_column_wrapper{
    {"x", "x", "a", "a", "b", "b", "a", "b", "a", "b", "a", "c", "a", "c", "a", "c", "b", "b"},
    {true,
     true,
     true,
     true,
     true,
     false,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     false,
     false,
     true,
     true}};
  auto struct_col = cudf::test::structs_column_wrapper{{col1, col2},
                                                       {false,
                                                        false,
                                                        false,
                                                        false,
                                                        false,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true}};

  auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18};

  auto list_nullmask = std::vector<bool>{true,
                                         true,
                                         false,
                                         false,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::make_lists_column(
    17, offsets.release(), struct_col.release(), null_count, std::move(null_mask));

  auto expect = cudf::test::fixed_width_column_wrapper<uint32_t>{83451479,
                                                                 83451479,
                                                                 83455332,
                                                                 83455332,
                                                                 -759684425,
                                                                 -959632766,
                                                                 -959632766,
                                                                 -959632766,
                                                                 -959636527,
                                                                 -656998704,
                                                                 613652814,
                                                                 1902080426,
                                                                 1902080426,
                                                                 2061025592,
                                                                 2061025592,
                                                                 -319840811,
                                                                 -319840811};

  auto const output = cudf::hashing::murmurhash3_x86_32(cudf::table_view({*list_column}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);

  auto expect_seeded = cudf::test::fixed_width_column_wrapper<uint32_t>{81710442u,
                                                                        81710442u,
                                                                        81729816u,
                                                                        81729816u,
                                                                        3532787573u,
                                                                        3642097855u,
                                                                        3642097855u,
                                                                        3642097855u,
                                                                        3642110391u,
                                                                        3889855760u,
                                                                        1494406307u,
                                                                        103934081u,
                                                                        103934081u,
                                                                        3462063680u,
                                                                        3462063680u,
                                                                        1696730835u,
                                                                        1696730835u};

  auto const seeded_output =
    cudf::hashing::murmurhash3_x86_32(cudf::table_view({*list_column}), 619);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_seeded, seeded_output->view(), verbosity);
}

TEST_F(MurmurHashTest, ListOfEmptyStruct)
{
  // []
  // []
  // Null
  // Null
  // [Null, Null]
  // [Null, Null]
  // [Null, Null]
  // [Null]
  // [Null]
  // [{}]
  // [{}]
  // [{}, {}]
  // [{}, {}]

  auto struct_validity = std::vector<bool>{
    false, false, false, false, false, false, false, false, true, true, true, true, true, true};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(struct_validity.begin(), struct_validity.end());
  auto struct_col = cudf::make_structs_column(14, {}, null_count, std::move(null_mask));

  auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    0, 0, 0, 0, 0, 2, 4, 6, 7, 8, 9, 10, 12, 14};
  auto list_nullmask = std::vector<bool>{
    true, true, false, false, true, true, true, true, true, true, true, true, true};
  std::tie(null_mask, null_count) =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::make_lists_column(
    13, offsets.release(), std::move(struct_col), null_count, std::move(null_mask));

  auto expect = cudf::test::fixed_width_column_wrapper<uint32_t>{2271818677u,
                                                                 2271818677u,
                                                                 2271818614u,
                                                                 2271818614u,
                                                                 3954409013u,
                                                                 3954409013u,
                                                                 3954409013u,
                                                                 2295666275u,
                                                                 2295666275u,
                                                                 2295666276u,
                                                                 2295666276u,
                                                                 3954409052u,
                                                                 3954409052u};

  auto output = cudf::hashing::murmurhash3_x86_32(cudf::table_view({*list_column}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);
}

TEST_F(MurmurHashTest, EmptyDeepList)
{
  // List<List<int>>, where all lists are empty
  // []
  // []
  // Null
  // Null

  // Internal empty list
  auto list1 = cudf::test::lists_column_wrapper<int>{};

  auto offsets       = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 0, 0, 0};
  auto list_nullmask = std::vector<bool>{true, true, false, false};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::make_lists_column(
    4, offsets.release(), list1.release(), null_count, std::move(null_mask));

  auto expect = cudf::test::fixed_width_column_wrapper<uint32_t>{
    2271818677u, 2271818677u, 2271818614u, 2271818614u};

  auto output = cudf::hashing::murmurhash3_x86_32(cudf::table_view({*list_column}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);
}

template <typename T>
class MurmurHashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(MurmurHashTestTyped, cudf::test::FixedWidthTypes);

TYPED_TEST(MurmurHashTestTyped, Equality)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> const col{0, 127, 1, 2, 8};
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hashing::murmurhash3_x86_32(input);
  auto const output2 = cudf::hashing::murmurhash3_x86_32(input);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(MurmurHashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  cudf::test::fixed_width_column_wrapper<T, int32_t> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<T, int32_t> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hashing::murmurhash3_x86_32(input1);
  auto const output2 = cudf::hashing::murmurhash3_x86_32(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class MurmurHashTestFloatTyped : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(MurmurHashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(MurmurHashTestFloatTyped, TestExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  cudf::test::fixed_width_column_wrapper<T> const col(
    {T(0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  cudf::test::fixed_width_column_wrapper<T> const col_neg_zero(
    {T(-0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  cudf::test::fixed_width_column_wrapper<T> const col_neg_nan(
    {T(0.0), T(100.0), T(-100.0), min, max, -nan, inf, -inf});

  auto const table_col          = cudf::table_view({col});
  auto const table_col_neg_zero = cudf::table_view({col_neg_zero});
  auto const table_col_neg_nan  = cudf::table_view({col_neg_nan});

  auto const hash_col          = cudf::hashing::murmurhash3_x86_32(table_col);
  auto const hash_col_neg_zero = cudf::hashing::murmurhash3_x86_32(table_col_neg_zero);
  auto const hash_col_neg_nan  = cudf::hashing::murmurhash3_x86_32(table_col_neg_nan);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_col, *hash_col_neg_zero, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_col, *hash_col_neg_nan, verbosity);
}

CUDF_TEST_PROGRAM_MAIN()
