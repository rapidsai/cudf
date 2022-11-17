/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

class HashTest : public cudf::test::BaseFixture {
};

TEST_F(HashTest, MultiValue)
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

  auto const output1 = cudf::hash(input1);
  auto const output2 = cudf::hash(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TEST_F(HashTest, MultiValueNulls)
{
  // Nulls with different values should be equal
  cudf::test::strings_column_wrapper const strings_col1(
    {"",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "All work and no play makes Jack a dull boy",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)"},
    {0, 1, 1, 0, 1});
  cudf::test::strings_column_wrapper const strings_col2(
    {"different but null",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "I am Jack's complete lack of null value",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)"},
    {0, 1, 1, 0, 1});

  // Nulls with different values should be equal
  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col1(
    {0, 100, -100, limits::min(), limits::max()}, {1, 0, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col2(
    {0, -200, 200, limits::min(), limits::max()}, {1, 0, 0, 1, 1});

  // Nulls with different values should be equal
  // Different truth values should be equal
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {1, 1, 0, 0, 1});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {1, 1, 0, 0, 1});

  // Nulls with different values should be equal
  using ts = cudf::timestamp_s;
  cudf::test::fixed_width_column_wrapper<ts, ts::duration> const secs_col1(
    {ts::duration::zero(),
     static_cast<ts::duration>(100),
     static_cast<ts::duration>(-100),
     ts::duration::min(),
     ts::duration::max()},
    {1, 0, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<ts, ts::duration> const secs_col2(
    {ts::duration::zero(),
     static_cast<ts::duration>(-200),
     static_cast<ts::duration>(200),
     ts::duration::min(),
     ts::duration::max()},
    {1, 0, 0, 1, 1});

  auto const input1 = cudf::table_view({strings_col1, ints_col1, bools_col1, secs_col1});
  auto const input2 = cudf::table_view({strings_col2, ints_col2, bools_col2, secs_col2});

  auto const output1 = cudf::hash(input1);
  auto const output2 = cudf::hash(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());

  auto const spark_output1 = cudf::hash(input1, cudf::hash_id::HASH_SPARK_MURMUR3, 0);
  auto const spark_output2 = cudf::hash(input2, cudf::hash_id::HASH_SPARK_MURMUR3);

  EXPECT_EQ(input1.num_rows(), spark_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(spark_output1->view(), spark_output2->view());
}

TEST_F(HashTest, BasicList)
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

  auto const output = cudf::hash(input);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);

  auto const expect_seeded = ICW{1607594268u,
                                 1607594268u,
                                 3658958173u,
                                 4162508905u,
                                 3658958173u,
                                 2286117305u,
                                 3271180885u,
                                 761198477u,
                                 761198477u,
                                 1340178469u,
                                 3271180885u,
                                 3271180885u};

  auto const seeded_output = cudf::hash(input, cudf::hash_id::HASH_MURMUR3, 15);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_seeded, seeded_output->view(), verbosity);
}

TEST_F(HashTest, NullableList)
{
  using LCW = cudf::test::lists_column_wrapper<uint64_t>;
  using ICW = cudf::test::fixed_width_column_wrapper<uint32_t>;

  auto const valids = std::vector<bool>{1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0};
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

  auto const output = cudf::hash(cudf::table_view({col}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);

  auto const expect_seeded = ICW{2271820643u,
                                 2271820643u,
                                 4263297392u,
                                 4263297392u,
                                 3089720935u,
                                 1865775808u,
                                 1865775808u,
                                 2271820578u,
                                 3089720935u,
                                 3089720935u,
                                 2271820578u};

  auto const seeded_output = cudf::hash(cudf::table_view({col}), cudf::hash_id::HASH_MURMUR3, 31);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_seeded, seeded_output->view(), verbosity);
}

TEST_F(HashTest, ListOfStruct)
{
  auto col1 = cudf::test::fixed_width_column_wrapper<int32_t>{
    {-1, -1, 0, 2, 2, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 0, 1, 2},
    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}};
  auto col2 = cudf::test::strings_column_wrapper{
    {"x", "x", "a", "a", "b", "b", "a", "b", "a", "b", "a", "c", "a", "c", "a", "c", "b", "b"},
    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1}};
  auto struct_col = cudf::test::structs_column_wrapper{
    {col1, col2}, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18};

  auto list_nullmask = std::vector<bool>{1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  auto nullmask_buf =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::make_lists_column(
    17, offsets.release(), struct_col.release(), cudf::UNKNOWN_NULL_COUNT, std::move(nullmask_buf));

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

  auto const output = cudf::hash(cudf::table_view({*list_column}));
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
                                                                        3624905718u,
                                                                        608933631u,
                                                                        1899376347u,
                                                                        1899376347u,
                                                                        2058877614u,
                                                                        2058877614u,
                                                                        4013395891u,
                                                                        4013395891u};

  auto const seeded_output =
    cudf::hash(cudf::table_view({*list_column}), cudf::hash_id::HASH_MURMUR3, 619);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_seeded, seeded_output->view(), verbosity);
}

TEST_F(HashTest, ListOfEmptyStruct)
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

  auto struct_validity = std::vector<bool>{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto struct_validity_buffer =
    cudf::test::detail::make_null_mask(struct_validity.begin(), struct_validity.end());
  auto struct_col =
    cudf::make_structs_column(14, {}, cudf::UNKNOWN_NULL_COUNT, std::move(struct_validity_buffer));

  auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    0, 0, 0, 0, 0, 2, 4, 6, 7, 8, 9, 10, 12, 14};
  auto list_nullmask = std::vector<bool>{1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  auto list_validity_buffer =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::make_lists_column(13,
                                             offsets.release(),
                                             std::move(struct_col),
                                             cudf::UNKNOWN_NULL_COUNT,
                                             std::move(list_validity_buffer));

  auto expect = cudf::test::fixed_width_column_wrapper<uint32_t>{-2023148619,
                                                                 -2023148619,
                                                                 -2023148682,
                                                                 -2023148682,
                                                                 -340558283,
                                                                 -340558283,
                                                                 -340558283,
                                                                 -1999301021,
                                                                 -1999301021,
                                                                 -1999301020,
                                                                 -1999301020,
                                                                 -340558244,
                                                                 -340558244};

  auto output = cudf::hash(cudf::table_view({*list_column}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);
}

TEST_F(HashTest, EmptyDeepList)
{
  // List<List<int>>, where all lists are empty
  // []
  // []
  // Null
  // Null

  // Internal empty list
  auto list1 = cudf::test::lists_column_wrapper<int>{};

  auto offsets       = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 0, 0, 0};
  auto list_nullmask = std::vector<bool>{1, 1, 0, 0};
  auto list_validity_buffer =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::make_lists_column(4,
                                             offsets.release(),
                                             list1.release(),
                                             cudf::UNKNOWN_NULL_COUNT,
                                             std::move(list_validity_buffer));

  auto expect = cudf::test::fixed_width_column_wrapper<uint32_t>{
    -2023148619, -2023148619, -2023148682, -2023148682};

  auto output = cudf::hash(cudf::table_view({*list_column}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);
}

template <typename T>
class HashTestTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(HashTestTyped, cudf::test::FixedWidthTypes);

TYPED_TEST(HashTestTyped, Equality)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> const col{0, 127, 1, 2, 8};
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hash(input);
  auto const output2 = cudf::hash(input);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());

  auto const spark_output1 = cudf::hash(input, cudf::hash_id::HASH_SPARK_MURMUR3, 0);
  auto const spark_output2 = cudf::hash(input, cudf::hash_id::HASH_SPARK_MURMUR3);

  EXPECT_EQ(input.num_rows(), spark_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(spark_output1->view(), spark_output2->view());
}

TYPED_TEST(HashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  cudf::test::fixed_width_column_wrapper<T, int32_t> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<T, int32_t> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1);
  auto const output2 = cudf::hash(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());

  auto const spark_output1 = cudf::hash(input1, cudf::hash_id::HASH_SPARK_MURMUR3, 0);
  auto const spark_output2 = cudf::hash(input2, cudf::hash_id::HASH_SPARK_MURMUR3);

  EXPECT_EQ(input1.num_rows(), spark_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(spark_output1->view(), spark_output2->view());
}

template <typename T>
class HashTestFloatTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(HashTestFloatTyped, TestExtremes)
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

  auto const hash_col          = cudf::hash(table_col);
  auto const hash_col_neg_zero = cudf::hash(table_col_neg_zero);
  auto const hash_col_neg_nan  = cudf::hash(table_col_neg_nan);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_col, *hash_col_neg_zero, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_col, *hash_col_neg_nan, verbosity);

  // Spark hash is sensitive to 0 and -0
  constexpr auto spark_hasher  = cudf::hash_id::HASH_SPARK_MURMUR3;
  auto const spark_col         = cudf::hash(table_col, spark_hasher, 0);
  auto const spark_col_neg_nan = cudf::hash(table_col_neg_nan, spark_hasher);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*spark_col, *spark_col_neg_nan);
}

class SparkMurmurHash3Test : public cudf::test::BaseFixture {
};

TEST_F(SparkMurmurHash3Test, MultiValueWithSeeds)
{
  // The hash values were determined by running the following Scala code in Apache Spark.
  // Note that Spark >= 3.2 normalizes the float/double value of -0. to +0. and both values hash
  // to the same result. This is normalized in the calling code (Spark RAPIDS plugin) for Spark
  // >= 3.2. However, the reference values for -0. below must be obtained with Spark < 3.2 and
  // libcudf will continue to implement the Spark < 3.2 behavior until Spark >= 3.2 is required and
  // the workaround in the calling code is removed. This also affects the combined hash values.

  /*
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types._
  import org.apache.spark.sql.Row
  import org.apache.spark.sql.catalyst.util.DateTimeUtils

  val schema = new StructType()
      .add("structs", new StructType()
          .add("a", IntegerType)
          .add("b", StringType)
          .add("c", new StructType()
              .add("x", FloatType)
              .add("y", LongType)))
      .add("strings", StringType)
      .add("doubles", DoubleType)
      .add("timestamps", TimestampType)
      .add("decimal64", DecimalType(18, 7))
      .add("longs", LongType)
      .add("floats", FloatType)
      .add("dates", DateType)
      .add("decimal32", DecimalType(9, 3))
      .add("ints", IntegerType)
      .add("shorts", ShortType)
      .add("bytes", ByteType)
      .add("bools", BooleanType)
      .add("decimal128", DecimalType(38, 11))

  val data = Seq(
      Row(Row(0, "a", Row(0f, 0L)), "", 0.toDouble,
          DateTimeUtils.toJavaTimestamp(0), BigDecimal(0), 0.toLong, 0.toFloat,
          DateTimeUtils.toJavaDate(0), BigDecimal(0), 0, 0.toShort, 0.toByte,
          false, BigDecimal(0)),
      Row(Row(100, "bc", Row(100f, 100L)), "The quick brown fox", -(0.toDouble),
          DateTimeUtils.toJavaTimestamp(100), BigDecimal("0.00001"), 100.toLong, -(0.toFloat),
          DateTimeUtils.toJavaDate(100), BigDecimal("0.1"), 100, 100.toShort, 100.toByte,
          true, BigDecimal("0.000000001")),
      Row(Row(-100, "def", Row(-100f, -100L)), "jumps over the lazy dog.", -Double.NaN,
          DateTimeUtils.toJavaTimestamp(-100), BigDecimal("-0.00001"), -100.toLong, -Float.NaN,
          DateTimeUtils.toJavaDate(-100), BigDecimal("-0.1"), -100, -100.toShort, -100.toByte,
          true, BigDecimal("-0.00000000001")),
      Row(Row(0x12345678, "ghij", Row(Float.PositiveInfinity, 0x123456789abcdefL)),
          "All work and no play makes Jack a dull boy", Double.MinValue,
          DateTimeUtils.toJavaTimestamp(Long.MinValue/1000000), BigDecimal("-99999999999.9999999"),
          Long.MinValue, Float.MinValue, DateTimeUtils.toJavaDate(Int.MinValue/100),
          BigDecimal("-999999.999"), Int.MinValue, Short.MinValue, Byte.MinValue, true,
          BigDecimal("-9999999999999999.99999999999")),
      Row(Row(-0x76543210, "klmno", Row(Float.NegativeInfinity, -0x123456789abcdefL)),
          "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721", Double.MaxValue,
          DateTimeUtils.toJavaTimestamp(Long.MaxValue/1000000), BigDecimal("99999999999.9999999"),
          Long.MaxValue, Float.MaxValue, DateTimeUtils.toJavaDate(Int.MaxValue/100),
          BigDecimal("999999.999"), Int.MaxValue, Short.MaxValue, Byte.MaxValue, false,
          BigDecimal("99999999999999999999999999.99999999999")))

  val df = spark.createDataFrame(sc.parallelize(data), schema)
  df.columns.foreach(c => println(s"$c => ${df.select(hash(col(c))).collect.mkString(",")}"))
  println(s"combined => ${df.select(hash(col("*"))).collect.mkString(",")}")
  */

  cudf::test::fixed_width_column_wrapper<int32_t> const hash_structs_expected(
    {-105406170, 90479889, -678041645, 1667387937, 301478567});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_strings_expected(
    {142593372, 1217302703, -715697185, -2061143941, -111635966});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_doubles_expected(
    {-1670924195, -853646085, -1281358385, 1897734433, -508695674});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_timestamps_expected(
    {-1670924195, 1114849490, 904948192, -1832979433, 1752430209});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_decimal64_expected(
    {-1670924195, 1114849490, 904948192, 1962370902, -1795328666});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_longs_expected(
    {-1670924195, 1114849490, 904948192, -853646085, -1604625029});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_floats_expected(
    {933211791, 723455942, -349261430, -1225560532, -338752985});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_dates_expected(
    {933211791, 751823303, -1080202046, -1906567553, -1503850410});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_decimal32_expected(
    {-1670924195, 1114849490, 904948192, -1454351396, -193774131});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_ints_expected(
    {933211791, 751823303, -1080202046, 723455942, 133916647});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_shorts_expected(
    {933211791, 751823303, -1080202046, -1871935946, 1249274084});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_bytes_expected(
    {933211791, 751823303, -1080202046, 1110053733, 1135925485});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_bools_expected(
    {933211791, -559580957, -559580957, -559580957, 933211791});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_decimal128_expected(
    {-783713497, -295670906, 1398487324, -52622807, -1359749815});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_combined_expected(
    {401603227, 588162166, 552160517, 1132537411, -326043017});

  using double_limits = std::numeric_limits<double>;
  using long_limits   = std::numeric_limits<int64_t>;
  using float_limits  = std::numeric_limits<float>;
  using int_limits    = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> a_col{0, 100, -100, 0x1234'5678, -0x7654'3210};
  cudf::test::strings_column_wrapper b_col{"a", "bc", "def", "ghij", "klmno"};
  cudf::test::fixed_width_column_wrapper<float> x_col{
    0.f, 100.f, -100.f, float_limits::infinity(), -float_limits::infinity()};
  cudf::test::fixed_width_column_wrapper<int64_t> y_col{
    0L, 100L, -100L, 0x0123'4567'89ab'cdefL, -0x0123'4567'89ab'cdefL};
  cudf::test::structs_column_wrapper c_col{{x_col, y_col}};
  cudf::test::structs_column_wrapper const structs_col{{a_col, b_col, c_col}};

  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721"});
  cudf::test::fixed_width_column_wrapper<double> const doubles_col(
    {0., -0., -double_limits::quiet_NaN(), double_limits::lowest(), double_limits::max()});
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep> const
    timestamps_col({0L, 100L, -100L, long_limits::min() / 1000000, long_limits::max() / 1000000});
  cudf::test::fixed_point_column_wrapper<int64_t> const decimal64_col(
    {0L, 100L, -100L, -999999999999999999L, 999999999999999999L}, numeric::scale_type{-7});
  cudf::test::fixed_width_column_wrapper<int64_t> const longs_col(
    {0L, 100L, -100L, long_limits::min(), long_limits::max()});
  cudf::test::fixed_width_column_wrapper<float> const floats_col(
    {0.f, -0.f, -float_limits::quiet_NaN(), float_limits::lowest(), float_limits::max()});
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> dates_col(
    {0, 100, -100, int_limits::min() / 100, int_limits::max() / 100});
  cudf::test::fixed_point_column_wrapper<int32_t> const decimal32_col(
    {0, 100, -100, -999999999, 999999999}, numeric::scale_type{-3});
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, int_limits::min(), int_limits::max()});
  cudf::test::fixed_width_column_wrapper<int16_t> const shorts_col({0, 100, -100, -32768, 32767});
  cudf::test::fixed_width_column_wrapper<int8_t> const bytes_col({0, 100, -100, -128, 127});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0});
  cudf::test::fixed_point_column_wrapper<__int128_t> const decimal128_col(
    {static_cast<__int128>(0),
     static_cast<__int128>(100),
     static_cast<__int128>(-1),
     (static_cast<__int128>(0xFFFF'FFFF'FCC4'D1C3u) << 64 | 0x602F'7FC3'1800'0001u),
     (static_cast<__int128>(0x0785'EE10'D5DA'46D9u) << 64 | 0x00F4'369F'FFFF'FFFFu)},
    numeric::scale_type{-11});

  constexpr auto hasher      = cudf::hash_id::HASH_SPARK_MURMUR3;
  auto const hash_structs    = cudf::hash(cudf::table_view({structs_col}), hasher, 42);
  auto const hash_strings    = cudf::hash(cudf::table_view({strings_col}), hasher, 42);
  auto const hash_doubles    = cudf::hash(cudf::table_view({doubles_col}), hasher, 42);
  auto const hash_timestamps = cudf::hash(cudf::table_view({timestamps_col}), hasher, 42);
  auto const hash_decimal64  = cudf::hash(cudf::table_view({decimal64_col}), hasher, 42);
  auto const hash_longs      = cudf::hash(cudf::table_view({longs_col}), hasher, 42);
  auto const hash_floats     = cudf::hash(cudf::table_view({floats_col}), hasher, 42);
  auto const hash_dates      = cudf::hash(cudf::table_view({dates_col}), hasher, 42);
  auto const hash_decimal32  = cudf::hash(cudf::table_view({decimal32_col}), hasher, 42);
  auto const hash_ints       = cudf::hash(cudf::table_view({ints_col}), hasher, 42);
  auto const hash_shorts     = cudf::hash(cudf::table_view({shorts_col}), hasher, 42);
  auto const hash_bytes      = cudf::hash(cudf::table_view({bytes_col}), hasher, 42);
  auto const hash_bools1     = cudf::hash(cudf::table_view({bools_col1}), hasher, 42);
  auto const hash_bools2     = cudf::hash(cudf::table_view({bools_col2}), hasher, 42);
  auto const hash_decimal128 = cudf::hash(cudf::table_view({decimal128_col}), hasher, 42);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_structs, hash_structs_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_strings, hash_strings_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_doubles, hash_doubles_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_timestamps, hash_timestamps_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_decimal64, hash_decimal64_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_longs, hash_longs_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_floats, hash_floats_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_dates, hash_dates_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_decimal32, hash_decimal32_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_ints, hash_ints_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_shorts, hash_shorts_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_bytes, hash_bytes_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_bools1, hash_bools_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_bools2, hash_bools_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_decimal128, hash_decimal128_expected, verbosity);

  auto const combined_table = cudf::table_view({structs_col,
                                                strings_col,
                                                doubles_col,
                                                timestamps_col,
                                                decimal64_col,
                                                longs_col,
                                                floats_col,
                                                dates_col,
                                                decimal32_col,
                                                ints_col,
                                                shorts_col,
                                                bytes_col,
                                                bools_col2,
                                                decimal128_col});
  auto const hash_combined  = cudf::hash(combined_table, hasher, 42);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_combined, hash_combined_expected, verbosity);
}

TEST_F(SparkMurmurHash3Test, StringsWithSeed)
{
  // The hash values were determined by running the following Scala code in Apache Spark:
  // val strs = Seq("", "The quick brown fox",
  //              "jumps over the lazy dog.",
  //              "All work and no play makes Jack a dull boy",
  //              "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721")
  // println(strs.map(org.apache.spark.unsafe.types.UTF8String.fromString)
  //   .map(org.apache.spark.sql.catalyst.expressions.Murmur3HashFunction.hash(
  //     _, org.apache.spark.sql.types.StringType, 314)))

  cudf::test::fixed_width_column_wrapper<int32_t> const hash_strings_expected_seed_314(
    {1467149710, 723257560, -1620282500, -2001858707, 1588473657});

  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721"});

  constexpr auto hasher   = cudf::hash_id::HASH_SPARK_MURMUR3;
  auto const hash_strings = cudf::hash(cudf::table_view({strings_col}), hasher, 314);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_strings, hash_strings_expected_seed_314, verbosity);
}

TEST_F(SparkMurmurHash3Test, ListValues)
{
  /*
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types.{ArrayType, IntegerType, StructType}
  import org.apache.spark.sql.Row

  val schema = new StructType()
    .add("lists",ArrayType(ArrayType(IntegerType)))

  val data = Seq(
    Row(null),
    Row(List(null)),
    Row(List(List())),
    Row(List(List(1))),
    Row(List(List(1, 2))),
    Row(List(List(1, 2, 3))),
    Row(List(List(1, 2), List(3))),
    Row(List(List(1), List(2, 3))),
    Row(List(List(1), List(null, 2, 3))),
    Row(List(List(1, 2), List(3), List(null))),
    Row(List(List(1, 2), null, List(3))),
  )

  val df = spark.createDataFrame(
    spark.sparkContext.parallelize(data), schema)

  val df2 = df.selectExpr("lists", "hash(lists) as hash")
  df2.printSchema()
  df2.show(false)
  */

  auto const null = -1;
  auto nested_list =
    cudf::test::lists_column_wrapper<int>({{},
                                           {1},
                                           {1, 2},
                                           {1, 2, 3},
                                           {1, 2},
                                           {3},
                                           {1},
                                           {2, 3},
                                           {1},
                                           {{null, 2, 3}, cudf::test::iterators::nulls_at({0})},
                                           {1, 2},
                                           {3},
                                           {{null}, cudf::test::iterators::nulls_at({0})},
                                           {1, 2},
                                           {},
                                           {3}},
                                          cudf::test::iterators::nulls_at({0, 14}));
  auto offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 0, 1, 2, 3, 4, 6, 8, 10, 13, 16};
  auto list_validity        = cudf::test::iterators::nulls_at({0});
  auto list_validity_buffer = cudf::test::detail::make_null_mask(list_validity, list_validity + 11);
  auto list_column          = cudf::make_lists_column(11,
                                             offsets.release(),
                                             nested_list.release(),
                                             cudf::UNKNOWN_NULL_COUNT,
                                             std::move(list_validity_buffer));

  auto expect = cudf::test::fixed_width_column_wrapper<int32_t>{42,
                                                                42,
                                                                42,
                                                                -559580957,
                                                                -222940379,
                                                                -912918097,
                                                                -912918097,
                                                                -912918097,
                                                                -912918097,
                                                                -912918097,
                                                                -912918097};

  auto output = cudf::hash(cudf::table_view({*list_column}), cudf::hash_id::HASH_SPARK_MURMUR3, 42);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);
}

TEST_F(SparkMurmurHash3Test, StructOfListValues)
{
  /*
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types.{ArrayType, IntegerType, StructType}
  import org.apache.spark.sql.Row

  val schema = new StructType()
    .add("structs", new StructType()
        .add("a", ArrayType(IntegerType))
        .add("b", ArrayType(IntegerType)))

  val data = Seq(
    Row(Row(List(), List())),
    Row(Row(List(0), List(0))),
    Row(Row(List(1, null), null)),
    Row(Row(List(1, null), List())),
    Row(Row(List(), List(null, 1))),
    Row(Row(null, List(1))),
    Row(Row(List(2, 3), List(4, 5))),
  )

  val df = spark.createDataFrame(
    spark.sparkContext.parallelize(data), schema)

  val df2 = df.selectExpr("lists", "hash(lists) as hash")
  df2.printSchema()
  df2.show(false)
  */

  auto const null = -1;
  auto col1 =
    cudf::test::lists_column_wrapper<int>({{},
                                           {0},
                                           {{1, null}, cudf::test::iterators::nulls_at({1})},
                                           {{1, null}, cudf::test::iterators::nulls_at({1})},
                                           {},
                                           {} /*NULL*/,
                                           {2, 3}},
                                          cudf::test::iterators::nulls_at({5}));
  auto col2 = cudf::test::lists_column_wrapper<int>(
    {{}, {0}, {} /*NULL*/, {}, {{null, 1}, cudf::test::iterators::nulls_at({0})}, {1}, {4, 5}},
    cudf::test::iterators::nulls_at({2}));
  auto struct_column = cudf::test::structs_column_wrapper{{col1, col2}};

  auto expect = cudf::test::fixed_width_column_wrapper<int32_t>{
    42, 59727262, -559580957, -559580957, -559580957, -559580957, 170038658};

  auto output =
    cudf::hash(cudf::table_view({struct_column}), cudf::hash_id::HASH_SPARK_MURMUR3, 42);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);
}

TEST_F(SparkMurmurHash3Test, ListOfStructValues)
{
  /*
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types.{ArrayType, IntegerType, StructType}
  import org.apache.spark.sql.Row

  val schema = new StructType()
    .add("lists", ArrayType(new StructType()
      .add("a", IntegerType)
      .add("b", IntegerType)))

  val data = Seq(
    Row(List(Row(0, 0))),
    Row(List(null)),
    Row(List(Row(null, null))),
    Row(List(Row(1, null))),
    Row(List(Row(null, 1))),
    Row(List(Row(null, 1), Row(2, 3))),
    Row(List(Row(2, 3), null)),
    Row(List(Row(2, 3), Row(4, 5))),
  )

  val df = spark.createDataFrame(
    spark.sparkContext.parallelize(data), schema)

  val df2 = df.selectExpr("lists", "hash(lists) as hash")
  df2.printSchema()
  df2.show(false)
  */

  auto const null = -1;
  auto col1       = cudf::test::fixed_width_column_wrapper<int32_t>(
    {0, null, null, 1, null, null, 2, 2, null, 2, 4},
    cudf::test::iterators::nulls_at({1, 2, 4, 5, 8}));
  auto col2 = cudf::test::fixed_width_column_wrapper<int32_t>(
    {0, null, null, null, 1, 1, 3, 3, null, 3, 5}, cudf::test::iterators::nulls_at({1, 2, 3, 8}));
  auto struct_column =
    cudf::test::structs_column_wrapper{{col1, col2}, {1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1}};
  auto offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2, 3, 4, 5, 7, 9, 11};
  auto list_nullmask = std::vector<bool>(1, 8);
  auto list_validity_buffer =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::make_lists_column(8,
                                             offsets.release(),
                                             struct_column.release(),
                                             cudf::UNKNOWN_NULL_COUNT,
                                             std::move(list_validity_buffer));

  // TODO: Lists of structs are not yet supported. Once support is added,
  // remove this EXPECT_THROW and uncomment the rest of this test.
  EXPECT_THROW(cudf::hash(cudf::table_view({*list_column}), cudf::hash_id::HASH_SPARK_MURMUR3, 42),
               cudf::logic_error);

  /*
  auto expect = cudf::test::fixed_width_column_wrapper<int32_t>{
    59727262, 42, 42, -559580957, -559580957, -912918097, 1092624418, 170038658};

  auto output = cudf::hash(cudf::table_view({*list_column}), cudf::hash_id::HASH_SPARK_MURMUR3, 42);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);
  */
}

class MD5HashTest : public cudf::test::BaseFixture {
};

TEST_F(MD5HashTest, MultiValue)
{
  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "A 60 character string to test MD5's message padding algorithm",
     "A very long (greater than 128 bytes/char string) to test a multi hash-step data point in the "
     "MD5 hash function. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)"});

  cudf::test::strings_column_wrapper const md5_string_results1(
    {"d41d8cd98f00b204e9800998ecf8427e",
     "682240021651ae166d08fe2a014d5c09",
     "3669d5225fddbb34676312ca3b78bbd9",
     "c61a4185135eda043f35e92c3505e180",
     "52da74c75cb6575d25be29e66bd0adde"});

  cudf::test::strings_column_wrapper const md5_string_results2(
    {"d41d8cd98f00b204e9800998ecf8427e",
     "e5a5682e82278e78dbaad9a689df7a73",
     "4121ab1bb6e84172fd94822645862ae9",
     "28970886501efe20164213855afe5850",
     "6bc1b872103cc6a02d882245b8516e2e"});

  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, limits::min(), limits::max()});

  // Different truth values should be equal
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0});

  auto const string_input1      = cudf::table_view({strings_col});
  auto const string_input2      = cudf::table_view({strings_col, strings_col});
  auto const md5_string_output1 = cudf::hash(string_input1, cudf::hash_id::HASH_MD5);
  auto const md5_string_output2 = cudf::hash(string_input2, cudf::hash_id::HASH_MD5);
  EXPECT_EQ(string_input1.num_rows(), md5_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), md5_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(md5_string_output1->view(), md5_string_results1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(md5_string_output2->view(), md5_string_results2);

  auto const input1      = cudf::table_view({strings_col, ints_col, bools_col1});
  auto const input2      = cudf::table_view({strings_col, ints_col, bools_col2});
  auto const md5_output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const md5_output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);
  EXPECT_EQ(input1.num_rows(), md5_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(md5_output1->view(), md5_output2->view());
}

TEST_F(MD5HashTest, MultiValueNulls)
{
  // Nulls with different values should be equal
  cudf::test::strings_column_wrapper const strings_col1(
    {"",
     "Different but null!",
     "A very long (greater than 128 bytes/char string) to test a multi hash-step data point in the "
     "MD5 hash function. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)"},
    {1, 0, 0, 1, 0});
  cudf::test::strings_column_wrapper const strings_col2(
    {"",
     "A 60 character string to test MD5's message padding algorithm",
     "Very different... but null",
     "All work and no play makes Jack a dull boy",
     ""},
    {1, 0, 0, 1, 1});  // empty string is equivalent to null

  // Nulls with different values should be equal
  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col1(
    {0, 100, -100, limits::min(), limits::max()}, {1, 0, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col2(
    {0, -200, 200, limits::min(), limits::max()}, {1, 0, 0, 1, 1});

  // Nulls with different values should be equal
  // Different truth values should be equal
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {1, 1, 0, 0, 1});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {1, 1, 0, 0, 1});

  auto const input1 = cudf::table_view({strings_col1, ints_col1, bools_col1});
  auto const input2 = cudf::table_view({strings_col2, ints_col2, bools_col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TEST_F(MD5HashTest, StringListsNulls)
{
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 0; });

  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "A 60 character string to test MD5's message padding algorithm",
     "A very long (greater than 128 bytes/char string) to test a multi hash-step data point in the "
     "MD5 hash function. This string needed to be longer. It needed to be even longer.",
     "All work and no play makes Jack a dull boy",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)"});

  cudf::test::lists_column_wrapper<cudf::string_view> strings_list_col(
    {{""},
     {{"NULL", "A 60 character string to test MD5's message padding algorithm"}, validity},
     {"A very long (greater than 128 bytes/char string) to test a multi hash-step data point in "
      "the "
      "MD5 hash function. This string needed to be longer.",
      " It needed to be even longer."},
     {"All ", "work ", "and", " no", " play ", "makes Jack", " a dull boy"},
     {"!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`", "{|}~"}});

  auto const input1 = cudf::table_view({strings_col});
  auto const input2 = cudf::table_view({strings_list_col});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class MD5HashTestTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(MD5HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(MD5HashTestTyped, Equality)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hash(input, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input, cudf::hash_id::HASH_MD5);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(MD5HashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  cudf::test::fixed_width_column_wrapper<T> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<T> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TEST_F(MD5HashTest, TestBoolListsWithNulls)
{
  cudf::test::fixed_width_column_wrapper<bool> const col1({0, 255, 255, 16, 27, 18, 100, 1, 2},
                                                          {1, 0, 0, 0, 1, 1, 1, 0, 0});
  cudf::test::fixed_width_column_wrapper<bool> const col2({0, 255, 255, 32, 81, 68, 3, 101, 4},
                                                          {1, 0, 0, 1, 0, 1, 0, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> const col3({0, 255, 255, 64, 49, 42, 5, 6, 102},
                                                          {1, 0, 0, 1, 1, 0, 0, 0, 1});

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; });
  cudf::test::lists_column_wrapper<bool> const list_col(
    {{0, 0, 0}, {1}, {}, {{1, 1, 1}, validity}, {1, 1}, {1, 1}, {1}, {1}, {1}}, validity);

  auto const input1 = cudf::table_view({col1, col2, col3});
  auto const input2 = cudf::table_view({list_col});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class MD5HashListTestTyped : public cudf::test::BaseFixture {
};

using NumericTypesNoBools =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(MD5HashListTestTyped, NumericTypesNoBools);

TYPED_TEST(MD5HashListTestTyped, TestListsWithNulls)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> const col1({0, 255, 255, 16, 27, 18, 100, 1, 2},
                                                       {1, 0, 0, 0, 1, 1, 1, 0, 0});
  cudf::test::fixed_width_column_wrapper<T> const col2({0, 255, 255, 32, 81, 68, 3, 101, 4},
                                                       {1, 0, 0, 1, 0, 1, 0, 1, 0});
  cudf::test::fixed_width_column_wrapper<T> const col3({0, 255, 255, 64, 49, 42, 5, 6, 102},
                                                       {1, 0, 0, 1, 1, 0, 0, 0, 1});

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 1; });
  cudf::test::lists_column_wrapper<T> const list_col(
    {{0, 0, 0}, {127}, {}, {{32, 127, 64}, validity}, {27, 49}, {18, 68}, {100}, {101}, {102}},
    validity);

  auto const input1 = cudf::table_view({col1, col2, col3});
  auto const input2 = cudf::table_view({list_col});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class MD5HashTestFloatTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(MD5HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(MD5HashTestFloatTyped, TestExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  cudf::test::fixed_width_column_wrapper<T> const col1(
    {T(0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  cudf::test::fixed_width_column_wrapper<T> const col2(
    {T(-0.0), T(100.0), T(-100.0), min, max, -nan, inf, -inf});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view(), verbosity);
}

TYPED_TEST(MD5HashTestFloatTyped, TestListExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  cudf::test::lists_column_wrapper<T> const col1(
    {{T(0.0)}, {T(100.0), T(-100.0)}, {min, max, nan}, {inf, -inf}});
  cudf::test::lists_column_wrapper<T> const col2(
    {{T(-0.0)}, {T(100.0), T(-100.0)}, {min, max, -nan}, {inf, -inf}});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view(), verbosity);
}

class SHA1HashTest : public cudf::test::BaseFixture {
};

TEST_F(SHA1HashTest, EmptyTable)
{
  auto const empty_table        = cudf::table_view{};
  auto const empty_column       = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  auto const output_empty_table = cudf::hash(empty_table, cudf::hash_id::HASH_SHA1);
  EXPECT_EQ(empty_column->size(), output_empty_table->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_empty_table->view());

  auto const table_one_empty_column  = cudf::table_view{{empty_column->view()}};
  auto const output_one_empty_column = cudf::hash(empty_table, cudf::hash_id::HASH_SHA1);
  EXPECT_EQ(empty_column->size(), output_one_empty_column->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_one_empty_column->view());
}

TEST_F(SHA1HashTest, MultiValue)
{
  strings_column_wrapper const strings_col(
    {"",
     "0",
     "A 56 character string to test message padding algorithm.",
     "A 63 character string to test message padding algorithm, again.",
     "A 64 character string to test message padding algorithm, again!!",
     "A very long (greater than 128 bytes/char string) to execute a multi hash-step data point in "
     "the hash function being tested. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"});

  strings_column_wrapper const sha1_string_results1({"da39a3ee5e6b4b0d3255bfef95601890afd80709",
                                                     "b6589fc6ab0dc82cf12099d1c2d40ab994e8410c",
                                                     "cb73203438ab46ea54491c53e288a2703c440c4a",
                                                     "c595ebd13a785c1c2659e010a42e2ff9987ef51f",
                                                     "4ffaf61804c55b8c2171be548bef2e1d0baca17a",
                                                     "595965dd18f38087186162c788485fe249242131",
                                                     "a62ca720fbab830c8890044eacbeac216f1ca2e4",
                                                     "11e16c52273b5669a41d17ec7c187475193f88b3"});

  strings_column_wrapper const sha1_string_results2({"da39a3ee5e6b4b0d3255bfef95601890afd80709",
                                                     "fb96549631c835eb239cd614cc6b5cb7d295121a",
                                                     "e3977ee0ea7f238134ec93c79988fa84b7c5d79e",
                                                     "f6f75b6fa3c3d8d86b44fcb2c98c9ad4b37dcdd0",
                                                     "c7abd431a775c604edf41a62f7f215e7258dc16a",
                                                     "153fdf20d2bd8ae76241197314d6e0be7fe10f50",
                                                     "8c3656f7cb37898f9296c1965000d6da13fed64e",
                                                     "b4a848399375ec842c2cb445d98b5f80a4dce94f"});

  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, limits::min(), limits::max(), 1, 2, 3});

  // Different truth values should be equal
  fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0, 1, 1, 1});
  fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0, 1, 2, 255});

  auto const string_input1       = cudf::table_view({strings_col});
  auto const string_input2       = cudf::table_view({strings_col, strings_col});
  auto const sha1_string_output1 = cudf::hash(string_input1, cudf::hash_id::HASH_SHA1);
  auto const sha1_string_output2 = cudf::hash(string_input2, cudf::hash_id::HASH_SHA1);
  EXPECT_EQ(string_input1.num_rows(), sha1_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), sha1_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha1_string_output1->view(), sha1_string_results1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha1_string_output2->view(), sha1_string_results2);

  auto const input1       = cudf::table_view({strings_col, ints_col, bools_col1});
  auto const input2       = cudf::table_view({strings_col, ints_col, bools_col2});
  auto const sha1_output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA1);
  auto const sha1_output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA1);
  EXPECT_EQ(input1.num_rows(), sha1_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha1_output1->view(), sha1_output2->view());
}

TEST_F(SHA1HashTest, MultiValueNulls)
{
  // Nulls with different values should be equal
  strings_column_wrapper const strings_col1(
    {"",
     "Different but null!",
     "A very long (greater than 128 bytes/char string) to execute a multi hash-step data point in "
     "the hash function being tested. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"},
    {1, 0, 0, 1, 0});
  strings_column_wrapper const strings_col2({"",
                                             "Another string that is null.",
                                             "Very different... but null",
                                             "All work and no play makes Jack a dull boy",
                                             ""},
                                            {1, 0, 0, 1, 0});

  // Nulls with different values should be equal
  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col1({0, 100, -100, limits::min(), limits::max()},
                                                      {1, 0, 0, 1, 0});
  fixed_width_column_wrapper<int32_t> const ints_col2({0, -200, 200, limits::min(), limits::max()},
                                                      {1, 0, 0, 0, 1});

  // Nulls with different values should be equal
  // Different truthy values should be equal
  fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {1, 1, 0, 0, 1});
  fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {1, 1, 0, 1, 0});

  auto const input1 = cudf::table_view({strings_col1, ints_col1, bools_col1});
  auto const input2 = cudf::table_view({strings_col2, ints_col2, bools_col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA1);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA1);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA1HashTestTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SHA1HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(SHA1HashTestTyped, Equality)
{
  fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hash(input, cudf::hash_id::HASH_SHA1);
  auto const output2 = cudf::hash(input, cudf::hash_id::HASH_SHA1);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(SHA1HashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  fixed_width_column_wrapper<T> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  fixed_width_column_wrapper<T> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA1);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA1);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA1HashTestFloatTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SHA1HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(SHA1HashTestFloatTyped, TestExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  fixed_width_column_wrapper<T> const col1({T(0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  fixed_width_column_wrapper<T> const col2(
    {T(-0.0), T(100.0), T(-100.0), min, max, -nan, inf, -inf});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA1);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA1);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

class SHA224HashTest : public cudf::test::BaseFixture {
};

TEST_F(SHA224HashTest, EmptyTable)
{
  auto const empty_table        = cudf::table_view{};
  auto const empty_column       = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  auto const output_empty_table = cudf::hash(empty_table, cudf::hash_id::HASH_SHA224);
  EXPECT_EQ(empty_column->size(), output_empty_table->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_empty_table->view());

  auto const table_one_empty_column  = cudf::table_view{{empty_column->view()}};
  auto const output_one_empty_column = cudf::hash(empty_table, cudf::hash_id::HASH_SHA224);
  EXPECT_EQ(empty_column->size(), output_one_empty_column->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_one_empty_column->view());
}

TEST_F(SHA224HashTest, MultiValue)
{
  strings_column_wrapper const strings_col(
    {"",
     "0",
     "A 56 character string to test message padding algorithm.",
     "A 63 character string to test message padding algorithm, again.",
     "A 64 character string to test message padding algorithm, again!!",
     "A very long (greater than 128 bytes/char string) to execute a multi hash-step data point in "
     "the hash function being tested. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"});

  strings_column_wrapper const sha224_string_results1(
    {"d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f",
     "dfd5f9139a820075df69d7895015360b76d0360f3d4b77a845689614",
     "5d1ed8373987e403482cefe1662a63fa3076c0a5331d141f41654bbe",
     "0662c91000b99de7a20c89097dd62f59120398d52499497489ccff95",
     "f9ea303770699483f3e53263b32a3b3c876d1b8808ce84df4b8ca1c4",
     "2da6cd4bdaa0a99fd7236cd5507c52e12328e71192e83b32d2f110f9",
     "e7d0adb165079efc6c6343112f8b154aa3644ca6326f658aaa0f8e4a",
     "309cc09eaa051beea7d0b0159daca9b4e8a533cb554e8f382c82709e"});

  strings_column_wrapper const sha224_string_results2(
    {"d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f",
     "5538ae2b02d4ae0b7090dc908ca69cd11a2ffad43c7435f1dbad5e6a",
     "8e1955a473a149368dc0a931f99379b44b0bb752f206dbdf68629232",
     "8581001e08295b7884428c022378cfdd643c977aefe4512f0252dc30",
     "d5854dfe3c32996345b103a6a16c7bdfa924723d620b150737e77370",
     "dd56deac5f2caa579a440ee814fc04a3afaf805d567087ac3317beb3",
     "14fb559f6309604bedd89183f585f3b433932b5b0e675848feebf8ec",
     "d219eefea538491efcb69bc5bbef4177ad991d1b6e1367b5981b8c31"});

  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, limits::min(), limits::max(), 1, 2, 3});

  // Different truth values should be equal
  fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0, 1, 1, 1});
  fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0, 1, 2, 255});

  auto const string_input1         = cudf::table_view({strings_col});
  auto const string_input2         = cudf::table_view({strings_col, strings_col});
  auto const sha224_string_output1 = cudf::hash(string_input1, cudf::hash_id::HASH_SHA224);
  auto const sha224_string_output2 = cudf::hash(string_input2, cudf::hash_id::HASH_SHA224);
  EXPECT_EQ(string_input1.num_rows(), sha224_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), sha224_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha224_string_output1->view(), sha224_string_results1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha224_string_output2->view(), sha224_string_results2);

  auto const input1         = cudf::table_view({strings_col, ints_col, bools_col1});
  auto const input2         = cudf::table_view({strings_col, ints_col, bools_col2});
  auto const sha224_output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA224);
  auto const sha224_output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA224);
  EXPECT_EQ(input1.num_rows(), sha224_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha224_output1->view(), sha224_output2->view());
}

TEST_F(SHA224HashTest, MultiValueNulls)
{
  // Nulls with different values should be equal
  strings_column_wrapper const strings_col1(
    {"",
     "Different but null!",
     "A very long (greater than 128 bytes/char string) to execute a multi hash-step data point in "
     "the hash function being tested. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"},
    {1, 0, 0, 1, 0});
  strings_column_wrapper const strings_col2({"",
                                             "Another string that is null.",
                                             "Very different... but null",
                                             "All work and no play makes Jack a dull boy",
                                             ""},
                                            {1, 0, 0, 1, 0});

  // Nulls with different values should be equal
  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col1({0, 100, -100, limits::min(), limits::max()},
                                                      {1, 0, 0, 1, 0});
  fixed_width_column_wrapper<int32_t> const ints_col2({0, -200, 200, limits::min(), limits::max()},
                                                      {1, 0, 0, 0, 1});

  // Nulls with different values should be equal
  // Different truthy values should be equal
  fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {1, 1, 0, 0, 1});
  fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {1, 1, 0, 1, 0});

  auto const input1 = cudf::table_view({strings_col1, ints_col1, bools_col1});
  auto const input2 = cudf::table_view({strings_col2, ints_col2, bools_col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA224);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA224);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA224HashTestTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SHA224HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(SHA224HashTestTyped, Equality)
{
  fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hash(input, cudf::hash_id::HASH_SHA224);
  auto const output2 = cudf::hash(input, cudf::hash_id::HASH_SHA224);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(SHA224HashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  fixed_width_column_wrapper<T> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  fixed_width_column_wrapper<T> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA224);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA224);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA224HashTestFloatTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SHA224HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(SHA224HashTestFloatTyped, TestExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  fixed_width_column_wrapper<T> const col1({T(0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  fixed_width_column_wrapper<T> const col2(
    {T(-0.0), T(100.0), T(-100.0), min, max, -nan, inf, -inf});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA224);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA224);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

class SHA256HashTest : public cudf::test::BaseFixture {
};

TEST_F(SHA256HashTest, EmptyTable)
{
  auto const empty_table        = cudf::table_view{};
  auto const empty_column       = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  auto const output_empty_table = cudf::hash(empty_table, cudf::hash_id::HASH_SHA256);
  EXPECT_EQ(empty_column->size(), output_empty_table->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_empty_table->view());

  auto const table_one_empty_column  = cudf::table_view{{empty_column->view()}};
  auto const output_one_empty_column = cudf::hash(empty_table, cudf::hash_id::HASH_SHA256);
  EXPECT_EQ(empty_column->size(), output_one_empty_column->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_one_empty_column->view());
}

TEST_F(SHA256HashTest, MultiValue)
{
  strings_column_wrapper const strings_col(
    {"",
     "0",
     "A 56 character string to test message padding algorithm.",
     "A 63 character string to test message padding algorithm, again.",
     "A 64 character string to test message padding algorithm, again!!",
     "A very long (greater than 128 bytes/char string) to execute a multi hash-step data point in "
     "the hash function being tested. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"});

  strings_column_wrapper const sha256_string_results1(
    {"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
     "5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9",
     "d16883c666112142c1d72c9080b41161be7563250539e3f6ab6e2fdf2210074b",
     "11174fa180460f5d683c2e63fcdd897dcbf10c28a9225d3ced9a8bbc3774415d",
     "10a7d211e692c6f71bb9f7524ba1437588c2797356f05fc585340f002fe7015e",
     "339d610dcb030bb4222bcf18c8ab82d911bfe7fb95b2cd9f6785fd4562b02401",
     "2ce9936a4a2234bf8a76c37d92e01d549d03949792242e7f8a1ad68575e4e4a8",
     "255fdd4d80a72f67921eb36f3e1157ea3e995068cee80e430c034e0d3692f614"});

  strings_column_wrapper const sha256_string_results2(
    {"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
     "f1534392279bddbf9d43dde8701cb5be14b82f76ec6607bf8d6ad557f60f304e",
     "96c204fa5d44b2487abfec105a05f8ae634551604f6596202ca99e3724e3953a",
     "2e7be264f3ecbb2930e7c54bf6c5fc1f310a8c63c50916bb713f34699ed11719",
     "224e4dce71d5dbd5e79ba65aaced7ad9c4f45dda146278087b2b61d164f056f0",
     "91f3108d4e9c696fdb37ae49fdc6a2237f1d1f977b7216406cc8a6365355f43b",
     "490be480afe271685e9c1fdf46daac0b9bf7f25602e153ca92a0ddb0e4b662ef",
     "4ddc45855d7ce3ab09efacff1fbafb33502f7dd468dc5a62826689c1c658dbce"});

  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, limits::min(), limits::max(), 1, 2, 3});

  // Different truth values should be equal
  fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0, 1, 1, 1});
  fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0, 1, 2, 255});

  auto const string_input1         = cudf::table_view({strings_col});
  auto const string_input2         = cudf::table_view({strings_col, strings_col});
  auto const sha256_string_output1 = cudf::hash(string_input1, cudf::hash_id::HASH_SHA256);
  auto const sha256_string_output2 = cudf::hash(string_input2, cudf::hash_id::HASH_SHA256);
  EXPECT_EQ(string_input1.num_rows(), sha256_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), sha256_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha256_string_output1->view(), sha256_string_results1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha256_string_output2->view(), sha256_string_results2);

  auto const input1         = cudf::table_view({strings_col, ints_col, bools_col1});
  auto const input2         = cudf::table_view({strings_col, ints_col, bools_col2});
  auto const sha256_output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA256);
  auto const sha256_output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA256);
  EXPECT_EQ(input1.num_rows(), sha256_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha256_output1->view(), sha256_output2->view());
}

TEST_F(SHA256HashTest, MultiValueNulls)
{
  // Nulls with different values should be equal
  strings_column_wrapper const strings_col1(
    {"",
     "Different but null!",
     "A very long (greater than 128 bytes/char string) to execute a multi hash-step data point in "
     "the hash function being tested. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"},
    {1, 0, 0, 1, 0});
  strings_column_wrapper const strings_col2({"",
                                             "Another string that is null.",
                                             "Very different... but null",
                                             "All work and no play makes Jack a dull boy",
                                             ""},
                                            {1, 0, 0, 1, 0});

  // Nulls with different values should be equal
  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col1({0, 100, -100, limits::min(), limits::max()},
                                                      {1, 0, 0, 1, 0});
  fixed_width_column_wrapper<int32_t> const ints_col2({0, -200, 200, limits::min(), limits::max()},
                                                      {1, 0, 0, 0, 1});

  // Nulls with different values should be equal
  // Different truthy values should be equal
  fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {1, 1, 0, 0, 1});
  fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {1, 1, 0, 1, 0});

  auto const input1 = cudf::table_view({strings_col1, ints_col1, bools_col1});
  auto const input2 = cudf::table_view({strings_col2, ints_col2, bools_col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA256);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA256);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA256HashTestTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SHA256HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(SHA256HashTestTyped, Equality)
{
  fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hash(input, cudf::hash_id::HASH_SHA256);
  auto const output2 = cudf::hash(input, cudf::hash_id::HASH_SHA256);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(SHA256HashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  fixed_width_column_wrapper<T> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  fixed_width_column_wrapper<T> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA256);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA256);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA256HashTestFloatTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SHA256HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(SHA256HashTestFloatTyped, TestExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  fixed_width_column_wrapper<T> const col1({T(0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  fixed_width_column_wrapper<T> const col2(
    {T(-0.0), T(100.0), T(-100.0), min, max, -nan, inf, -inf});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA256);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA256);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

class SHA384HashTest : public cudf::test::BaseFixture {
};

TEST_F(SHA384HashTest, EmptyTable)
{
  auto const empty_table        = cudf::table_view{};
  auto const empty_column       = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  auto const output_empty_table = cudf::hash(empty_table, cudf::hash_id::HASH_SHA384);
  EXPECT_EQ(empty_column->size(), output_empty_table->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_empty_table->view());

  auto const table_one_empty_column  = cudf::table_view{{empty_column->view()}};
  auto const output_one_empty_column = cudf::hash(empty_table, cudf::hash_id::HASH_SHA384);
  EXPECT_EQ(empty_column->size(), output_one_empty_column->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_one_empty_column->view());
}

TEST_F(SHA384HashTest, MultiValue)
{
  strings_column_wrapper const strings_col(
    {"",
     "0",
     "A 56 character string to test message padding algorithm.",
     "A 63 character string to test message padding algorithm, again.",
     "A 64 character string to test message padding algorithm, again!!",
     "A very long (greater than 128 bytes/char string) to execute a multi hash-step data point in "
     "the hash function being tested. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"});

  strings_column_wrapper const sha384_string_results1(
    {"38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b"
     "95b",
     "5f91550edb03f0bb8917da57f0f8818976f5da971307b7ee4886bb951c4891a1f16f840dae8f655aa5df718884ebc"
     "15b",
     "982000cce895dc439edbcb7ba5b908cb5b7e939fe913d58506a486735a914b0dfbcebb02c33c428287baa0bfc7fe0"
     "948",
     "c3ea54e4d6d97c2a84dac9ac48ed9dd1a49118be880d8466044720cfdcd23427bf556f12204bb34ede29dbf207033"
     "78c",
     "5d7a853a18138fa90feac07c896dfca65a0f1eb2ed40f1fd7be6238dd7ef429bb1aeb0236735500eb954c9b4ba923"
     "254",
     "c72bcaf3a4b01986711cd5d2614aa8f9d7fad61455613eac4561b1468f9a25dd26566c8ad1190dec7567be4f6fc1d"
     "b29",
     "281826f23bebb3f835d2f15edcb0cdb3078ae2d7dc516f3a366af172dff4db6dd5833bc1e5ee411d52c598773e939"
     "7b6",
     "3a9d1a870a5f6a4c04df1daf1808163d33852897ebc757a5b028a1214fbc758485a392159b11bc360cfadc79f9512"
     "822"});

  strings_column_wrapper const sha384_string_results2(
    {"38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b"
     "95b",
     "34ae2cd40efabf896d8d4173e500278d10671b2d914efb5480e8349190bc7e8e1d532ad568d00a8295ea536a9b42b"
     "bc6",
     "e80c25efd8032ea94dad1509a68f9bf745ce1184b8a148714c28c7e0fae1100ab14057417394f83118eaa151e014d"
     "917",
     "69eaddc4ef2ed967fc6a86d3ed3777b2c2015df4cf8bbbf65681556f451a4a0ae805a89c2d56641b4422b5f248c56"
     "77d",
     "112a6f9c74741d490747db90f5e901a88b7a32f637c030d6d96e5f89a70a5f1ee209e018648842c0e1d32002f95fd"
     "d07",
     "dc6f24bb0eb2c96fb53c52c402f073de089f3aeae9594be0c4f4cb31b13bd48769b80aa97d83a25ece1edf0c83373"
     "f56",
     "781a33adfdcdcbb514318728c074fbb59d44002995825642e0c9bfef8a2ccf3fb637b39ff3dd265df8cd93c86e945"
     "ce9",
     "d2efb1591c4503f23c34ddb4da6bb1017d3d4d7c9f23ee6aa52e71c98d41060bc35eb22f41b6130d5c42a6e717fb3"
     "edf"});

  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, limits::min(), limits::max(), 1, 2, 3});

  // Different truth values should be equal
  fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0, 1, 1, 1});
  fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0, 1, 2, 255});

  auto const string_input1         = cudf::table_view({strings_col});
  auto const string_input2         = cudf::table_view({strings_col, strings_col});
  auto const sha384_string_output1 = cudf::hash(string_input1, cudf::hash_id::HASH_SHA384);
  auto const sha384_string_output2 = cudf::hash(string_input2, cudf::hash_id::HASH_SHA384);
  EXPECT_EQ(string_input1.num_rows(), sha384_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), sha384_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha384_string_output1->view(), sha384_string_results1, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha384_string_output2->view(), sha384_string_results2, verbosity);

  auto const input1         = cudf::table_view({strings_col, ints_col, bools_col1});
  auto const input2         = cudf::table_view({strings_col, ints_col, bools_col2});
  auto const sha384_output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA384);
  auto const sha384_output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA384);
  EXPECT_EQ(input1.num_rows(), sha384_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha384_output1->view(), sha384_output2->view(), verbosity);
}

TEST_F(SHA384HashTest, MultiValueNulls)
{
  // Nulls with different values should be equal
  strings_column_wrapper const strings_col1(
    {"",
     "Different but null!",
     "A very long (greater than 128 bytes/char string) to execute a multi hash-step data point in "
     "the hash function being tested. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"},
    {1, 0, 0, 1, 0});
  strings_column_wrapper const strings_col2({"",
                                             "Another string that is null.",
                                             "Very different... but null",
                                             "All work and no play makes Jack a dull boy",
                                             ""},
                                            {1, 0, 0, 1, 0});

  // Nulls with different values should be equal
  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col1({0, 100, -100, limits::min(), limits::max()},
                                                      {1, 0, 0, 1, 0});
  fixed_width_column_wrapper<int32_t> const ints_col2({0, -200, 200, limits::min(), limits::max()},
                                                      {1, 0, 0, 0, 1});

  // Nulls with different values should be equal
  // Different truthy values should be equal
  fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {1, 1, 0, 0, 1});
  fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {1, 1, 0, 1, 0});

  auto const input1 = cudf::table_view({strings_col1, ints_col1, bools_col1});
  auto const input2 = cudf::table_view({strings_col2, ints_col2, bools_col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA384);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA384);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA384HashTestTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SHA384HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(SHA384HashTestTyped, Equality)
{
  fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hash(input, cudf::hash_id::HASH_SHA384);
  auto const output2 = cudf::hash(input, cudf::hash_id::HASH_SHA384);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(SHA384HashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  fixed_width_column_wrapper<T> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  fixed_width_column_wrapper<T> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA384);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA384);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA384HashTestFloatTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SHA384HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(SHA384HashTestFloatTyped, TestExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  fixed_width_column_wrapper<T> const col1({T(0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  fixed_width_column_wrapper<T> const col2(
    {T(-0.0), T(100.0), T(-100.0), min, max, -nan, inf, -inf});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA384);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA384);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

class SHA512HashTest : public cudf::test::BaseFixture {
};

TEST_F(SHA512HashTest, EmptyTable)
{
  auto const empty_table        = cudf::table_view{};
  auto const empty_column       = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  auto const output_empty_table = cudf::hash(empty_table, cudf::hash_id::HASH_SHA512);
  EXPECT_EQ(empty_column->size(), output_empty_table->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_empty_table->view());

  auto const table_one_empty_column  = cudf::table_view{{empty_column->view()}};
  auto const output_one_empty_column = cudf::hash(empty_table, cudf::hash_id::HASH_SHA512);
  EXPECT_EQ(empty_column->size(), output_one_empty_column->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_one_empty_column->view());
}

TEST_F(SHA512HashTest, MultiValue)
{
  strings_column_wrapper const strings_col(
    {"",
     "0",
     "A 56 character string to test message padding algorithm.",
     "A 63 character string to test message padding algorithm, again.",
     "A 64 character string to test message padding algorithm, again!!",
     "A very long (greater than 128 bytes/char string) to execute a multi hash-step data point in "
     "the hash function being tested. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"});

  strings_column_wrapper const sha512_string_results1(
    {"cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877ee"
     "c2f63b931bd47417a81a538327af927da3e",
     "31bca02094eb78126a517b206a88c73cfa9ec6f704c7030d18212cace820f025f00bf0ea68dbf3f3a5436ca63b53b"
     "f7bf80ad8d5de7d8359d0b7fed9dbc3ab99",
     "1d8b355dbe0c4ad81c9815a1490f0b6a6fa710e42ca60767ffd6d845acd116defe307c9496a80c4a67653873af6ed"
     "83e2e04c2102f55f9cd402677b246832e4c",
     "8ac8ae9de5597aa630f071f81fcb94dc93b6a8f92d8f2cdd5a469764a5daf6ef387b6465ae097dcd6e0c64286260d"
     "cc3d2c789d2cf5960df648c78a765e6c27c",
     "9c436e24be60e17425a1a829642d97e7180b57485cf95db007cf5b32bbae1f2325b6874b3377e37806b15b739bffa"
     "412ea6d095b726487d70e7b50e92d56c750",
     "6a25ca1f20f6e79faea2a0770075e4262beb66b40f59c22d3e8abdb6188ef8d8914faf5dbf6df76165bb61b81dfda"
     "46643f0d6366a39f7bd3d270312f9d3cf87",
     "bae9eb4b5c05a4c5f85750b70b2f0ce78e387f992f0927a017eb40bd180a13004f6252a6bbf9816f195fb7d86668c"
     "393dc0985aaf7168f48e8b905f3b9b02df2",
     "05a4ca1c523dcab32edb7d8793934a4cdf41a9062b229d711f5326e297bda83fa965118b9d7636172b43688e8e149"
     "008b3f967f1a969962b7e959af894a8a315"});

  strings_column_wrapper const sha512_string_results2(
    {"cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877ee"
     "c2f63b931bd47417a81a538327af927da3e",
     "8ab3361c051a97ddc3c665d29f2762f8ac4240d08995f8724b6d07d8cbedd32c28f589ccdae514f20a6c8eea6f755"
     "408dd3dd6837d66932ca2352eaeab594427",
     "338b22eb841420affff9904f903ed14c91bf8f4d1b10f25c145a31018367607a2cf562121ba7eaa2d08db3382cc82"
     "149805198c1fa3e7dc714fc2782e0f6ebd8",
     "d3045ecde16ea036d2f2ff3fa685beb46d5fcb73de71f0aee653265f18b22e4c131255e6eb5ad3be2f32914408ec6"
     "67911b49d951714decbdbfca1957be8ba10",
     "da7706221f8861ef522ab9555f57306382fb18c337536545d839e431dede4ff9f9affafb82ab5588734a8fc6631e6"
     "a0cd864634b62e24a42755c863c5d5c5848",
     "04dadc8fdf205fe535c8eb38f20882fc2a0e308081052d7588e74f6620aa207749039468c126db7407050def80415"
     "1d037cb188d5d4d459015032972a9e9f001",
     "aae2e742074847889a029a8d3170f9e17177d48ec0b9dabe572aa68dd3001af0c512f164ba84aa75b13950948170a"
     "0912912d16c98d2f05cb633c0d5b6a9105e",
     "77f46e99a7a51ac04b4380ebca70c0782381629f711169a3b9dad3fc9aa6221a9c0cdaa9b9ea4329773e773e2987c"
     "d1eebe0661386909684927d67819a2cf736"});

  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, limits::min(), limits::max(), 1, 2, 3});

  // Different truth values should be equal
  fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0, 1, 1, 1});
  fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0, 1, 2, 255});

  auto const string_input1         = cudf::table_view({strings_col});
  auto const string_input2         = cudf::table_view({strings_col, strings_col});
  auto const sha512_string_output1 = cudf::hash(string_input1, cudf::hash_id::HASH_SHA512);
  auto const sha512_string_output2 = cudf::hash(string_input2, cudf::hash_id::HASH_SHA512);
  EXPECT_EQ(string_input1.num_rows(), sha512_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), sha512_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha512_string_output1->view(), sha512_string_results1, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha512_string_output2->view(), sha512_string_results2, verbosity);

  auto const input1         = cudf::table_view({strings_col, ints_col, bools_col1});
  auto const input2         = cudf::table_view({strings_col, ints_col, bools_col2});
  auto const sha512_output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA512);
  auto const sha512_output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA512);
  EXPECT_EQ(input1.num_rows(), sha512_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha512_output1->view(), sha512_output2->view(), verbosity);
}

TEST_F(SHA512HashTest, MultiValueNulls)
{
  // Nulls with different values should be equal
  strings_column_wrapper const strings_col1(
    {"",
     "Different but null!",
     "A very long (greater than 128 bytes/char string) to execute a multi hash-step data point in "
     "the hash function being tested. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"},
    {1, 0, 0, 1, 0});
  strings_column_wrapper const strings_col2({"",
                                             "Another string that is null.",
                                             "Very different... but null",
                                             "All work and no play makes Jack a dull boy",
                                             ""},
                                            {1, 0, 0, 1, 0});

  // Nulls with different values should be equal
  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col1({0, 100, -100, limits::min(), limits::max()},
                                                      {1, 0, 0, 1, 0});
  fixed_width_column_wrapper<int32_t> const ints_col2({0, -200, 200, limits::min(), limits::max()},
                                                      {1, 0, 0, 0, 1});

  // Nulls with different values should be equal
  // Different truthy values should be equal
  fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {1, 1, 0, 0, 1});
  fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {1, 1, 0, 1, 0});

  auto const input1 = cudf::table_view({strings_col1, ints_col1, bools_col1});
  auto const input2 = cudf::table_view({strings_col2, ints_col2, bools_col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA512);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA512);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA512HashTestTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SHA512HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(SHA512HashTestTyped, Equality)
{
  fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hash(input, cudf::hash_id::HASH_SHA512);
  auto const output2 = cudf::hash(input, cudf::hash_id::HASH_SHA512);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(SHA512HashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  fixed_width_column_wrapper<T> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  fixed_width_column_wrapper<T> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA512);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA512);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA512HashTestFloatTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SHA512HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(SHA512HashTestFloatTyped, TestExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  fixed_width_column_wrapper<T> const col1({T(0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  fixed_width_column_wrapper<T> const col2(
    {T(-0.0), T(100.0), T(-100.0), min, max, -nan, inf, -inf});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA512);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA512);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

CUDF_TEST_PROGRAM_MAIN()
