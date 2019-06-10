/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <stream_compaction.hpp>
#include <table.hpp>

#include <utilities/error_utils.hpp>

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>

#include<algorithm>
#include<limits>

struct ApplyBooleanMaskErrorTest : GdfTest {};

// Test ill-formed inputs

TEST_F(ApplyBooleanMaskErrorTest, NullPtrs)
{
  constexpr gdf_size_type column_size{1000};

  gdf_column bad_input, bad_mask;
  gdf_column_view(&bad_input, 0, 0, 0, GDF_INT32);
  gdf_column_view(&bad_mask,  0, 0, 0, GDF_BOOL8);

  cudf::test::column_wrapper<int32_t> source{column_size};
  cudf::test::column_wrapper<cudf::bool8> mask{column_size};

  CUDF_EXPECT_NO_THROW(cudf::apply_boolean_mask(bad_input, mask));

  bad_input.valid = reinterpret_cast<gdf_valid_type*>(0x0badf00d);
  bad_input.null_count = 2;
  bad_input.size = column_size; 
  // nonzero, with non-null valid mask, so non-null input expected
  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(bad_input, mask),
                            "Null input data");

  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(source, bad_mask),
                            "Null boolean_mask");
}

TEST_F(ApplyBooleanMaskErrorTest, SizeMismatch)
{
  constexpr gdf_size_type column_size{1000};
  constexpr gdf_size_type mask_size{500};

  cudf::test::column_wrapper<int32_t> source{column_size};
  cudf::test::column_wrapper<cudf::bool8> mask{mask_size};
             
  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(source, mask), 
                            "Column size mismatch");
}

TEST_F(ApplyBooleanMaskErrorTest, NonBooleanMask)
{
  constexpr gdf_size_type column_size{1000};

  cudf::test::column_wrapper<int32_t> source{column_size};
  cudf::test::column_wrapper<float> nonbool_mask{column_size};
             
  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(source, nonbool_mask), 
                            "Mask must be Boolean type");

  cudf::test::column_wrapper<cudf::bool8> bool_mask{column_size, true};
  EXPECT_NO_THROW(cudf::apply_boolean_mask(source, bool_mask));
}

template <typename T>
struct ApplyBooleanMaskTest : GdfTest {};

using test_types =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;
TYPED_TEST_CASE(ApplyBooleanMaskTest, test_types);

// Test computation

/*
 * Runs apply_boolean_mask checking for errors, and compares the result column 
 * to the specified expected result column.
 */
template <typename T>
void BooleanMaskTest(cudf::test::column_wrapper<T> source,
                     cudf::test::column_wrapper<cudf::bool8> mask,
                     cudf::test::column_wrapper<T> expected)
{
  gdf_column result;
  EXPECT_NO_THROW(result = cudf::apply_boolean_mask(source, mask));

  EXPECT_TRUE(expected == result);

  /*if (!(expected == result)) {
    std::cout << "expected\n";
    expected.print();
    std::cout << expected.get()->null_count << "\n";
    std::cout << "result\n";
    print_gdf_column(&result);
    std::cout << result.null_count << "\n";
  }*/

  gdf_column_free(&result);
}

constexpr gdf_size_type column_size{1000000};

TYPED_TEST(ApplyBooleanMaskTest, Identity)
{
  BooleanMaskTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<cudf::bool8>{column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }});
}

TYPED_TEST(ApplyBooleanMaskTest, MaskAllFalse)
{
  BooleanMaskTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<cudf::bool8>{column_size,
      [](gdf_index_type row) { return cudf::bool8{false}; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<TypeParam>{0, false});
}

TYPED_TEST(ApplyBooleanMaskTest, MaskAllNull)
{
  BooleanMaskTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<cudf::bool8>{column_size, 
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return false; }},
    cudf::test::column_wrapper<TypeParam>{0, false});
}

TYPED_TEST(ApplyBooleanMaskTest, MaskEvensFalse)
{
  BooleanMaskTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<cudf::bool8>{column_size,
      [](gdf_index_type row) { return cudf::bool8{row % 2 == 1}; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<TypeParam>{column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1;  },
      [](gdf_index_type row) { return true; }});
}

TYPED_TEST(ApplyBooleanMaskTest, MaskEvensNull)
{
  // mix it up a bit by setting the input odd values to be null
  // Since the bool mask has even values null, the output
  // vector should have all values nulled

  BooleanMaskTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return row % 2 == 0; }},
    cudf::test::column_wrapper<cudf::bool8>{column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return row % 2 == 1; }},
    cudf::test::column_wrapper<TypeParam>{column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1;  },
      [](gdf_index_type row) { return false; }});
}

TYPED_TEST(ApplyBooleanMaskTest, NonalignedGap)
{
  const int start{1}, end{column_size / 4};

  BooleanMaskTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<cudf::bool8>{column_size,
      [](gdf_index_type row) { return cudf::bool8{(row < start) || (row >= end)}; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<TypeParam>{column_size - (end - start),
      [](gdf_index_type row) { return (row < start) ? row : row + end - start; },
      [&](gdf_index_type row) { return true; }});
}

TYPED_TEST(ApplyBooleanMaskTest, NoNullMask)
{
  std::vector<TypeParam> source(column_size, TypeParam{0});
  std::vector<TypeParam> expected(column_size / 2, TypeParam{0});
  std::iota(source.begin(), source.end(), TypeParam{0});
  std::generate(expected.begin(), expected.end(), 
                [n = -1] () mutable { return n+=2; });

  BooleanMaskTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{source},
    cudf::test::column_wrapper<cudf::bool8>{column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return row % 2 == 1; }},
    cudf::test::column_wrapper<TypeParam> {expected});
}

struct DropNullsErrorTest : GdfTest {};

TEST_F(DropNullsErrorTest, EmptyInput)
{
  gdf_column bad_input;
  gdf_column_view(&bad_input, 0, 0, 0, GDF_INT32);

  // zero size, so expect no error, just empty output column
  gdf_column output;
  CUDF_EXPECT_NO_THROW(output = cudf::drop_nulls(bad_input));
  EXPECT_EQ(output.size, 0);
  EXPECT_EQ(output.null_count, 0);
  EXPECT_EQ(output.data, nullptr);
  EXPECT_EQ(output.valid, nullptr);

  bad_input.valid = reinterpret_cast<gdf_valid_type*>(0x0badf00d);
  bad_input.null_count = 1;
  bad_input.size = 2; 
  // nonzero, with non-null valid mask, so non-null input expected
  CUDF_EXPECT_THROW_MESSAGE(cudf::drop_nulls(bad_input), "Null input data");
}

/*
 * Runs drop_nulls checking for errors, and compares the result column 
 * to the specified expected result column.
 */
template <typename T>
void DropNulls(cudf::test::column_wrapper<T> source,
               cudf::test::column_wrapper<T> expected)
{
  gdf_column result;
  EXPECT_NO_THROW(result = cudf::drop_nulls(source));
  EXPECT_EQ(result.null_count, 0);
  EXPECT_TRUE(expected == result);

  /*if (!(expected == result)) {
    std::cout << "expected\n";
    expected.print();
    std::cout << expected.get()->null_count << "\n";
    std::cout << "result\n";
    print_gdf_column(&result);
    std::cout << result.null_count << "\n";
  }*/

  gdf_column_free(&result);
}

template <typename T>
struct DropNullsTest : GdfTest {};

TYPED_TEST_CASE(DropNullsTest, test_types);

TYPED_TEST(DropNullsTest, Identity)
{
  DropNulls<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }});
}

TYPED_TEST(DropNullsTest, AllNull)
{
  DropNulls<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return false; }},
    cudf::test::column_wrapper<TypeParam>{0, false});
}

TYPED_TEST(DropNullsTest, EvensNull)
{
  DropNulls<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return row % 2 == 1; }},
    cudf::test::column_wrapper<TypeParam>{column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1;  },
      [](gdf_index_type row) { return true; }});
}

TYPED_TEST(DropNullsTest, NonalignedGap)
{
  const int start{1}, end{column_size / 4};

  DropNulls<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return (row < start) || (row >= end); }},
    cudf::test::column_wrapper<TypeParam>{column_size - (end - start),
      [](gdf_index_type row) { return (row < start) ? row : row + end - start; },
      [&](gdf_index_type row) { return true; }});
}

TYPED_TEST(DropNullsTest, NoNullMask)
{
  std::vector<TypeParam> source(column_size, TypeParam{0});
  std::vector<TypeParam> expected(column_size, TypeParam{0});
  std::iota(source.begin(), source.end(), TypeParam{0});
  std::iota(expected.begin(), expected.end(), TypeParam{0});

  DropNulls<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{source},
    cudf::test::column_wrapper<TypeParam> {expected});
}


template <typename T>
struct DropDuplicatesTest : GdfTest { };

using test_types2 =
  ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_CASE(DropDuplicatesTest, test_types2);

template <typename T>
void TypedDropDuplicatesTest(cudf::test::column_wrapper<T> source,
                     cudf::test::column_wrapper<T> expected)
{
  gdf_column* inrow[]{source.get()};
  cudf::table input_table(inrow, 1);
  cudf::duplicate_keep_option keep=cudf::duplicate_keep_option::KEEP_LAST;
  cudf::table out_table;

  EXPECT_NO_THROW(out_table = cudf::drop_duplicates(input_table, input_table, keep));

  gdf_column result = *(out_table.get_column(0));
  EXPECT_TRUE(expected == result);

  /*
  if (!(expected == result)) {
    std::cout << "expected_col1["<< expected.get()->size<< "]\n";
    expected.print();
    std::cout << "result["<< result.size << "]\n";
    print_gdf_column(&result);
  }
  */

  gdf_column_free(&result);
}

TYPED_TEST(DropDuplicatesTest, Empty)
{
  TypedDropDuplicatesTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{0, false},
    cudf::test::column_wrapper<TypeParam>{0, false});
}

TYPED_TEST(DropDuplicatesTest, Distinct)
{
  constexpr gdf_size_type column_size = 
    std::numeric_limits<TypeParam>::max() >1000000? 1000000:
    std::numeric_limits<TypeParam>::max();
  TypedDropDuplicatesTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; }},
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; }});
}

TYPED_TEST(DropDuplicatesTest, SingleValue)
{
  TypedDropDuplicatesTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return 2; }},
    cudf::test::column_wrapper<TypeParam>{1,
      [](gdf_index_type row) { return 2; }});
}

TYPED_TEST(DropDuplicatesTest, Duplicate)
{
  TypedDropDuplicatesTest<TypeParam>(
      cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row%100; }},
      cudf::test::column_wrapper<TypeParam>{100,
      [](gdf_index_type row) { return row;  }});
}


template <class T>
struct DropDuplicatesDoubleTest : GdfTest { };

//using TestingTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float,
//                                      double, cudf::date32, cudf::date64,
//                                      cudf::timestamp, cudf::category>;
//                                   cudf::nvstring_category, cudf::bool8>;

template <typename A, typename B>
struct TypeDefinitions
{
  typedef A Type0;
  typedef B Type1;
};

// The list of types we want to test.
typedef ::testing::Types <TypeDefinitions<int32_t,int32_t>,
                          TypeDefinitions<int32_t,float>,
                          TypeDefinitions<float,int32_t>,
                          TypeDefinitions<int32_t,double>,
                          TypeDefinitions<double, cudf::date32>,
                          TypeDefinitions<cudf::date32, cudf::date64>
                                  > Implementations;

TYPED_TEST_CASE(DropDuplicatesDoubleTest, Implementations);

template <class T>
void TypedDropDuplicatesTest(
    cudf::test::column_wrapper<typename T::Type0> source_col1, 
    cudf::test::column_wrapper<typename T::Type1> source_col2, 
    cudf::test::column_wrapper<typename T::Type0> expected_col1, 
    cudf::test::column_wrapper<typename T::Type1> expected_col2,
    cudf::duplicate_keep_option keep)
{
  gdf_column* inrow[]{source_col1.get(), source_col2.get()};
  cudf::table input_table(inrow, 2);
  cudf::table out_table;

  EXPECT_NO_THROW(out_table = cudf::drop_duplicates(input_table, input_table, keep));

  gdf_column result_col1 = *(out_table.get_column(0));
  EXPECT_TRUE(expected_col1 == result_col1);
  gdf_column result_col2 = *(out_table.get_column(1));
  EXPECT_TRUE(expected_col2 == result_col2);

  /*
  if (!(expected_col1 == result_col1)) {
    std::cout << "expected_col1["<< expected_col1.get()->size<< "]\n";
    expected_col1.print();
    std::cout << "result_col1["<< result_col1.size << "]\n";
    print_gdf_column(&result_col1);
  }
  if (!(expected_col2 == result_col2)) {
    std::cout << "expected_col2["<< expected_col2.get()->size<< "]\n";
    expected_col2.print();
    std::cout << "result_col2["<< result_col2.size << "]\n";
    print_gdf_column(&result_col2);
  }
  */

  gdf_column_free(&result_col1);
  gdf_column_free(&result_col2);
}

TYPED_TEST(DropDuplicatesDoubleTest, Empty)
{
  TypedDropDuplicatesTest<TypeParam>(
    cudf::test::column_wrapper<typename TypeParam::Type0>{0, false},
    cudf::test::column_wrapper<typename TypeParam::Type1>{0, false},
    cudf::test::column_wrapper<typename TypeParam::Type0>{0, false},
    cudf::test::column_wrapper<typename TypeParam::Type1>{0, false},
    cudf::duplicate_keep_option::KEEP_LAST);
}

TYPED_TEST(DropDuplicatesDoubleTest, Distinct)
{
  auto lamda_type0 = [](gdf_index_type row) { return typename TypeParam::Type0(row); };
  auto lamda_type1 = [](gdf_index_type row) { return typename TypeParam::Type1(row); };
  cudf::test::column_wrapper<typename TypeParam::Type0> col1{column_size, lamda_type0};
  cudf::test::column_wrapper<typename TypeParam::Type1> col2{column_size, lamda_type1};
  TypedDropDuplicatesTest<TypeParam>(
    col1,
    col2,
    cudf::test::column_wrapper<typename TypeParam::Type0>{column_size, lamda_type0},
    cudf::test::column_wrapper<typename TypeParam::Type1>{column_size, lamda_type1},
    cudf::duplicate_keep_option::KEEP_FIRST);

  TypedDropDuplicatesTest<TypeParam>(
    col1,
    col2,
    cudf::test::column_wrapper<typename TypeParam::Type0>{column_size, lamda_type0},
    cudf::test::column_wrapper<typename TypeParam::Type1>{column_size, lamda_type1},
    cudf::duplicate_keep_option::KEEP_LAST);

  TypedDropDuplicatesTest<TypeParam>(
    col1,
    col2,
    cudf::test::column_wrapper<typename TypeParam::Type0>{column_size, lamda_type0},
    cudf::test::column_wrapper<typename TypeParam::Type1>{column_size, lamda_type1},
    cudf::duplicate_keep_option::KEEP_FALSE);
}

TYPED_TEST(DropDuplicatesDoubleTest, Duplicate)
{
  auto lamda_type0 = [](gdf_index_type row) { return typename TypeParam::Type0(row%100); };
  auto lamda_type1 = [](gdf_index_type row) { return typename TypeParam::Type1(row%7); };
  cudf::test::column_wrapper<typename TypeParam::Type0> col1{column_size, lamda_type0};
  cudf::test::column_wrapper<typename TypeParam::Type1> col2{column_size, lamda_type1};
  TypedDropDuplicatesTest<TypeParam>(
    col1,
    col2,
    cudf::test::column_wrapper<typename TypeParam::Type0>{700, lamda_type0},
    cudf::test::column_wrapper<typename TypeParam::Type1>{700, lamda_type1},
    cudf::duplicate_keep_option::KEEP_FIRST);

  TypedDropDuplicatesTest<TypeParam>(
    col1,
    col2,
    cudf::test::column_wrapper<typename TypeParam::Type0>{700, lamda_type0},
    cudf::test::column_wrapper<typename TypeParam::Type1>{700,
      [](gdf_index_type row) { return typename TypeParam::Type1((column_size-700+row)%7); }},
    cudf::duplicate_keep_option::KEEP_LAST);

  TypedDropDuplicatesTest<TypeParam>(
    col1,
    col2,
    cudf::test::column_wrapper<typename TypeParam::Type0>{0, lamda_type0},
    cudf::test::column_wrapper<typename TypeParam::Type1>{0, lamda_type1},
    cudf::duplicate_keep_option::KEEP_FALSE);
}

