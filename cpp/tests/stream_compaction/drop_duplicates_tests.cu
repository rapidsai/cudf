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

#include <cudf/stream_compaction.hpp>
#include <cudf/legacy/table.hpp>
#include <cudf/copying.hpp>

#include <utilities/error_utils.hpp>

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>

#include<limits>


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
  enum cudf::duplicate_keep_option keep=cudf::duplicate_keep_option::KEEP_LAST;
  cudf::table out_table;

  EXPECT_NO_THROW(out_table = cudf::drop_duplicates(input_table, input_table, keep));

  gdf_column result = *(out_table.get_column(0));
  EXPECT_TRUE(expected == result);

  //*
  if (!(expected == result)) {
    std::cout << "expected["<< expected.get()->size<< "]\n";
    expected.print();
    std::cout << "result["<< result.size << "]\n";
    print_gdf_column(&result);
  }
  //*/

  gdf_column_free(&result);
}

constexpr gdf_size_type column_size{1000000};

TYPED_TEST(DropDuplicatesTest, Empty)
{
  TypedDropDuplicatesTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{}, //{0, false},
    cudf::test::column_wrapper<TypeParam>{}); //{0, false});
}

TYPED_TEST(DropDuplicatesTest, Distinct)
{
  constexpr gdf_size_type column_size = 
    std::numeric_limits<TypeParam>::max() >1000000? 1000000:
    std::numeric_limits<TypeParam>::max();
  TypedDropDuplicatesTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; }, false},
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }
      });
}

TYPED_TEST(DropDuplicatesTest, SingleValue)
{
  TypedDropDuplicatesTest<TypeParam>(
    cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return 2; }, false},
    cudf::test::column_wrapper<TypeParam>{1,
      [](gdf_index_type row) { return 2; }, 
      [](gdf_index_type row) { return true; }
      });
      //[](gdf_index_type row) { return true; }});
}

TYPED_TEST(DropDuplicatesTest, Duplicate)
{
  TypedDropDuplicatesTest<TypeParam>(
      cudf::test::column_wrapper<TypeParam>{column_size,
      [](gdf_index_type row) { return row%100; }, false},
      cudf::test::column_wrapper<TypeParam>{100,
      [](gdf_index_type row) { return row;  }, 
      [](gdf_index_type row) { return true; }
      });
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
bool compare_columns_indexed(
    cudf::table out_table,
    cudf::test::column_wrapper<typename T::Type0> expected_col1, 
    cudf::test::column_wrapper<typename T::Type1> expected_col2
    )
{
  int rows = expected_col1.size();
  gdf_column index_col = *(out_table.get_column(0));
  gdf_index_type* index_ptr = ((gdf_index_type*)index_col.data);

  rmm::device_vector<gdf_index_type> ordered(rows);
  auto exec = rmm::exec_policy(0)->on(0);
  thrust::sequence(exec, ordered.begin(), ordered.end());
  thrust::sort_by_key(exec, index_ptr , index_ptr + rows, ordered.data().get());

  cudf::table destination_table(rows,
                                cudf::column_dtypes(out_table),
                                cudf::column_dtype_infos(out_table), true);
  cudf::gather(&out_table, ordered.data().get(), &destination_table);

  gdf_column result_col1 = *(destination_table.get_column(1));
  EXPECT_TRUE(expected_col1 == result_col1);

  gdf_column result_col2 = *(destination_table.get_column(2));
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

  gdf_column_free(&index_col);
  gdf_column_free(&result_col1);
  gdf_column_free(&result_col2);
  return true;
}

template <class T>
void TypedDropDuplicatesTest(
    cudf::test::column_wrapper<typename T::Type0> source_col1, 
    cudf::test::column_wrapper<typename T::Type1> source_col2, 
    cudf::test::column_wrapper<typename T::Type0> expected_col1, 
    cudf::test::column_wrapper<typename T::Type1> expected_col2,
    enum cudf::duplicate_keep_option keep)
{
  cudf::test::column_wrapper<gdf_index_type> index{source_col1.size(),
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }};
   
  gdf_column* inrow[]{index.get(), source_col1.get(), source_col2.get()};
  cudf::table input_table(inrow, 3);
  cudf::table keycol_table(inrow+1, 2);
  cudf::table out_table;

  EXPECT_NO_THROW(out_table = cudf::drop_duplicates(input_table, keycol_table, keep));

  //reorder and compare columns
  compare_columns_indexed<T>(out_table, expected_col1, expected_col2);
}

auto lamda_valid = [](gdf_index_type row) { return true; };

TYPED_TEST(DropDuplicatesDoubleTest, Empty)
{
  TypedDropDuplicatesTest<TypeParam>(
    cudf::test::column_wrapper<typename TypeParam::Type0>{}, //{0, false},
    cudf::test::column_wrapper<typename TypeParam::Type1>{}, //{0, false},
    cudf::test::column_wrapper<typename TypeParam::Type0>{}, //{0, false},
    cudf::test::column_wrapper<typename TypeParam::Type1>{}, //{0, false},
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
    cudf::test::column_wrapper<typename TypeParam::Type0>{column_size, lamda_type0, lamda_valid},
    cudf::test::column_wrapper<typename TypeParam::Type1>{column_size, lamda_type1, lamda_valid},
    cudf::duplicate_keep_option::KEEP_FIRST);

  TypedDropDuplicatesTest<TypeParam>(
    col1,
    col2,
    cudf::test::column_wrapper<typename TypeParam::Type0>{column_size, lamda_type0, lamda_valid},
    cudf::test::column_wrapper<typename TypeParam::Type1>{column_size, lamda_type1, lamda_valid},
    cudf::duplicate_keep_option::KEEP_LAST);

  TypedDropDuplicatesTest<TypeParam>(
    col1,
    col2,
    cudf::test::column_wrapper<typename TypeParam::Type0>{column_size, lamda_type0, lamda_valid},
    cudf::test::column_wrapper<typename TypeParam::Type1>{column_size, lamda_type1, lamda_valid},
    cudf::duplicate_keep_option::KEEP_NONE);
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
    cudf::test::column_wrapper<typename TypeParam::Type0>{700, lamda_type0, lamda_valid},
    cudf::test::column_wrapper<typename TypeParam::Type1>{700, lamda_type1, lamda_valid},
    cudf::duplicate_keep_option::KEEP_FIRST);

  TypedDropDuplicatesTest<TypeParam>(
    col1,
    col2,
    cudf::test::column_wrapper<typename TypeParam::Type0>{700, lamda_type0, lamda_valid},
    cudf::test::column_wrapper<typename TypeParam::Type1>{700,
      [](gdf_index_type row) { return typename TypeParam::Type1((column_size-700+row)%7); }, lamda_valid},
    cudf::duplicate_keep_option::KEEP_LAST);

  TypedDropDuplicatesTest<TypeParam>(
    col1,
    col2,
    cudf::test::column_wrapper<typename TypeParam::Type0>{0, lamda_type0, lamda_valid},
    cudf::test::column_wrapper<typename TypeParam::Type1>{0, lamda_type1, lamda_valid},
    cudf::duplicate_keep_option::KEEP_NONE);
}

