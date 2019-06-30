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
#include <cudf/copying.hpp>

#include <utilities/error_utils.hpp>

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/column_wrapper_factory.hpp>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>

#include <sstream>

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

struct ApplyBooleanMaskErrorTest : GdfTest {};

// Test ill-formed inputs

TEST_F(ApplyBooleanMaskErrorTest, NullPtrs)
{
  constexpr gdf_size_type column_size{1000};

  gdf_column bad_input, bad_mask;
  gdf_column_view(&bad_input, 0, 0, 0, GDF_INT32);
  gdf_column_view(&bad_mask,  0, 0, 0, GDF_BOOL8);

  column_wrapper<int32_t> source(column_size);
  column_wrapper<cudf::bool8> mask(column_size);

  {
    column_wrapper<int32_t> empty_column(cudf::empty_like(bad_input));
    gdf_column output;
    CUDF_EXPECT_NO_THROW(output = cudf::apply_boolean_mask(bad_input, mask));
    EXPECT_TRUE(empty_column == output);
  }

  bad_input.valid = reinterpret_cast<gdf_valid_type*>(0x0badf00d);
  bad_input.null_count = 2;
  bad_input.size = column_size; 
  // nonzero, with non-null valid mask, so non-null input expected
  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(bad_input, mask),
                            "Null input data");

  {
    column_wrapper<int32_t> empty_column(cudf::empty_like(source));
    gdf_column output;
    CUDF_EXPECT_NO_THROW(output = cudf::apply_boolean_mask(source, bad_mask));
    EXPECT_TRUE(empty_column == output);
  }

  // null mask pointers but non-zero mask size
  bad_mask.size = column_size;
  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(source, bad_mask),
                            "Null boolean_mask");
}

TEST_F(ApplyBooleanMaskErrorTest, SizeMismatch)
{
  constexpr gdf_size_type column_size{1000};
  constexpr gdf_size_type mask_size{500};

  column_wrapper<int32_t> source(column_size);
  column_wrapper<cudf::bool8> mask(mask_size);
             
  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(source, mask), 
                            "Column size mismatch");
}

TEST_F(ApplyBooleanMaskErrorTest, NonBooleanMask)
{
  constexpr gdf_size_type column_size{1000};

  column_wrapper<int32_t> source(column_size);
  column_wrapper<float> nonbool_mask(column_size);

  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(source, nonbool_mask), 
                            "Mask must be Boolean type");

  column_wrapper<cudf::bool8> bool_mask(column_size, true);
  EXPECT_NO_THROW(cudf::apply_boolean_mask(source, bool_mask));
}

template <typename T>
struct ApplyBooleanMaskTest : GdfTest
{
  cudf::test::column_wrapper_factory<T> factory;
};

using test_types =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double,
                     cudf::bool8, cudf::nvstring_category>;
TYPED_TEST_CASE(ApplyBooleanMaskTest, test_types);

// Test computation

/*
 * Runs apply_boolean_mask checking for errors, and compares the result column 
 * to the specified expected result column.
 */
template <typename T>
void BooleanMaskTest(column_wrapper<T> const& source,
                     column_wrapper<cudf::bool8> const& mask,
                     column_wrapper<T> const& expected)
{
  gdf_column result{};
  EXPECT_NO_THROW(result = cudf::apply_boolean_mask(source, mask));

  EXPECT_TRUE(expected == result);

  if (!(expected == result)) {
    std::cout << "expected\n";
    expected.print();
    std::cout << expected.get()->null_count << "\n";
    std::cout << "result\n";
    print_gdf_column(&result);
    std::cout << result.null_count << "\n";
  }

  gdf_column_free(&result);
}

constexpr gdf_size_type column_size{100000};

TYPED_TEST(ApplyBooleanMaskTest, Identity)
{
  BooleanMaskTest<TypeParam>(
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }),
    column_wrapper<cudf::bool8>(column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return true; }),
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }));
}

TYPED_TEST(ApplyBooleanMaskTest, MaskAllFalse)
{
  BooleanMaskTest<TypeParam>(
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }),
    column_wrapper<cudf::bool8>(column_size,
      [](gdf_index_type row) { return cudf::bool8{false}; },
      [](gdf_index_type row) { return true; }),
    column_wrapper<TypeParam>(0, false));
}

TYPED_TEST(ApplyBooleanMaskTest, MaskAllNull)
{
  BooleanMaskTest<TypeParam>(
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }),
    column_wrapper<cudf::bool8>(column_size, 
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return false; }),
    column_wrapper<TypeParam>(0, false));
}

TYPED_TEST(ApplyBooleanMaskTest, MaskEvensFalse)
{
  BooleanMaskTest<TypeParam>(
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }),
    column_wrapper<cudf::bool8>(column_size,
      [](gdf_index_type row) { return cudf::bool8{row % 2 == 1}; },
      [](gdf_index_type row) { return true; }),
    this->factory.make(column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1; },
      [](gdf_index_type row) { return true; }));
}

TYPED_TEST(ApplyBooleanMaskTest, MaskEvensNull)
{
  // mix it up a bit by setting the input odd values to be null
  // Since the bool mask has even values null, the output
  // vector should have all values nulled

  BooleanMaskTest<TypeParam>(
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return row % 2 == 0; }),
    column_wrapper<cudf::bool8>(column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return row % 2 == 1; }),
    this->factory.make(column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1; },
      [](gdf_index_type row) { return false; }));
}

TYPED_TEST(ApplyBooleanMaskTest, NonalignedGap)
{
  const int start{1}, end{column_size / 4};

  BooleanMaskTest<TypeParam>(
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }),
    column_wrapper<cudf::bool8>(column_size,
      [](gdf_index_type row) { return cudf::bool8{(row < start) || (row >= end)}; },
      [](gdf_index_type row) { return true; }),
    this->factory.make(column_size - (end - start),
      [](gdf_index_type row) { 
        return (row < start) ? row : row + end - start; 
      },
      [](gdf_index_type row) { return true; }));
}

TYPED_TEST(ApplyBooleanMaskTest, NoNullMask)
{
  BooleanMaskTest<TypeParam>(
    this->factory.make(column_size, 
      [](gdf_index_type row) { return row; }),
    column_wrapper<cudf::bool8>(column_size,
      [](gdf_index_type row) { return cudf::bool8{true}; },
      [](gdf_index_type row) { return row % 2 == 1; }),
     this->factory.make(column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1; }));
}

struct DropNullsErrorTest : GdfTest {};

TEST_F(DropNullsErrorTest, EmptyInput)
{
  gdf_column bad_input{};
  gdf_column_view(&bad_input, 0, 0, 0, GDF_INT32);

  // zero size, so expect no error, just empty output column
  gdf_column output{};
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
void DropNulls(column_wrapper<T> const& source,
               column_wrapper<T> const& expected)
{
  gdf_column result{};
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
struct DropNullsTest : GdfTest 
{
  cudf::test::column_wrapper_factory<T> factory;
};

TYPED_TEST_CASE(DropNullsTest, test_types);

TYPED_TEST(DropNullsTest, Identity)
{
  auto col = this->factory.make(column_size,
    [](gdf_index_type row) { return row; },
    [](gdf_index_type row) { return true; });
  DropNulls<TypeParam>(col, col);
}

TYPED_TEST(DropNullsTest, AllNull)
{
  DropNulls<TypeParam>(
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return false; }),
    column_wrapper<TypeParam>(0, false));
}

TYPED_TEST(DropNullsTest, EvensNull)
{
  DropNulls<TypeParam>(
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return row % 2 == 1; }),
    this->factory.make(column_size / 2,
      [](gdf_index_type row) { return 2 * row + 1; },
      [](gdf_index_type row) { return true; }));
}

TYPED_TEST(DropNullsTest, NonalignedGap)
{
  const int start{1}, end{column_size / 4};

  DropNulls<TypeParam>(
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return (row < start) || (row >= end); }),
    this->factory.make(column_size - (end - start),
      [](gdf_index_type row) { 
        return (row < start) ? row : row + end - start;
      },
      [](gdf_index_type row) { return true; }));
}

TYPED_TEST(DropNullsTest, NoNullMask)
{
  DropNulls<TypeParam>(
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; }),
    this->factory.make(column_size,
      [](gdf_index_type row) { return row; }));
}
