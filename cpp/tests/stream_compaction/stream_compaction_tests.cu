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

#include <utilities/error_utils.hpp>

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

struct ApplyBooleanMaskErrorTest : GdfTest {};

// Test ill-formed inputs

TEST_F(ApplyBooleanMaskErrorTest, NullPtrs)
{
  constexpr gdf_size_type column_size{1000};

  cudf::test::column_wrapper<int32_t> source{column_size};
  cudf::test::column_wrapper<gdf_bool> mask{column_size};
             
  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(nullptr, mask), 
                            "Null input");

  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(source, nullptr),
                            "Null boolean_mask");
}

TEST_F(ApplyBooleanMaskErrorTest, SizeMismatch)
{
  constexpr gdf_size_type column_size{1000};
  constexpr gdf_size_type mask_size{500};

  cudf::test::column_wrapper<int32_t> source{column_size};
  cudf::test::column_wrapper<gdf_bool> mask{mask_size};
             
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

template<typename T>
using ValueInitializer = std::function< T (gdf_index_type) >;
using BitInitializer   = ValueInitializer<bool>;


/* Helper function that takes initializer functions / lambdas for the data
 * and valid bits of an input column and a boolean mask column, as well as a
 * initializers for the expected value and bitmask of the output of
 * cudf::apply_boolean_mask. Runs apply_boolean_mask checking for errors,
 * and compares the result column to the expected result column.
 */
template <typename T>
void BooleanMaskTest(gdf_size_type                 in_size,
                     ValueInitializer<T>           in_data,
                     BitInitializer                in_valid,
                     ValueInitializer<cudf::bool8> mask_data,
                     BitInitializer                mask_valid,
                     gdf_size_type                 expected_size,
                     ValueInitializer<T>           expected_data,
                     BitInitializer                expected_valid)
{
  cudf::test::column_wrapper<T> source{in_size, in_data, in_valid};
  cudf::test::column_wrapper<cudf::bool8> mask{in_size, mask_data, mask_valid};
  cudf::test::column_wrapper<T> expected{expected_size, 
                                         expected_data, expected_valid};

  gdf_column result;
  EXPECT_NO_THROW(result = cudf::apply_boolean_mask(source, mask));

  EXPECT_TRUE(expected == result);

  gdf_column_free(&result);
}

TYPED_TEST(ApplyBooleanMaskTest, Identity)
{
  constexpr gdf_size_type column_size{1000};

  BooleanMaskTest<TypeParam>(
    column_size,
    [](gdf_index_type row) { return static_cast<TypeParam>(row); },
    [](gdf_index_type row) { return true; },
    [](gdf_index_type row) { return cudf::bool8{true}; },
    [](gdf_index_type row) { return true; },
    column_size,
    [](gdf_index_type row) { return static_cast<TypeParam>(row); },
    [](gdf_index_type row) { return true; });
}

TYPED_TEST(ApplyBooleanMaskTest, MaskAllFalse)
{
  constexpr gdf_size_type column_size{1000};

  BooleanMaskTest<TypeParam>(
    column_size,
    [](gdf_index_type row) { return static_cast<TypeParam>(row); },
    [](gdf_index_type row) { return true; },
    [](gdf_index_type row) { return cudf::bool8{false}; },
    [](gdf_index_type row) { return true; },
    0,
    [](gdf_index_type row) { return static_cast<TypeParam>(row); },
    [](gdf_index_type row) { return true; });
}

TYPED_TEST(ApplyBooleanMaskTest, MaskAllNull)
{
  constexpr gdf_size_type column_size{1000};

  BooleanMaskTest<TypeParam>(
    column_size,
    [](gdf_index_type row) { return row; },
    [](gdf_index_type row) { return true; },
    [](gdf_index_type row) { return cudf::bool8{true}; },
    [](gdf_index_type row) { return false; },
    0,
    [](gdf_index_type row) { return row; },
    [](gdf_index_type row) { return true; });
}

TYPED_TEST(ApplyBooleanMaskTest, MaskEvensFalse)
{
  constexpr gdf_size_type column_size{1000};

  BooleanMaskTest<TypeParam>(
    column_size,
    [](gdf_index_type row) { return row; },
    [](gdf_index_type row) { return true; },
    [](gdf_index_type row) { return cudf::bool8{row % 2 == 1}; },
    [](gdf_index_type row) { return true; },
    (column_size + 1) / 2,
    [](gdf_index_type row) { return 2 * row + 1;  },
    [](gdf_index_type row) { return true; });
}

TYPED_TEST(ApplyBooleanMaskTest, MaskEvensNull)
{
  constexpr gdf_size_type column_size{1000};

  // mix it up a bit by setting the input odd values to be null
  // Since the bool mask has even values null, the output
  // vector should have all values nulled

  BooleanMaskTest<TypeParam>(
    column_size,
    [](gdf_index_type row) { return row; },
    [](gdf_index_type row) { return row % 2 == 0; },
    [](gdf_index_type row) { return cudf::bool8{true}; },
    [](gdf_index_type row) { return row % 2 == 1; },
    (column_size + 1) / 2,
    [](gdf_index_type row) { return 2 * row + 1;  },
    [](gdf_index_type row) { return false; });
}

