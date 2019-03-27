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

#include "stream_compaction.hpp"

#include <utilities/error_utils.hpp>

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
//#include "tests/utilities/cudf_test_utils.cuh"

#include <random>

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

TYPED_TEST(ApplyBooleanMaskTest, Identity)
{
  constexpr gdf_size_type column_size{10};

  cudf::test::column_wrapper<TypeParam> source{
    column_size,
    [](gdf_index_type row) { return row; },
    [](gdf_index_type row) { return true; }};

  cudf::test::column_wrapper<TypeParam> expected{source};

  cudf::test::column_wrapper<cudf::bool8> bool_mask{
    column_size,
    [](gdf_index_type row) { return cudf::bool8{true}; },
    [](gdf_index_type row) { return true; }};

  gdf_column result;
  EXPECT_NO_THROW(result = cudf::apply_boolean_mask(source, bool_mask));

  EXPECT_TRUE(expected == result);

  gdf_column_free(&result);
}

TYPED_TEST(ApplyBooleanMaskTest, AllFalse)
{
  constexpr gdf_size_type column_size{10};

  cudf::test::column_wrapper<TypeParam> source{
    column_size,
    [](gdf_index_type row) { return row; },
    [](gdf_index_type row) { return true; }};

  cudf::test::column_wrapper<cudf::bool8> bool_mask{
    column_size,
    [](gdf_index_type row) { return cudf::bool8{false}; },
    [](gdf_index_type row) { return true; }};

  gdf_column result;
  EXPECT_NO_THROW(result = cudf::apply_boolean_mask(source, bool_mask));

  EXPECT_TRUE(result.size == 0);
  EXPECT_TRUE(result.data == 0);
  EXPECT_TRUE(result.null_count == 0);
  EXPECT_TRUE(result.valid == 0);

  gdf_column_free(&result);
}