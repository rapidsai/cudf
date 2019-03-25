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

struct ApplyBooleanMaskBasicTest : GdfTest {};

TEST_F(ApplyBooleanMaskBasicTest, NullPtrs)
{
  constexpr gdf_size_type column_size{1000};

  cudf::test::column_wrapper<int32_t> source{column_size};
  cudf::test::column_wrapper<gdf_bool> mask{column_size};
             
  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(nullptr, mask), 
                            "Null input");

  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(source, nullptr),
                            "Null boolean_mask");
}

TEST_F(ApplyBooleanMaskBasicTest, SizeMismatch)
{
  constexpr gdf_size_type column_size{1000};
  constexpr gdf_size_type mask_size{500};

  cudf::test::column_wrapper<int32_t> source{column_size};
  cudf::test::column_wrapper<gdf_bool> mask{mask_size};
             
  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(source, mask), 
                            "Column size mismatch");
}

TEST_F(ApplyBooleanMaskBasicTest, NonBooleanMask)
{
  constexpr gdf_size_type column_size{1000};
  constexpr gdf_size_type mask_size{1000};

  cudf::test::column_wrapper<int32_t> source{column_size};
  cudf::test::column_wrapper<float> nonbool_mask{mask_size};
             
  CUDF_EXPECT_THROW_MESSAGE(cudf::apply_boolean_mask(source, nonbool_mask), 
                            "Mask must be Boolean type");

  //cudf::test::column_wrapper<gdf_bool> bool_mask{mask_size};
  //EXPECT_NO_THROW(cudf::apply_boolean_mask(source, bool_mask));
}

template <typename T>
struct ApplyBooleanMaskTest : GdfTest {};

using test_types =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;
TYPED_TEST_CASE(ApplyBooleanMaskTest, test_types);
