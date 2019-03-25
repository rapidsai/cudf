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

/*#include "stream_compaction.hpp"

#include "tests/utilities/column_wrapper.cuh"
#include "tests/utilities/cudf_test_fixtures.h"
#include "tests/utilities/cudf_test_utils.cuh"

#include <random>

template <typename T>
struct ApplyBooleanMaskTest : GdfTest {};

using test_types =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;
TYPED_TEST_CASE(ApplyBooleanMaskTest, test_types);

TEST_F(ApplyBooleanMaskTest, NullPtrs)
{
  EXPECT_THROW(cudf::apply_boolean_mask(nullptr, nullptr), 
               cudf::logic_error);

  constexpr gdf_size_type column_size{1000};

  cudf::test::column_wrapper<int32_t> source{column_size};
  cudf::test::column_wrapper<bool> mask{column_size};
             
  gdf_column * raw_source = source.get();
  gdf_column * raw_destination = destination.get();

  EXPECT_THROW(cudf::apply_boolean_mask(source, nullptr), 
               cudf::logic_error);
  EXPECT_THROW(cudf::apply_boolean_mask(nullptr, mask), 
               cudf::logic_error);

*/

