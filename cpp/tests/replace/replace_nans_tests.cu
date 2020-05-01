/*
 * Copyright 2019, NVIDIA CORPORATION.
 *
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <cristhian@blazingdb.com>
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

#include <cudf/replace.hpp>

#include <cudf/cudf.h>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

struct ReplaceNaNsErrorTest : public cudf::test::BaseFixture {
};

// Error: old-values and new-values size mismatch
TEST_F(ReplaceNaNsErrorTest, SizeMismatch)
{
  cudf::test::fixed_width_column_wrapper<float> input_column{7, 5, 6, 3, 1, 8, 4};
  cudf::test::fixed_width_column_wrapper<float> replacement_column{{10, 11, 12, 13}};

  EXPECT_THROW(cudf::experimental::replace_nans(input_column, replacement_column),
               cudf::logic_error);
}

// Error : column type mismatch
TEST_F(ReplaceNaNsErrorTest, TypeMismatch)
{
  cudf::test::fixed_width_column_wrapper<float> input_column{7, 5, 6, 3, 1, 2, 8, 4};
  cudf::test::fixed_width_column_wrapper<double> replacement_column{10, 11, 12, 13, 14, 15, 16, 17};

  EXPECT_THROW(cudf::experimental::replace_nans(input_column, replacement_column),
               cudf::logic_error);
}

// Error: column type mismatch
TEST_F(ReplaceNaNsErrorTest, TypeMismatchScalar)
{
  cudf::test::fixed_width_column_wrapper<double> input_column{7, 5, 6, 3, 1, 2, 8, 4};
  cudf::numeric_scalar<float> replacement(1);

  EXPECT_THROW(cudf::experimental::replace_nans(input_column, replacement), cudf::logic_error);
}

// Error: column type mismatch
TEST_F(ReplaceNaNsErrorTest, NonFloatType)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_column{7, 5, 6, 3, 1, 2, 8, 4};
  cudf::numeric_scalar<float> replacement(1);

  EXPECT_THROW(cudf::experimental::replace_nans(input_column, replacement), cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
