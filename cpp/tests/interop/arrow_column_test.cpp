/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <tests/interop/arrow_utils.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>

#include <cudf/filling.hpp>
#include <cudf/interop.hpp>
#include <cudf/types.hpp>

#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>

struct ArrowColumnTest : public cudf::test::BaseFixture {};

TEST_F(ArrowColumnTest, TwoWayConversion)
{
  cudf::test::fixed_width_column_wrapper<int32_t> int_col{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  // This column will be moved into the arrow_column, making it invalid to
  // access, but the original int_col stays valid for the remainder of the test
  // scope for comparison.
  auto col = cudf::column(int_col);

  auto arrow_column_from_cudf_column = cudf::arrow_column(std::move(col));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col, *arrow_column_from_cudf_column.view());

  // Now we can extract an ArrowDeviceArray from the arrow_column
  ArrowSchema arrow_schema_from_cudf_column;
  arrow_column_from_cudf_column.to_arrow_schema(&arrow_schema_from_cudf_column);
  ArrowDeviceArray arrow_array_from_arrow_column;
  arrow_column_from_cudf_column.to_arrow(&arrow_array_from_arrow_column, ARROW_DEVICE_CUDA);

  // Now let's convert it back to an arrow_column
  auto arrow_column_from_arrow_array =
    cudf::arrow_column(&arrow_schema_from_cudf_column, &arrow_array_from_arrow_column);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col, *arrow_column_from_arrow_array.view());
}
