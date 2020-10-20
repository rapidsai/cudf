/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

// #include <cudf/column/column.hpp>
// #include <cudf/column/column_view.hpp>
#include <cudf/round.hpp>
// #include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>
// #include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
// #include <cudf_test/type_lists.hpp>

struct RoundTests : public cudf::test::BaseFixture {
};

TEST_F(RoundTests, SimpleTest)
{
  using namespace numeric;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<int32_t>;

  auto const col      = fp_wrapper{{1140, 1150, 1160}, scale_type{-3}};
  auto const expected = fp_wrapper{{11, 12, 12}, scale_type{-1}};
  auto const result   = cudf::round(col, 1, cudf::round_option::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}
