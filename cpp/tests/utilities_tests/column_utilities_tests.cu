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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

template <typename T>
struct ColumnUtilitiesTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(ColumnUtilitiesTest, cudf::test::NumericTypes);

TYPED_TEST(ColumnUtilitiesTest, First) { ASSERT_TRUE(true); }

TYPED_TEST(ColumnUtilitiesTest, PrintColumn) {
  const char *delimiter = ",";

  cudf::test::fixed_width_column_wrapper<TypeParam> cudf_col({1, 2, 3, 4, 5});
  std::vector<TypeParam>                            std_col({1, 2, 3, 4, 5});

  std::ostringstream tmp;
  std::copy(std_col.begin(), std_col.end(), std::ostream_iterator<TypeParam>(tmp, delimiter));

  EXPECT_EQ(tmp.str(), cudf::test::column_view_to_str(cudf_col, delimiter));
}

