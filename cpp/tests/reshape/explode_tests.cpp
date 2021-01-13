/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/reshape.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf_test/table_utilities.hpp>

using namespace cudf::test;

class ExplodeTest : public cudf::test::BaseFixture {
};

TEST_F(ExplodeTest, Empty)
{
  lists_column_wrapper<int32_t> a{};
  fixed_width_column_wrapper<int32_t> b{};

  cudf::table_view t({a, b});

  auto ret = cudf::explode(t, 0);

  fixed_width_column_wrapper<int32_t> expected_a{};
  fixed_width_column_wrapper<int32_t> expected_b{};
  cudf::table_view expected({expected_a, expected_b});

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);
}

TEST_F(ExplodeTest, NonList)
{
  fixed_width_column_wrapper<int32_t> a{100, 200, 300};
  fixed_width_column_wrapper<int32_t> b{100, 200, 300};

  cudf::table_view t({a, b});

  EXPECT_THROW(cudf::explode(t, 1), cudf::logic_error);
}

TEST_F(ExplodeTest, Basics)
{
  /*
      a                   b
      [1, 2, 7]           100
      [5, 6]              200
      [0, 3]              300
  */

  lists_column_wrapper<int32_t> a{lists_column_wrapper<int32_t>{1, 2, 7},
                                  lists_column_wrapper<int32_t>{5, 6},
                                  lists_column_wrapper<int32_t>{0, 3}};
  fixed_width_column_wrapper<int32_t> b{100, 200, 300};

  fixed_width_column_wrapper<int32_t> expected_a{1, 2, 7, 5, 6, 0, 3};
  fixed_width_column_wrapper<int32_t> expected_b{100, 100, 100, 200, 200, 300, 300};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);
}

TEST_F(ExplodeTest, Nested)
{
  /*
      a                   b
      [[1, 2], [7, 6, 5]] 100
      [[5, 6]]            200
      [[0, 3],[5],[2, 1]] 300
  */

  lists_column_wrapper<int32_t> a{
    lists_column_wrapper<int32_t>{lists_column_wrapper<int32_t>{1, 2},
                                  lists_column_wrapper<int32_t>{7, 6, 5}},
    lists_column_wrapper<int32_t>{lists_column_wrapper<int32_t>{5, 6}},
    lists_column_wrapper<int32_t>{lists_column_wrapper<int32_t>{0, 3},
                                  lists_column_wrapper<int32_t>{5},
                                  lists_column_wrapper<int32_t>{2, 1}}};
  fixed_width_column_wrapper<int32_t> b{100, 200, 300};

  lists_column_wrapper<int32_t> expected_a{lists_column_wrapper<int32_t>{1, 2},
                                           lists_column_wrapper<int32_t>{7, 6, 5},
                                           lists_column_wrapper<int32_t>{5, 6},
                                           lists_column_wrapper<int32_t>{0, 3},
                                           lists_column_wrapper<int32_t>{5},
                                           lists_column_wrapper<int32_t>{2, 1}};
  fixed_width_column_wrapper<int32_t> expected_b{100, 100, 200, 300, 300, 300};

  cudf::table_view t({a, b});
  cudf::table_view expected({expected_a, expected_b});

  auto ret = cudf::explode(t, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(ret->view(), expected);
}
