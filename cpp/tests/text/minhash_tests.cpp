/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <nvtext/minhash.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <vector>

struct MinHashTest : public cudf::test::BaseFixture {
};

TEST_F(MinHashTest, Basic)
{
  auto input = cudf::test::strings_column_wrapper({"doc 1", "", "this is doc 2", "", "doc 3", "d"},
                                                  {1, 0, 1, 1, 1, 1});

  auto view = cudf::strings_column_view(input);

  auto results = nvtext::minhash(view);

  auto expected = cudf::test::fixed_width_column_wrapper<cudf::hash_value_type>(
    {1207251914u, 0u, 21141582u, 0u, 1207251914u, 655955059u}, {1, 0, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(MinHashTest, EmptyTest)
{
  auto input   = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  auto view    = cudf::strings_column_view(input->view());
  auto results = nvtext::minhash(view);
  EXPECT_EQ(results->size(), 0);
}

TEST_F(MinHashTest, ErrorsTest)
{
  auto input = cudf::test::strings_column_wrapper({"pup"});
  EXPECT_THROW(nvtext::minhash(cudf::strings_column_view(input), 0), cudf::logic_error);
}
