/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct StringsFindMultipleTest : public cudf::test::BaseFixture {};

TEST_F(StringsFindMultipleTest, FindMultiple)
{
  std::vector<char const*> h_strings{"Héllo", "thesé", nullptr, "lease", "test strings", ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<char const*> h_targets{"é", "a", "e", "i", "o", "u", "es"};
  cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());
  auto targets_view = cudf::strings_column_view(targets);

  auto results = cudf::strings::find_multiple(strings_view, targets_view);

  using LCW = cudf::test::lists_column_wrapper<int32_t>;
  LCW expected({LCW{1, -1, -1, -1, 4, -1, -1},
                LCW{4, -1, 2, -1, -1, -1, 2},
                LCW{-1, -1, -1, -1, -1, -1, -1},
                LCW{-1, 2, 1, -1, -1, -1, -1},
                LCW{-1, -1, 1, 8, -1, -1, 1},
                LCW{-1, -1, -1, -1, -1, -1, -1}});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsFindMultipleTest, ZeroSizeStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  auto strings_view                   = cudf::strings_column_view(zero_size_strings_column);
  std::vector<char const*> h_targets{""};
  cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());
  auto targets_view = cudf::strings_column_view(targets);

  auto results = cudf::strings::find_multiple(strings_view, targets_view);
  EXPECT_EQ(results->size(), 0);
}

TEST_F(StringsFindMultipleTest, ErrorTest)
{
  cudf::test::strings_column_wrapper strings({"this string intentionally left blank"}, {false});
  auto strings_view = cudf::strings_column_view(strings);

  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  auto empty_view                     = cudf::strings_column_view(zero_size_strings_column);
  // targets must have at least one string
  EXPECT_THROW(cudf::strings::find_multiple(strings_view, empty_view), cudf::logic_error);

  // targets cannot have nulls
  EXPECT_THROW(cudf::strings::find_multiple(strings_view, strings_view), cudf::logic_error);
}
