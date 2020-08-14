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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <vector>

struct StringsFindMultipleTest : public cudf::test::BaseFixture {
};

TEST_F(StringsFindMultipleTest, FindMultiple)
{
  std::vector<const char*> h_strings{"Héllo", "thesé", nullptr, "lease", "test strings", ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<const char*> h_targets{"é", "a", "e", "i", "o", "u", "es"};
  cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());
  auto targets_view = cudf::strings_column_view(targets);

  auto results                = cudf::strings::find_multiple(strings_view, targets_view);
  cudf::size_type total_count = static_cast<cudf::size_type>(h_strings.size() * h_targets.size());
  EXPECT_EQ(total_count, results->size());

  cudf::test::fixed_width_column_wrapper<int32_t> expected(
    {1,  -1, -1, -1, 4,  -1, -1, 4,  -1, 2, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1,
     -1, 2,  1,  -1, -1, -1, -1, -1, -1, 1, 8,  -1, -1, 1, -1, -1, -1, -1, -1, -1, -1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsFindMultipleTest, ZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  std::vector<const char*> h_targets{""};
  cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());
  auto targets_view = cudf::strings_column_view(targets);

  auto results = cudf::strings::find_multiple(strings_view, targets_view);
  EXPECT_EQ(results->size(), 0);
}

TEST_F(StringsFindMultipleTest, ErrorTest)
{
  cudf::test::strings_column_wrapper strings({"this string intentionally left blank"}, {0});
  auto strings_view = cudf::strings_column_view(strings);

  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto empty_view = cudf::strings_column_view(zero_size_strings_column);
  // targets must have at least one string
  EXPECT_THROW(cudf::strings::find_multiple(strings_view, empty_view), cudf::logic_error);

  // targets cannot have nulls
  EXPECT_THROW(cudf::strings::find_multiple(strings_view, strings_view), cudf::logic_error);
}
