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
#include <tests/strings/utilities.h>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

class GatherTestStr : public cudf::test::BaseFixture {
};

TEST_F(GatherTestStr, StringColumn)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{1, 2, 3, 4, 5, 6}, {1, 1, 0, 1, 0, 1}};
  cudf::test::strings_column_wrapper col2{{"This", "is", "not", "a", "string", "type"},
                                          {1, 1, 1, 1, 1, 0}};
  cudf::table_view source_table{{col1, col2}};

  cudf::test::fixed_width_column_wrapper<int16_t> gather_map{{0, 1, 3, 4}};

  cudf::test::fixed_width_column_wrapper<int16_t> exp_col1{{1, 2, 4, 5}, {1, 1, 1, 0}};
  cudf::test::strings_column_wrapper exp_col2{{"This", "is", "a", "string"}, {1, 1, 1, 1}};
  cudf::table_view expected{{exp_col1, exp_col2}};

  auto got = cudf::gather(source_table, gather_map);

  cudf::test::expect_tables_equal(expected, got->view());
}

TEST_F(GatherTestStr, Gather)
{
  std::vector<const char*> h_strings{"eee", "bb", "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  cudf::table_view source_table({strings});

  std::vector<int32_t> h_map{4, 1, 5, 2, 7};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(h_map.begin(), h_map.end());
  auto results = cudf::detail::gather(source_table,
                                      gather_map,
                                      cudf::detail::out_of_bounds_policy::IGNORE,
                                      cudf::detail::negative_index_policy::NOT_ALLOWED);

  std::vector<const char*> h_expected;
  std::vector<int32_t> expected_validity;
  for (auto itr = h_map.begin(); itr != h_map.end(); ++itr) {
    auto index = *itr;
    if ((0 <= index) && (index < static_cast<decltype(index)>(h_strings.size()))) {
      h_expected.push_back(h_strings[index]);
      expected_validity.push_back(1);
    } else {
      h_expected.push_back("");
      expected_validity.push_back(0);
    }
  }
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), expected_validity.begin());
  cudf::test::expect_columns_equal(results->view().column(0), expected);
}

TEST_F(GatherTestStr, GatherIgnoreOutOfBounds)
{
  std::vector<const char*> h_strings{"eee", "bb", "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  cudf::table_view source_table({strings});

  std::vector<int32_t> h_map{3, 4, 0, 0};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(h_map.begin(), h_map.end());
  auto results = cudf::detail::gather(source_table,
                                      gather_map,
                                      cudf::detail::out_of_bounds_policy::IGNORE,
                                      cudf::detail::negative_index_policy::NOT_ALLOWED);

  std::vector<const char*> h_expected;
  std::vector<int32_t> expected_validity;
  for (auto itr = h_map.begin(); itr != h_map.end(); ++itr) {
    h_expected.push_back(h_strings[*itr]);
    expected_validity.push_back(1);
  }
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), expected_validity.begin());
  cudf::test::expect_columns_equal(results->view().column(0), expected);
}

TEST_F(GatherTestStr, GatherZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  rmm::device_vector<cudf::size_type> gather_map{};
  auto results = cudf::detail::gather(
    cudf::table_view({zero_size_strings_column}), gather_map.begin(), gather_map.end(), true);
  cudf::test::expect_strings_empty(results->get_column(0).view());
}
