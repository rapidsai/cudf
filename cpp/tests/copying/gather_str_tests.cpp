/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

class GatherTestStr : public cudf::test::BaseFixture {};

TEST_F(GatherTestStr, StringColumn)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{1, 2, 3, 4, 5, 6},
                                                       {true, true, false, true, false, true}};
  cudf::test::strings_column_wrapper col2{{"This", "is", "not", "a", "string", "type"},
                                          {true, true, true, true, true, false}};
  cudf::table_view source_table{{col1, col2}};

  cudf::test::fixed_width_column_wrapper<int16_t> gather_map{{0, 1, 3, 4}};

  cudf::test::fixed_width_column_wrapper<int16_t> exp_col1{{1, 2, 4, 5}, {true, true, true, false}};
  cudf::test::strings_column_wrapper exp_col2{{"This", "is", "a", "string"},
                                              {true, true, true, true}};
  cudf::table_view expected{{exp_col1, exp_col2}};

  auto got = cudf::gather(source_table, gather_map);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(GatherTestStr, GatherSlicedStringsColumn)
{
  cudf::test::strings_column_wrapper strings{{"This", "is", "not", "a", "string", "type"},
                                             {true, true, true, true, true, false}};
  std::vector<cudf::size_type> slice_indices{0, 2, 2, 3, 3, 6};
  auto sliced_strings = cudf::slice(strings, slice_indices);
  {
    cudf::test::fixed_width_column_wrapper<int16_t> gather_map{{1, 0, 1}};
    cudf::test::strings_column_wrapper expected_strings{{"is", "This", "is"}, {true, true, true}};
    cudf::table_view expected{{expected_strings}};
    auto result = cudf::gather(cudf::table_view{{sliced_strings[0]}}, gather_map);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view());
  }
  {
    cudf::test::fixed_width_column_wrapper<int16_t> gather_map{{0, 0, 0}};
    cudf::test::strings_column_wrapper expected_strings{{"not", "not", "not"}, {true, true, true}};
    cudf::table_view expected{{expected_strings}};
    auto result = cudf::gather(cudf::table_view{{sliced_strings[1]}}, gather_map);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view());
  }
  {
    cudf::test::fixed_width_column_wrapper<int16_t> gather_map{{2, 1, 0}};
    cudf::test::strings_column_wrapper expected_strings{{"", "string", "a"}, {false, true, true}};
    cudf::table_view expected{{expected_strings}};
    auto result = cudf::gather(cudf::table_view{{sliced_strings[2]}}, gather_map);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view());
  }
}

TEST_F(GatherTestStr, Gather)
{
  std::vector<char const*> h_strings{"eee", "bb", "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  cudf::table_view source_table({strings});

  std::vector<int32_t> h_map{4, 1, 5, 2, 7};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(h_map.begin(), h_map.end());
  auto results = cudf::detail::gather(source_table,
                                      gather_map,
                                      cudf::out_of_bounds_policy::NULLIFY,
                                      cudf::detail::negative_index_policy::NOT_ALLOWED,
                                      cudf::get_default_stream(),
                                      cudf::get_current_device_resource_ref());

  std::vector<char const*> h_expected;
  std::vector<int32_t> expected_validity;
  for (int index : h_map) {
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
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
}

TEST_F(GatherTestStr, GatherDontCheckOutOfBounds)
{
  std::vector<char const*> h_strings{"eee", "bb", "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  cudf::table_view source_table({strings});

  std::vector<int32_t> h_map{3, 4, 0, 0};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(h_map.begin(), h_map.end());
  auto results = cudf::detail::gather(source_table,
                                      gather_map,
                                      cudf::out_of_bounds_policy::DONT_CHECK,
                                      cudf::detail::negative_index_policy::NOT_ALLOWED,
                                      cudf::get_default_stream(),
                                      cudf::get_current_device_resource_ref());

  std::vector<char const*> h_expected;
  for (int itr : h_map) {
    h_expected.push_back(h_strings[itr]);
  }
  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
}

TEST_F(GatherTestStr, GatherEmptyMapStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map;
  auto results = cudf::detail::gather(cudf::table_view({zero_size_strings_column->view()}),
                                      gather_map,
                                      cudf::out_of_bounds_policy::NULLIFY,
                                      cudf::detail::negative_index_policy::NOT_ALLOWED,
                                      cudf::get_default_stream(),
                                      cudf::get_current_device_resource_ref());
  cudf::test::expect_column_empty(results->get_column(0).view());
}

TEST_F(GatherTestStr, GatherZeroSizeStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING);
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map({0});
  cudf::test::strings_column_wrapper expected{std::pair<std::string, bool>{"", false}};
  auto results = cudf::detail::gather(cudf::table_view({zero_size_strings_column->view()}),
                                      gather_map,
                                      cudf::out_of_bounds_policy::NULLIFY,
                                      cudf::detail::negative_index_policy::NOT_ALLOWED,
                                      cudf::get_default_stream(),
                                      cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results->get_column(0).view());
}
