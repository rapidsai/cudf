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

#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/sorting.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <vector>

struct DictionaryGatherTest : public cudf::test::BaseFixture {
};

TEST_F(DictionaryGatherTest, Gather)
{
  cudf::test::strings_column_wrapper strings{
    "eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"};
  auto dictionary = cudf::dictionary::encode(strings);
  cudf::dictionary_column_view view(dictionary->view());

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map{0, 4, 3, 1};
  auto table_result = cudf::gather(cudf::table_view{{view.parent()}}, gather_map)->release();
  auto result       = cudf::dictionary_column_view(table_result.front()->view());

  cudf::test::strings_column_wrapper expected{"eee", "ccc", "bbb", "aaa"};
  auto decoded = cudf::dictionary::decode(result);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());
}

TEST_F(DictionaryGatherTest, GatherWithNulls)
{
  cudf::test::fixed_width_column_wrapper<int64_t> data{{1, 5, 5, 3, 7, 1}, {0, 1, 0, 1, 1, 1}};

  auto dictionary = cudf::dictionary::encode(data);
  cudf::dictionary_column_view view(dictionary->view());

  cudf::test::fixed_width_column_wrapper<int16_t> gather_map{{4, 1, 2, 4}};
  auto table_result = cudf::gather(cudf::table_view{{dictionary->view()}}, gather_map);
  auto result       = cudf::dictionary_column_view(table_result->view().column(0));

  cudf::test::fixed_width_column_wrapper<int64_t> expected{{7, 5, 5, 7}, {1, 1, 0, 1}};
  auto result_decoded = cudf::dictionary::decode(result);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result_decoded->view());
}

TEST_F(DictionaryGatherTest, SortStrings)
{
  std::vector<std::string> h_strings{"eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());

  auto dictionary = cudf::dictionary::encode(strings);
  cudf::dictionary_column_view view(dictionary->view());

  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  auto result = cudf::sort(cudf::table_view{{dictionary->view()}},
                           std::vector<cudf::order>{cudf::order::ASCENDING})
                  ->release();

  std::sort(h_strings.begin(), h_strings.end());
  auto result_decoded = cudf::dictionary::decode(result.front()->view());
  cudf::test::strings_column_wrapper expected(h_strings.begin(), h_strings.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result_decoded->view());
}

TEST_F(DictionaryGatherTest, SortFloat)
{
  std::vector<double> h_data{1.25, -5.75, 8.125, 1e9, 9.7};
  cudf::test::fixed_width_column_wrapper<double> data(h_data.begin(), h_data.end());

  auto dictionary = cudf::dictionary::encode(data);
  cudf::dictionary_column_view view(dictionary->view());

  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  auto result = cudf::sort(cudf::table_view{{dictionary->view()}},
                           std::vector<cudf::order>{cudf::order::ASCENDING})
                  ->release();

  std::sort(h_data.begin(), h_data.end());
  auto result_decoded = cudf::dictionary::decode(result.front()->view());
  cudf::test::fixed_width_column_wrapper<double> expected(h_data.begin(), h_data.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result_decoded->view());
}
