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

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>

#include <thrust/iterator/transform_iterator.h>

struct DictionaryFactoriesTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryFactoriesTest, CreateFromColumnViews)
{
  cudf::test::strings_column_wrapper keys({"aaa", "ccc", "ddd", "www"});
  cudf::test::fixed_width_column_wrapper<int32_t> values{2, 0, 3, 1, 2, 2, 2, 3, 0};

  auto dictionary = cudf::make_dictionary_column(keys, values);
  cudf::dictionary_column_view view(dictionary->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.keys(), keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), values);
}

TEST_F(DictionaryFactoriesTest, ColumnViewsWithNulls)
{
  cudf::test::fixed_width_column_wrapper<float> keys{-11.75, 4.25, 7.125, 0.5, 12.0};
  std::vector<int32_t> h_values{1, 3, 2, 0, 1, 4, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> indices(
    h_values.begin(), h_values.end(), thrust::make_transform_iterator(h_values.begin(), [](auto v) {
      return v > 0;
    }));
  auto dictionary = cudf::make_dictionary_column(keys, indices);
  cudf::dictionary_column_view view(dictionary->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.keys(), keys);
  cudf::test::fixed_width_column_wrapper<int32_t> values_expected(h_values.begin(), h_values.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), values_expected);
}

TEST_F(DictionaryFactoriesTest, CreateFromColumns)
{
  std::vector<std::string> h_keys{"pear", "apple", "fruit", "macintosh"};
  cudf::test::strings_column_wrapper keys(h_keys.begin(), h_keys.end());
  std::vector<int32_t> h_values{1, 2, 3, 1, 2, 3, 0};
  cudf::test::fixed_width_column_wrapper<int32_t> values(h_values.begin(), h_values.end());

  auto dictionary =
    cudf::make_dictionary_column(keys.release(), values.release(), rmm::device_buffer{}, 0);
  cudf::dictionary_column_view view(dictionary->view());

  cudf::test::strings_column_wrapper keys_expected(h_keys.begin(), h_keys.end());
  cudf::test::fixed_width_column_wrapper<int32_t> values_expected(h_values.begin(), h_values.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.keys(), keys_expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), values_expected);
}

TEST_F(DictionaryFactoriesTest, ColumnsWithNulls)
{
  std::vector<int64_t> h_keys{-1234567890, -987654321, 0, 19283714};
  cudf::test::fixed_width_column_wrapper<int64_t> keys(h_keys.begin(), h_keys.end());
  std::vector<int32_t> h_values{1, 2, 3, 1, 2, 3, 0};
  cudf::test::fixed_width_column_wrapper<int32_t> values(h_values.begin(), h_values.end());
  auto size                    = static_cast<cudf::size_type>(h_values.size());
  rmm::device_buffer null_mask = create_null_mask(size, cudf::mask_state::ALL_NULL);
  auto dictionary =
    cudf::make_dictionary_column(keys.release(), values.release(), std::move(null_mask), size);
  cudf::dictionary_column_view view(dictionary->view());
  EXPECT_EQ(size, view.size());
  EXPECT_EQ(size, view.null_count());

  cudf::test::fixed_width_column_wrapper<int64_t> keys_expected(h_keys.begin(), h_keys.end());
  cudf::test::fixed_width_column_wrapper<int32_t> values_expected(h_values.begin(), h_values.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.keys(), keys_expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), values_expected);
}

TEST_F(DictionaryFactoriesTest, KeysWithNulls)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{{0, 1, 2, 3, 4},
                                                       {true, true, true, false, true}};
  cudf::test::fixed_width_column_wrapper<int32_t> indices{5, 4, 3, 2, 1, 0};
  EXPECT_THROW(cudf::make_dictionary_column(keys, indices), cudf::logic_error);
}

TEST_F(DictionaryFactoriesTest, IndicesWithNulls)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{0, 1, 2, 3, 4};
  cudf::test::fixed_width_column_wrapper<int32_t> indices{{5, 4, 3, 2, 1, 0},
                                                          {true, true, true, false, true, false}};
  EXPECT_THROW(
    cudf::make_dictionary_column(keys.release(), indices.release(), rmm::device_buffer{}, 0),
    cudf::logic_error);
}

TEST_F(DictionaryFactoriesTest, InvalidIndices)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{0, 1, 2, 3, 4};
  cudf::test::fixed_width_column_wrapper<uint16_t> indices{5, 4, 3, 2, 1, 0};
  EXPECT_THROW(cudf::make_dictionary_column(keys, indices), cudf::logic_error);
  EXPECT_THROW(
    cudf::make_dictionary_column(keys.release(), indices.release(), rmm::device_buffer{}, 0),
    cudf::logic_error);
}
