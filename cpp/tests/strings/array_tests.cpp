/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/copying.hpp>
#include <cudf/strings/detail/scatter.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/sorting.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <tests/strings/utilities.h>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <vector>

struct StringsColumnTest : public cudf::test::BaseFixture {
};

TEST_F(StringsColumnTest, Sort)
{
  // cannot initialize std::string with a nullptr so use "<null>" as a place-holder
  cudf::test::strings_column_wrapper h_strings({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"},
                                               {1, 1, 0, 1, 1, 1, 1});
  cudf::test::strings_column_wrapper h_expected({"<null>", "", "aa", "bb", "bbb", "eee", "ééé"},
                                                {0, 1, 1, 1, 1, 1, 1});

  auto strings_view = cudf::strings_column_view(h_strings);
  auto results      = cudf::strings::detail::sort(strings_view, cudf::strings::detail::name);
  cudf::test::expect_columns_equal(*results, h_expected);
}

TEST_F(StringsColumnTest, SortZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  auto results      = cudf::strings::detail::sort(strings_view, cudf::strings::detail::name);
  cudf::test::expect_strings_empty(results->view());
}

class SliceParmsTest : public StringsColumnTest,
                       public testing::WithParamInterface<cudf::size_type> {
};

TEST_P(SliceParmsTest, Slice)
{
  std::vector<const char*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::size_type start = 3;
  cudf::size_type end   = GetParam();
  std::vector<const char*> h_expected;
  if (end > start) {
    for (cudf::size_type idx = start; (idx < end) && (idx < (cudf::size_type)h_strings.size());
         ++idx)
      h_expected.push_back(h_strings[idx]);
  }
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::detail::slice(strings_view, start, end);

  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end());
  // thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_P(SliceParmsTest, SliceAllNulls)
{
  std::vector<const char*> h_strings{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::size_type start = 3;
  cudf::size_type end   = GetParam();
  std::vector<const char*> h_expected;
  if (end > start) {
    for (cudf::size_type idx = start; (idx < end) && (idx < (cudf::size_type)h_strings.size());
         ++idx)
      h_expected.push_back(h_strings[idx]);
  }
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::detail::slice(strings_view, start, end);
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_P(SliceParmsTest, SliceAllEmpty)
{
  std::vector<const char*> h_strings{"", "", "", "", "", "", ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::size_type start = 3;
  cudf::size_type end   = GetParam();
  std::vector<const char*> h_expected;
  if (end > start) {
    for (cudf::size_type idx = start; (idx < end) && (idx < (cudf::size_type)h_strings.size());
         ++idx)
      h_expected.push_back(h_strings[idx]);
  }
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::detail::slice(strings_view, start, end);
  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end());
  // thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
  cudf::test::expect_columns_equal(*results, expected);
}

INSTANTIATE_TEST_CASE_P(SliceParms,
                        SliceParmsTest,
                        testing::ValuesIn(std::array<cudf::size_type, 3>{5, 6, 7}));

TEST_F(StringsColumnTest, SliceZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  auto results      = cudf::strings::detail::slice(strings_view, 1, 2);
  cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsColumnTest, Gather)
{
  std::vector<const char*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map{{4, 1}};
  auto results = cudf::gather(cudf::table_view{{strings}}, gather_map)->release();

  std::vector<const char*> h_expected{"aa", "bb"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  cudf::test::expect_columns_equal(results.front()->view(), expected);
}

TEST_F(StringsColumnTest, GatherZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  cudf::column_view map_view(cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0);
  auto results = cudf::gather(cudf::table_view{{zero_size_strings_column}}, map_view)->release();
  cudf::test::expect_strings_empty(results.front()->view());
}

struct column_to_string_view_vector {
  cudf::column_device_view const d_strings;
  __device__ cudf::string_view operator()(cudf::size_type idx) const
  {
    cudf::string_view d_str{nullptr, 0};
    if (d_strings.is_valid(idx)) d_str = d_strings.element<cudf::string_view>(idx);
    return d_str;
  }
};

TEST_F(StringsColumnTest, Scatter)
{
  std::vector<const char*> h_strings1{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings1(
    h_strings1.begin(),
    h_strings1.end(),
    thrust::make_transform_iterator(h_strings1.begin(), [](auto str) { return str != nullptr; }));
  auto target = cudf::strings_column_view(strings1);
  std::vector<const char*> h_strings2{"1", "22"};
  cudf::test::strings_column_wrapper strings2(
    h_strings2.begin(),
    h_strings2.end(),
    thrust::make_transform_iterator(h_strings2.begin(), [](auto str) { return str != nullptr; }));
  auto source = cudf::strings_column_view(strings2);

  rmm::device_vector<int32_t> scatter_map;
  scatter_map.push_back(4);
  scatter_map.push_back(1);

  auto source_column = cudf::column_device_view::create(source.parent());
  auto begin = thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(0),
                                               column_to_string_view_vector{*source_column});

  auto results =
    cudf::strings::detail::scatter(begin, begin + source.size(), scatter_map.begin(), target);

  std::vector<const char*> h_expected{"eee", "22", nullptr, "", "1", "bbb", "ééé"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsColumnTest, ScatterScalar)
{
  std::vector<const char*> h_strings1{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings1(
    h_strings1.begin(),
    h_strings1.end(),
    thrust::make_transform_iterator(h_strings1.begin(), [](auto str) { return str != nullptr; }));
  auto target = cudf::strings_column_view(strings1);

  rmm::device_vector<int32_t> scatter_map;
  scatter_map.push_back(0);
  scatter_map.push_back(5);

  cudf::string_scalar scalar("__");
  auto begin = thrust::make_constant_iterator(cudf::string_view(scalar.data(), scalar.size()));

  auto results =
    cudf::strings::detail::scatter(begin, begin + scatter_map.size(), scatter_map.begin(), target);

  std::vector<const char*> h_expected{"__", "bb", nullptr, "", "aa", "__", "ééé"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsColumnTest, ScatterZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto source = cudf::strings_column_view(zero_size_strings_column);
  cudf::column_view values(cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto target = cudf::strings_column_view(values);

  rmm::device_vector<int32_t> scatter_map;
  cudf::string_scalar scalar("");
  auto begin = thrust::make_constant_iterator(cudf::string_view(scalar.data(), scalar.size()));

  auto results = cudf::strings::detail::scatter(begin, begin, scatter_map.begin(), target);
  cudf::test::expect_strings_empty(results->view());
}

CUDF_TEST_PROGRAM_MAIN()
