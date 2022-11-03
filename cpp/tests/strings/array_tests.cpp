/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/detail/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>

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

  auto results =
    cudf::sort(cudf::table_view({h_strings}), {cudf::order::ASCENDING}, {cudf::null_order::BEFORE});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), h_expected);
}

TEST_F(StringsColumnTest, SortZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto results = cudf::sort(cudf::table_view({zero_size_strings_column}));
  cudf::test::expect_column_empty(results->view().column(0));
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
  auto results = cudf::strings::detail::copy_slice(cudf::strings_column_view(strings), start, end);

  cudf::test::strings_column_wrapper expected(
    h_strings.begin() + start,
    h_strings.begin() + end,
    thrust::make_transform_iterator(h_strings.begin() + start,
                                    [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
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
  auto results = cudf::strings::detail::copy_slice(cudf::strings_column_view(strings), start, end);

  cudf::test::strings_column_wrapper expected(
    h_strings.begin() + start,
    h_strings.begin() + end,
    thrust::make_transform_iterator(h_strings.begin() + start,
                                    [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_P(SliceParmsTest, SliceAllEmpty)
{
  std::vector<const char*> h_strings{"", "", "", "", "", "", ""};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());

  cudf::size_type start = 3;
  cudf::size_type end   = GetParam();
  auto results = cudf::strings::detail::copy_slice(cudf::strings_column_view(strings), start, end);

  cudf::test::strings_column_wrapper expected(h_strings.begin() + start, h_strings.begin() + end);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

INSTANTIATE_TEST_CASE_P(StringsColumnTest,
                        SliceParmsTest,
                        testing::ValuesIn(std::array<cudf::size_type, 3>{5, 6, 7}));

TEST_F(StringsColumnTest, SliceZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  auto results      = cudf::strings::detail::copy_slice(strings_view, 1, 2);
  cudf::test::expect_column_empty(results->view());
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
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.front()->view(), expected);
}

TEST_F(StringsColumnTest, GatherZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  cudf::column_view map_view(cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0);
  auto results = cudf::gather(cudf::table_view{{zero_size_strings_column}}, map_view)->release();
  cudf::test::expect_column_empty(results.front()->view());
}

TEST_F(StringsColumnTest, GatherTooBig)
{
  std::vector<int8_t> h_chars(3000000);
  cudf::test::fixed_width_column_wrapper<int8_t> chars(h_chars.begin(), h_chars.end());
  cudf::test::fixed_width_column_wrapper<cudf::offset_type> offsets({0, 3000000});
  auto input = cudf::column_view(
    cudf::data_type{cudf::type_id::STRING}, 1, nullptr, nullptr, 0, 0, {offsets, chars});
  auto map = thrust::constant_iterator<int8_t>(0);
  cudf::test::fixed_width_column_wrapper<int8_t> gather_map(map, map + 1000);
  EXPECT_THROW(cudf::gather(cudf::table_view{{input}}, gather_map), cudf::logic_error);
}

TEST_F(StringsColumnTest, Scatter)
{
  cudf::test::strings_column_wrapper target({"eee", "bb", "", "", "aa", "bbb", "ééé"},
                                            {1, 1, 0, 1, 1, 1, 1});
  cudf::test::strings_column_wrapper source({"1", "22"});

  cudf::test::fixed_width_column_wrapper<int32_t> scatter_map({4, 1});

  auto results = cudf::scatter(cudf::table_view({source}), scatter_map, cudf::table_view({target}));

  cudf::test::strings_column_wrapper expected({"eee", "22", "", "", "1", "bbb", "ééé"},
                                              {1, 1, 0, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
}

TEST_F(StringsColumnTest, ScatterScalar)
{
  cudf::test::strings_column_wrapper target({"eee", "bb", "", "", "aa", "bbb", "ééé"},
                                            {1, 1, 0, 1, 1, 1, 1});

  cudf::test::fixed_width_column_wrapper<int32_t> scatter_map({0, 5});

  cudf::string_scalar scalar("__");
  auto source  = std::vector<std::reference_wrapper<const cudf::scalar>>({scalar});
  auto results = cudf::scatter(source, scatter_map, cudf::table_view({target}));

  cudf::test::strings_column_wrapper expected({"__", "bb", "", "", "aa", "__", "ééé"},
                                              {1, 1, 0, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
}

TEST_F(StringsColumnTest, ScatterZeroSizeStringsColumn)
{
  cudf::column_view source(cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  cudf::column_view target(cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  cudf::column_view scatter_map(cudf::data_type{cudf::type_id::INT8}, 0, nullptr, nullptr, 0);

  auto results = cudf::scatter(cudf::table_view({source}), scatter_map, cudf::table_view({target}));
  cudf::test::expect_column_empty(results->view().column(0));

  cudf::string_scalar scalar("");
  auto scalar_source = std::vector<std::reference_wrapper<const cudf::scalar>>({scalar});
  results            = cudf::scatter(scalar_source, scatter_map, cudf::table_view({target}));
  cudf::test::expect_column_empty(results->view().column(0));
}

TEST_F(StringsColumnTest, OffsetsBeginEnd)
{
  cudf::test::strings_column_wrapper input({"eee", "bb", "", "", "aa", "bbb", "ééé"},
                                           {1, 1, 0, 1, 1, 1, 1});

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 5});
  auto scv = cudf::strings_column_view(input);
  EXPECT_EQ(std::distance(scv.offsets_begin(), scv.offsets_end()),
            static_cast<std::ptrdiff_t>(scv.size() + 1));

  scv = cudf::strings_column_view(cudf::slice(input, {1, 5}).front());
  EXPECT_EQ(std::distance(scv.offsets_begin(), scv.offsets_end()),
            static_cast<std::ptrdiff_t>(scv.size() + 1));
  EXPECT_EQ(std::distance(scv.chars_begin(), scv.chars_end()), 16L);
}

CUDF_TEST_PROGRAM_MAIN()
