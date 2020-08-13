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

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/split/partition.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>

#include <tests/strings/utilities.h>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>

#include <vector>

struct StringsSplitTest : public cudf::test::BaseFixture {
};

TEST_F(StringsSplitTest, Split)
{
  std::vector<const char*> h_strings{
    "Héllo thesé", nullptr, "are some", "tést String", "", "no-delimiter"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view strings_view(strings);

  std::vector<const char*> h_expected1{"Héllo", nullptr, "are", "tést", "", "no-delimiter"};
  cudf::test::strings_column_wrapper expected1(
    h_expected1.begin(),
    h_expected1.end(),
    thrust::make_transform_iterator(h_expected1.begin(), [](auto str) { return str != nullptr; }));
  std::vector<const char*> h_expected2{"thesé", nullptr, "some", "String", nullptr, nullptr};
  cudf::test::strings_column_wrapper expected2(
    h_expected2.begin(),
    h_expected2.end(),
    thrust::make_transform_iterator(h_expected2.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto results = cudf::strings::split(strings_view, cudf::string_scalar(" "));
  EXPECT_TRUE(results->num_columns() == 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, SplitWithMax)
{
  cudf::test::strings_column_wrapper strings(
    {"Héllo::thesé::world", "are::some", "tést::String:", ":last::one", ":::", "x::::y"});
  cudf::strings_column_view strings_view(strings);

  cudf::test::strings_column_wrapper expected1({"Héllo", "are", "tést", ":last", "", "x"});
  cudf::test::strings_column_wrapper expected2(
    {"thesé::world", "some", "String:", "one", ":", "::y"});
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto results = cudf::strings::split(strings_view, cudf::string_scalar("::"), 1);
  EXPECT_TRUE(results->num_columns() == 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, SplitWhitespace)
{
  std::vector<const char*> h_strings{
    "Héllo thesé", nullptr, "are\tsome", "tést\nString", "  ", " a  b ", ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view strings_view(strings);

  std::vector<const char*> h_expected1{"Héllo", nullptr, "are", "tést", nullptr, "a", nullptr};
  cudf::test::strings_column_wrapper expected1(
    h_expected1.begin(),
    h_expected1.end(),
    thrust::make_transform_iterator(h_expected1.begin(), [](auto str) { return str != nullptr; }));
  std::vector<const char*> h_expected2{"thesé", nullptr, "some", "String", nullptr, "b", nullptr};
  cudf::test::strings_column_wrapper expected2(
    h_expected2.begin(),
    h_expected2.end(),
    thrust::make_transform_iterator(h_expected2.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto results = cudf::strings::split(strings_view);
  EXPECT_TRUE(results->num_columns() == 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, SplitWhitespaceWithMax)
{
  cudf::test::strings_column_wrapper strings(
    {"a bc d", "a  bc  d", " ab cd e", "ab cd e ", " ab cd e "});
  cudf::strings_column_view strings_view(strings);

  cudf::test::strings_column_wrapper expected1({"a", "a", "ab", "ab", "ab"});
  cudf::test::strings_column_wrapper expected2({"bc d", "bc  d", "cd e", "cd e ", "cd e "});
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto results = cudf::strings::split(strings_view, cudf::string_scalar(""), 1);
  EXPECT_TRUE(results->num_columns() == 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, RSplit)
{
  std::vector<const char*> h_strings{
    "héllo", nullptr, "a_bc_déf", "a__bc", "_ab_cd", "ab_cd_", "", " a b ", " a  bbb   c"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view strings_view(strings);

  std::vector<const char*> h_expected1{
    "héllo", nullptr, "a", "a", "", "ab", "", " a b ", " a  bbb   c"};
  cudf::test::strings_column_wrapper expected1(
    h_expected1.begin(),
    h_expected1.end(),
    thrust::make_transform_iterator(h_expected1.begin(), [](auto str) { return str != nullptr; }));
  std::vector<const char*> h_expected2{
    nullptr, nullptr, "bc", "", "ab", "cd", nullptr, nullptr, nullptr};
  cudf::test::strings_column_wrapper expected2(
    h_expected2.begin(),
    h_expected2.end(),
    thrust::make_transform_iterator(h_expected2.begin(), [](auto str) { return str != nullptr; }));
  std::vector<const char*> h_expected3{
    nullptr, nullptr, "déf", "bc", "cd", "", nullptr, nullptr, nullptr};
  cudf::test::strings_column_wrapper expected3(
    h_expected3.begin(),
    h_expected3.end(),
    thrust::make_transform_iterator(h_expected3.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  expected_columns.push_back(expected3.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto results = cudf::strings::rsplit(strings_view, cudf::string_scalar("_"));
  EXPECT_TRUE(results->num_columns() == 3);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, RSplitWithMax)
{
  cudf::test::strings_column_wrapper strings(
    {"Héllo::thesé::world", "are::some", "tést::String:", ":last::one", ":::", "x::::y"});
  cudf::strings_column_view strings_view(strings);

  cudf::test::strings_column_wrapper expected1(
    {"Héllo::thesé", "are", "tést", ":last", ":", "x::"});
  cudf::test::strings_column_wrapper expected2({"world", "some", "String:", "one", "", "y"});
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto results = cudf::strings::rsplit(strings_view, cudf::string_scalar("::"), 1);
  EXPECT_TRUE(results->num_columns() == 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, RSplitWhitespace)
{
  std::vector<const char*> h_strings{"héllo", nullptr, "a_bc_déf", "", " a\tb ", " a\r bbb   c"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::strings_column_view strings_view(strings);
  std::vector<const char*> h_expected1{"héllo", nullptr, "a_bc_déf", nullptr, "a", "a"};
  cudf::test::strings_column_wrapper expected1(
    h_expected1.begin(),
    h_expected1.end(),
    thrust::make_transform_iterator(h_expected1.begin(), [](auto str) { return str != nullptr; }));
  std::vector<const char*> h_expected2{nullptr, nullptr, nullptr, nullptr, "b", "bbb"};
  cudf::test::strings_column_wrapper expected2(
    h_expected2.begin(),
    h_expected2.end(),
    thrust::make_transform_iterator(h_expected2.begin(), [](auto str) { return str != nullptr; }));
  std::vector<const char*> h_expected3{nullptr, nullptr, nullptr, nullptr, nullptr, "c"};
  cudf::test::strings_column_wrapper expected3(
    h_expected3.begin(),
    h_expected3.end(),
    thrust::make_transform_iterator(h_expected3.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  expected_columns.push_back(expected3.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto results = cudf::strings::rsplit(strings_view);
  EXPECT_TRUE(results->num_columns() == 3);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, RSplitWhitespaceWithMax)
{
  cudf::test::strings_column_wrapper strings(
    {"a bc d", "a  bc  d", " ab cd e", "ab cd e ", " ab cd e "});
  cudf::strings_column_view strings_view(strings);

  cudf::test::strings_column_wrapper expected1({"a bc", "a  bc", " ab cd", "ab cd", " ab cd"});
  cudf::test::strings_column_wrapper expected2({"d", "d", "e", "e", "e"});
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto results = cudf::strings::rsplit(strings_view, cudf::string_scalar(""), 1);
  EXPECT_TRUE(results->num_columns() == 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, SplitZeroSizeStringsColumns)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto results = cudf::strings::split(zero_size_strings_column);
  EXPECT_TRUE(results->num_columns() == 1);
  cudf::test::expect_strings_empty(results->get_column(0));
  results = cudf::strings::rsplit(zero_size_strings_column);
  EXPECT_TRUE(results->num_columns() == 1);
  cudf::test::expect_strings_empty(results->get_column(0));
}

// This test specifically for https://github.com/rapidsai/custrings/issues/119
TEST_F(StringsSplitTest, AllNullsCase)
{
  std::vector<const char*> h_strings{nullptr, nullptr, nullptr};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto results = cudf::strings::split(cudf::strings_column_view(strings));
  EXPECT_TRUE(results->num_columns() == 1);
  auto column = results->get_column(0).view();
  EXPECT_TRUE(column.size() == 3);
  EXPECT_TRUE(column.has_nulls());
  EXPECT_TRUE(column.null_count() == column.size());
  results = cudf::strings::split(cudf::strings_column_view(strings), cudf::string_scalar("-"));
  EXPECT_TRUE(results->num_columns() == 1);
  column = results->get_column(0);
  EXPECT_TRUE(column.size() == 3);
  EXPECT_TRUE(column.has_nulls());
  EXPECT_TRUE(column.null_count() == column.size());
}

TEST_F(StringsSplitTest, SplitRecord)
{
  std::vector<const char*> h_strings{" Héllo thesé", nullptr, "are some  ", "tést String", ""};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);

  auto result =
    cudf::strings::split_record(cudf::strings_column_view(strings), cudf::string_scalar(" "));
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected(
    {LCW{"", "Héllo", "thesé"}, LCW{}, LCW{"are", "some", "", ""}, LCW{"tést", "String"}, LCW{""}},
    validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, SplitRecordWithMaxSplit)
{
  std::vector<const char*> h_strings{" Héllo thesé", nullptr, "are some  ", "tést String", ""};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);

  auto result =
    cudf::strings::split_record(cudf::strings_column_view(strings), cudf::string_scalar(" "), 1);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected(
    {LCW{"", "Héllo thesé"}, LCW{}, LCW{"are", "some  "}, LCW{"tést", "String"}, LCW{""}},
    validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, SplitRecordWhitespace)
{
  std::vector<const char*> h_strings{
    "   Héllo thesé", nullptr, "are\tsome  ", "tést\nString", "  "};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);

  auto result = cudf::strings::split_record(cudf::strings_column_view(strings));
  using LCW   = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"Héllo", "thesé"}, LCW{}, LCW{"are", "some"}, LCW{"tést", "String"}, LCW{}},
               validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, SplitRecordWhitespaceWithMaxSplit)
{
  std::vector<const char*> h_strings{
    "   Héllo thesé  ", nullptr, "are\tsome  ", "tést\nString", "  "};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);

  auto result =
    cudf::strings::split_record(cudf::strings_column_view(strings), cudf::string_scalar(""), 1);
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"Héllo", "thesé  "}, LCW{}, LCW{"are", "some  "}, LCW{"tést", "String"}, LCW{}},
               validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, RSplitRecord)
{
  std::vector<const char*> h_strings{
    "héllo", nullptr, "a_bc_déf", "a__bc", "_ab_cd", "ab_cd_", "", " a b ", " a  bbb   c"};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"héllo"},
                LCW{},
                LCW{"a", "bc", "déf"},
                LCW{"a", "", "bc"},
                LCW{"", "ab", "cd"},
                LCW{"ab", "cd", ""},
                LCW{""},
                LCW{" a b "},
                LCW{" a  bbb   c"}},
               validity);
  auto result =
    cudf::strings::rsplit_record(cudf::strings_column_view(strings), cudf::string_scalar("_"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, RSplitRecordWithMaxSplit)
{
  std::vector<const char*> h_strings{"héllo",
                                     nullptr,
                                     "a_bc_déf",
                                     "___a__bc",
                                     "_ab_cd_",
                                     "ab_cd_",
                                     "",
                                     " a b ___",
                                     "___ a  bbb   c"};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"héllo"},
                LCW{},
                LCW{"a", "bc", "déf"},
                LCW{"___a", "", "bc"},
                LCW{"_ab", "cd", ""},
                LCW{"ab", "cd", ""},
                LCW{""},
                LCW{" a b _", "", ""},
                LCW{"_", "", " a  bbb   c"}},
               validity);

  auto result =
    cudf::strings::rsplit_record(cudf::strings_column_view(strings), cudf::string_scalar("_"), 2);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, RSplitRecordWhitespace)
{
  std::vector<const char*> h_strings{"héllo", nullptr, "a_bc_déf", "", " a\tb ", " a\r bbb   c"};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"héllo"}, LCW{}, LCW{"a_bc_déf"}, LCW{}, LCW{"a", "b"}, LCW{"a", "bbb", "c"}},
               validity);

  auto result = cudf::strings::rsplit_record(cudf::strings_column_view(strings));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, RSplitRecordWhitespaceWithMaxSplit)
{
  std::vector<const char*> h_strings{
    "  héllo Asher ", nullptr, "   a_bc_déf   ", "", " a\tb ", " a\r bbb   c"};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected(
    {LCW{"  héllo", "Asher"}, LCW{}, LCW{"a_bc_déf"}, LCW{}, LCW{" a", "b"}, LCW{" a\r bbb", "c"}},
    validity);

  auto result =
    cudf::strings::rsplit_record(cudf::strings_column_view(strings), cudf::string_scalar(""), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, SplitRecordZeroSizeStringsColumns)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto split_record_result = cudf::strings::split_record(zero_size_strings_column);
  EXPECT_TRUE(split_record_result->size() == 0);
  auto rsplit_record_result = cudf::strings::rsplit_record(zero_size_strings_column);
  EXPECT_TRUE(rsplit_record_result->size() == 0);
}

TEST_F(StringsSplitTest, Partition)
{
  std::vector<const char*> h_strings{
    "héllo", nullptr, "a_bc_déf", "a__bc", "_ab_cd", "ab_cd_", "", " a b "};
  std::vector<const char*> h_expecteds{
    "héllo", nullptr, "a", "a", "", "ab",    "",       " a b ", "",      nullptr, "_", "_",
    "_",     "_",     "",  "",  "", nullptr, "bc_déf", "_bc",   "ab_cd", "cd_",   "",  ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view strings_view(strings);
  auto results = cudf::strings::partition(strings_view, cudf::string_scalar("_"));
  EXPECT_TRUE(results->num_columns() == 3);

  auto exp_itr = h_expecteds.begin();
  cudf::test::strings_column_wrapper expected1(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected2(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected3(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  expected_columns.push_back(expected3.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, PartitionWhitespace)
{
  std::vector<const char*> h_strings{
    "héllo", nullptr, "a bc déf", "a  bc", " ab cd", "ab cd ", "", "a_b"};
  std::vector<const char*> h_expecteds{"héllo", nullptr, "a",      "a",   "",      "ab",  "", "a_b",
                                       "",      nullptr, " ",      " ",   " ",     " ",   "", "",
                                       "",      nullptr, "bc déf", " bc", "ab cd", "cd ", "", ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view strings_view(strings);
  auto results = cudf::strings::partition(strings_view);
  EXPECT_TRUE(results->num_columns() == 3);

  auto exp_itr = h_expecteds.begin();
  cudf::test::strings_column_wrapper expected1(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected2(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected3(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  expected_columns.push_back(expected3.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, RPartition)
{
  std::vector<const char*> h_strings{
    "héllo", nullptr, "a_bc_déf", "a__bc", "_ab_cd", "ab_cd_", "", " a b "};
  std::vector<const char*> h_expecteds{"",      nullptr, "a_bc", "a_", "_ab", "ab_cd", "", "",
                                       "",      nullptr, "_",    "_",  "_",   "_",     "", "",
                                       "héllo", nullptr, "déf",  "bc", "cd",  "",      "", " a b "};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view strings_view(strings);
  auto results = cudf::strings::rpartition(strings_view, cudf::string_scalar("_"));
  EXPECT_TRUE(results->num_columns() == 3);

  auto exp_itr = h_expecteds.begin();
  cudf::test::strings_column_wrapper expected1(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected2(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected3(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  expected_columns.push_back(expected3.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, RPartitionWhitespace)
{
  std::vector<const char*> h_strings{
    "héllo", nullptr, "a bc déf", "a  bc", " ab cd", "ab cd ", "", "a_b"};
  std::vector<const char*> h_expecteds{"",      nullptr, "a bc", "a ", " ab", "ab cd", "", "",
                                       "",      nullptr, " ",    " ",  " ",   " ",     "", "",
                                       "héllo", nullptr, "déf",  "bc", "cd",  "",      "", "a_b"};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view strings_view(strings);
  auto results = cudf::strings::rpartition(strings_view);
  EXPECT_TRUE(results->num_columns() == 3);

  auto exp_itr = h_expecteds.begin();
  cudf::test::strings_column_wrapper expected1(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected2(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected3(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  expected_columns.push_back(expected3.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, PartitionZeroSizeStringsColumns)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto results = cudf::strings::partition(zero_size_strings_column);
  EXPECT_TRUE(results->num_columns() == 0);
  results = cudf::strings::rpartition(zero_size_strings_column);
  EXPECT_TRUE(results->num_columns() == 0);
}

TEST_F(StringsSplitTest, InvalidParameter)
{
  std::vector<const char*> h_strings{"string left intentionally blank"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto strings_view = cudf::strings_column_view(strings);
  EXPECT_THROW(cudf::strings::split(strings_view, cudf::string_scalar("", false)),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::rsplit(strings_view, cudf::string_scalar("", false)),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::partition(strings_view, cudf::string_scalar("", false)),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::rpartition(strings_view, cudf::string_scalar("", false)),
               cudf::logic_error);
}
