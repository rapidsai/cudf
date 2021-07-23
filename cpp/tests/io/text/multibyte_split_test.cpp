/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/io/text/data_chunk_source_factories.hpp>
#include <cudf/io/text/multibyte_split.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <sstream>

using namespace cudf;
using namespace test;

constexpr bool print_all{true};

struct MultibyteSplitTest : public BaseFixture {
};

TEST_F(MultibyteSplitTest, SimpleStreaming)
{
  // 😀 | F0 9F 98 80 | 11110000 10011111 01100010 01010000
  // 😎 | F0 9F 98 8E | 11110000 10011111 01100010 11101000
  auto delimiters = std::vector<std::string>({"😀", "😎", ",", "::"});
  auto host_input = std::string(
    "aaa😀"
    "bbb😀"
    "ccc😀"
    "ddd😀"
    "eee😀"
    "fff::"
    "ggg😀"
    "hhh😀"
    "___,"
    "here,"
    "is,"
    "another,"
    "simple😀"
    "text😎"
    "seperated😎"
    "by😎"
    "emojis,"
    "which,"
    "are😎"
    "multiple,"
    "bytes::"
    "and😎"
    "used😎"
    "as😎"
    "delimeters.😎"
    "::"
    ","
    "😀");

  auto expected = strings_column_wrapper{
    "aaa😀",         "bbb😀",   "ccc😀", "ddd😀",      "eee😀",    "fff::", "ggg😀",       "hhh😀",
    "___,",         "here,",  "is,",  "another,",  "simple😀", "text😎", "seperated😎", "by😎",
    "emojis,",      "which,", "are😎", "multiple,", "bytes::", "and😎",  "used😎",      "as😎",
    "delimeters.😎", "::",     ",",    "😀",         ""};

  auto source = cudf::io::text::make_source(host_input);
  auto out    = cudf::io::text::multibyte_split(*source, delimiters);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *out, print_all);
}
