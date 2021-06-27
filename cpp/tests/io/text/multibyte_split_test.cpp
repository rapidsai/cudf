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

#include <cudf/io/text/multibyte_split.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <sstream>

using namespace cudf;
using namespace test;

constexpr bool print_all{false};

struct MultibyteSplitTest : public BaseFixture {
};

TEST_F(MultibyteSplitTest, Simple)
{
  std::string separator = "😎";  // F0 9F 98 8E | 11110000 11111001 1100010 11101000
  std::string input =
    "here😎"
    "is😎"
    "some😎"
    "simple😎"
    "text😎"
    "seperated😎"
    "by😎"
    "emojis😎"
    "which😎"
    "are😎"
    "multple😎"
    "bytes😎"
    "and😎"
    "used😎"
    "as😎"
    "delimeters.";

  auto expected = strings_column_wrapper{"here",
                                         "is",
                                         "some",
                                         "simple",
                                         "text",
                                         "seperated",
                                         "by",
                                         "emojis",
                                         "which",
                                         "are",
                                         "multple",
                                         "bytes",
                                         "and",
                                         "used",
                                         "as",
                                         "delimeters."};

  auto input_stream = std::basic_istringstream(input);

  auto out = cudf::io::text::multibyte_split(input_stream, separator);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *out, print_all);
}
