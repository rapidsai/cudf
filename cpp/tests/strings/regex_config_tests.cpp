/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/strings/regex/config.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/device_uvector.hpp>

#include <string>

struct StringsRegexConfigTest : public cudf::test::BaseFixture {
};

TEST_F(StringsRegexConfigTest, Basic)
{
  cudf::test::strings_column_wrapper input({"abc", "", "defghijk", "lmnop", "", "qrstuvwxyz"},
                                           {1, 1, 1, 1, 0, 1});
  auto sv = cudf::strings_column_view(input);

  auto results = cudf::strings::compute_regex_state_memory(sv, "hello");
  EXPECT_EQ(results.first, 736);
  EXPECT_EQ(results.second, sv.size());

  results = cudf::strings::compute_regex_state_memory(sv, "");
  EXPECT_EQ(results.first, 160);
  EXPECT_EQ(results.second, sv.size());
}

TEST_F(StringsRegexConfigTest, Large)
{
  auto const d_chars   = rmm::device_uvector<char>{0, rmm::cuda_stream_default};
  auto const d_offsets = cudf::detail::make_zeroed_device_uvector_sync<cudf::size_type>(16000001);
  auto const d_nulls   = rmm::device_uvector<cudf::bitmask_type>{0, rmm::cuda_stream_default};
  auto const input     = cudf::make_strings_column(d_chars, d_offsets, d_nulls, 0);
  auto const sv        = cudf::strings_column_view(input->view());

  std::string pattern =
    "a very large regular expression pattern whose contents do not really matter as much as the "
    "length does";

  auto results = cudf::strings::compute_regex_state_memory(sv, pattern);
  EXPECT_EQ(results.first, 8344000000);
  EXPECT_EQ(results.second, sv.size() / 4);
}

TEST_F(StringsRegexConfigTest, Empty)
{
  auto empty_col = cudf::make_empty_column(cudf::type_id::STRING);
  auto sv        = cudf::strings_column_view(empty_col->view());

  auto results = cudf::strings::compute_regex_state_memory(sv, "a");
  EXPECT_EQ(results.first, 0);
  EXPECT_EQ(results.second, sv.size());
}
