/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <cstring>
#include <vector>

struct StringsFactoriesTest : public cudf::test::BaseFixture {
};

TEST_F(StringsFactoriesTest, CreateColumnFromPair)
{
  std::vector<const char*> h_test_strings{"the quick brown fox jumps over the lazy dog",
                                          "the fat cat lays next to the other accénted cat",
                                          "a slow moving turtlé cannot catch the bird",
                                          "which can be composéd together to form a more complete",
                                          "thé result does not include the value in the sum in",
                                          "",
                                          nullptr,
                                          "absent stop words"};

  cudf::size_type memsize = 0;
  for (auto itr = h_test_strings.begin(); itr != h_test_strings.end(); ++itr)
    memsize += *itr ? (cudf::size_type)strlen(*itr) : 0;
  cudf::size_type count = (cudf::size_type)h_test_strings.size();
  thrust::host_vector<char> h_buffer(memsize);
  thrust::device_vector<char> d_buffer(memsize);
  thrust::host_vector<thrust::pair<const char*, cudf::size_type>> strings(count);
  thrust::host_vector<cudf::size_type> h_offsets(count + 1);
  cudf::size_type offset = 0;
  cudf::size_type nulls  = 0;
  h_offsets[0]           = 0;
  for (cudf::size_type idx = 0; idx < count; ++idx) {
    const char* str = h_test_strings[idx];
    if (!str) {
      strings[idx] = thrust::pair<const char*, cudf::size_type>{nullptr, 0};
      nulls++;
    } else {
      cudf::size_type length = (cudf::size_type)strlen(str);
      memcpy(h_buffer.data() + offset, str, length);
      strings[idx] =
        thrust::pair<const char*, cudf::size_type>{d_buffer.data().get() + offset, length};
      offset += length;
    }
    h_offsets[idx + 1] = offset;
  }
  rmm::device_vector<thrust::pair<const char*, cudf::size_type>> d_strings(strings);
  CUDA_TRY(cudaMemcpy(d_buffer.data().get(), h_buffer.data(), memsize, cudaMemcpyHostToDevice));
  auto column = cudf::make_strings_column(d_strings);
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_id::STRING});
  EXPECT_EQ(column->null_count(), nulls);
  if (nulls) {
    EXPECT_TRUE(column->nullable());
    EXPECT_TRUE(column->has_nulls());
  }
  EXPECT_EQ(2, column->num_children());

  cudf::strings_column_view strings_view(column->view());
  EXPECT_EQ(strings_view.size(), count);
  EXPECT_EQ(strings_view.offsets().size(), count + 1);
  EXPECT_EQ(strings_view.chars().size(), memsize);

  // check string data
  auto strings_data = cudf::strings::create_offsets(strings_view);
  thrust::host_vector<char> h_chars_data(strings_data.first);
  thrust::host_vector<cudf::size_type> h_offsets_data(strings_data.second);
  EXPECT_EQ(memcmp(h_buffer.data(), h_chars_data.data(), h_buffer.size()), 0);
  EXPECT_EQ(
    memcmp(h_offsets.data(), h_offsets_data.data(), h_offsets.size() * sizeof(cudf::size_type)), 0);
}

TEST_F(StringsFactoriesTest, CreateColumnFromOffsets)
{
  std::vector<const char*> h_test_strings{"the quick brown fox jumps over the lazy dog",
                                          "the fat cat lays next to the other accénted cat",
                                          "a slow moving turtlé cannot catch the bird",
                                          "which can be composéd together to form a more complete",
                                          "thé result does not include the value in the sum in",
                                          "",
                                          nullptr,
                                          "absent stop words"};

  cudf::size_type memsize = 0;
  for (auto itr = h_test_strings.begin(); itr != h_test_strings.end(); ++itr)
    memsize += *itr ? (cudf::size_type)strlen(*itr) : 0;
  cudf::size_type count = (cudf::size_type)h_test_strings.size();
  std::vector<char> h_buffer(memsize);
  std::vector<cudf::size_type> h_offsets(count + 1);
  cudf::size_type offset         = 0;
  h_offsets[0]                   = offset;
  cudf::bitmask_type h_null_mask = 0;
  cudf::size_type null_count     = 0;
  for (cudf::size_type idx = 0; idx < count; ++idx) {
    h_null_mask     = (h_null_mask << 1);
    const char* str = h_test_strings[idx];
    if (str) {
      cudf::size_type length = (cudf::size_type)strlen(str);
      memcpy(h_buffer.data() + offset, str, length);
      offset += length;
      h_null_mask |= 1;
    } else
      null_count++;
    h_offsets[idx + 1] = offset;
  }
  std::vector<cudf::bitmask_type> h_nulls{h_null_mask};
  rmm::device_vector<char> d_buffer(h_buffer);
  rmm::device_vector<cudf::size_type> d_offsets(h_offsets);
  rmm::device_vector<cudf::bitmask_type> d_nulls(h_nulls);
  auto column = cudf::make_strings_column(d_buffer, d_offsets, d_nulls, null_count);
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_id::STRING});
  EXPECT_EQ(column->null_count(), null_count);
  EXPECT_EQ(2, column->num_children());

  cudf::strings_column_view strings_view(column->view());
  EXPECT_EQ(strings_view.size(), count);
  EXPECT_EQ(strings_view.offsets().size(), count + 1);
  EXPECT_EQ(strings_view.chars().size(), memsize);

  // check string data
  auto strings_data = cudf::strings::create_offsets(strings_view);
  thrust::host_vector<char> h_chars_data(strings_data.first);
  thrust::host_vector<cudf::size_type> h_offsets_data(strings_data.second);
  EXPECT_EQ(memcmp(h_buffer.data(), h_chars_data.data(), h_buffer.size()), 0);
  EXPECT_EQ(
    memcmp(h_offsets.data(), h_offsets_data.data(), h_offsets.size() * sizeof(cudf::size_type)), 0);

  // check host version of the factory too
  auto column2 = cudf::make_strings_column(h_buffer, h_offsets, h_nulls, null_count);
  cudf::test::expect_columns_equal(column->view(), column2->view());
}

TEST_F(StringsFactoriesTest, CreateScalar)
{
  std::string value = "test string";
  auto s            = cudf::make_string_scalar(value);
  auto string_s     = static_cast<cudf::string_scalar*>(s.get());

  EXPECT_EQ(string_s->to_string(), value);
  EXPECT_TRUE(string_s->is_valid());
  EXPECT_TRUE(s->is_valid());
}

TEST_F(StringsFactoriesTest, EmptyStringsColumn)
{
  rmm::device_vector<char> d_chars;
  rmm::device_vector<cudf::size_type> d_offsets(1, 0);
  rmm::device_vector<cudf::bitmask_type> d_nulls;

  auto results = cudf::make_strings_column(d_chars, d_offsets, d_nulls, 0);
  cudf::test::expect_strings_empty(results->view());

  rmm::device_vector<thrust::pair<const char*, cudf::size_type>> d_strings;
  results = cudf::make_strings_column(d_strings);
  cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsFactoriesTest, CreateOffsets)
{
  std::vector<std::string> strings      = {"this", "is", "a", "column", "of", "strings"};
  cudf::test::strings_column_wrapper sw = {strings.begin(), strings.end()};
  cudf::column_view col(sw);
  std::vector<cudf::size_type> indices{0, 2, 3, 6};
  auto result = cudf::slice(col, indices);

  std::vector<std::vector<std::string>> expecteds{
    std::vector<std::string>{"this", "is"},              // [0,2)
    std::vector<std::string>{"column", "of", "strings"}  // [3,6)
  };
  for (size_t idx = 0; idx < result.size(); idx++) {
    auto strings_data = cudf::strings::create_offsets(cudf::strings_column_view(result[idx]));
    thrust::host_vector<char> h_chars(strings_data.first);
    thrust::host_vector<cudf::size_type> h_offsets(strings_data.second);
    auto expected_strings = expecteds[idx];
    for (size_t jdx = 0; jdx < h_offsets.size() - 1; ++jdx) {
      auto offset = h_offsets[jdx];
      auto length = h_offsets[jdx + 1] - offset;
      std::string str(h_chars.data() + offset, length);
      EXPECT_EQ(str, expected_strings[jdx]);
    }
  }
}
