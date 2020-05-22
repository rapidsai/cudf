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

#include <cudf/copying.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

namespace cudf {
namespace test {

struct PackUnpackTest : public BaseFixture {
  void run_test(std::vector<column_view> const& t)
  {
    auto packed = pack(t);
    auto packed2 =
      std::make_unique<packed_columns>(std::make_unique<std::vector<uint8_t>>(*packed.metadata),
                                       std::make_unique<rmm::device_buffer>(*packed.data));

    unpack_result unpacked = unpack(std::move(packed2));

    for (size_t i = 0; i < t.size(); i++) { expect_columns_equal(t[i], unpacked.columns[i]); }
  }
};

// clang-format off
TEST_F(PackUnpackTest, SingleColumnFixedWidth)
{
  fixed_width_column_wrapper<int64_t> col1 ({ 1, 2, 3, 4, 5, 6, 7},
                                            { 1, 1, 1, 0, 1, 0, 1});

  this->run_test({col1});
}

TEST_F(PackUnpackTest, SingleColumnFixedWidthNonNullable)
{
  fixed_width_column_wrapper<int64_t> col1 ({ 1, 2, 3, 4, 5, 6, 7});

  this->run_test({col1});
}

TEST_F(PackUnpackTest, MultiColumnFixedWidth)
{
  fixed_width_column_wrapper<int16_t> col1 ({ 1, 2, 3, 4, 5, 6, 7},
                                            { 1, 1, 1, 0, 1, 0, 1});
  fixed_width_column_wrapper<float>   col2 ({ 7, 8, 6, 5, 4, 3, 2},
                                            { 1, 0, 1, 1, 1, 1, 1});
  fixed_width_column_wrapper<double>  col3 ({ 8, 4, 2, 0, 7, 1, 9, 3},
                                            { 0, 1, 1, 1, 1, 1, 1, 1});

  this->run_test({col1, col2, col3});
}

TEST_F(PackUnpackTest, MultiColumnWithStrings)
{
  fixed_width_column_wrapper<int16_t> col1 ({ 1, 2, 3, 4, 5, 6, 7},
                                            { 1, 1, 1, 0, 1, 0, 1});
  strings_column_wrapper              col2 ({"Lorem", "ipsum", "dolor", "sit", "amet"},
                                            {      1,       0,       1,     1,      1});
  strings_column_wrapper              col3 ({"", "this", "is", "a", "column", "of", "strings"});

  this->run_test({col1, col2, col3});
}
// clang-format on

}  // namespace test
}  // namespace cudf
