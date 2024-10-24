/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cudf_test/testing_main.hpp>

#include <src/io/utilities/file_io_utilities.hpp>

// Base test fixture for tests
struct CuFileIOTest : public cudf::test::BaseFixture {};

TEST_F(CuFileIOTest, SliceSize)
{
  std::vector<std::pair<size_t, size_t>> test_cases{
    {1 << 20, 1 << 18}, {1 << 18, 1 << 20}, {1 << 20, 3333}, {0, 1 << 18}, {0, 0}, {1 << 20, 0}};
  for (auto const& test_case : test_cases) {
    auto const slices = cudf::io::detail::make_file_io_slices(test_case.first, test_case.second);
    if (slices.empty()) {
      ASSERT_EQ(test_case.first, 0);
    } else {
      ASSERT_EQ(slices.front().offset, 0);
      ASSERT_EQ(slices.back().offset + slices.back().size, test_case.first);
      for (auto i = 1u; i < slices.size(); ++i) {
        ASSERT_EQ(slices[i].offset, slices[i - 1].offset + slices[i - 1].size);
      }
    }
  }
}

CUDF_TEST_PROGRAM_MAIN()
