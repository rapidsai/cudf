/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf_test/cudf_gtest.hpp>

#include <cudf/io/thread.hpp>
#include <cudf/utilities/error.hpp>

#include <kvikio/defaults.hpp>

TEST(ThreadTest, ChangeSettings)
{
  std::vector<unsigned int> expected_settings{1u, 4u, 16u, 4u, 1u};
  for (auto const& expected_setting : expected_settings) {
    cudf::io::detail::set_num_io_threads(expected_setting);
    EXPECT_EQ(cudf::io::detail::num_io_threads(), expected_setting);
  }
}

TEST(ThreadTest, InvalidSettings)
{
  EXPECT_THROW({ cudf::io::detail::set_num_io_threads(0u); }, cudf::logic_error);
}
