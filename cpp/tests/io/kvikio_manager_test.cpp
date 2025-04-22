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

#include <cudf/io/kvikio_manager.hpp>
#include <cudf/utilities/error.hpp>

#include <kvikio/defaults.hpp>

TEST(KvikIOManagerTest, ChangeSettings)
{
  std::vector<unsigned int> settings{1u, 4u, 16u, 4u, 1u};
  for (auto const& current_setting : settings) {
    cudf::io::kvikio_manager::set_num_io_threads(current_setting);
    auto actual_setting   = kvikio::defaults::thread_pool_nthreads();
    auto expected_setting = cudf::io::kvikio_manager::num_io_threads();
    EXPECT_EQ(actual_setting, expected_setting);
  }
}

TEST(KvikIOManagerTest, InvalidSettings)
{
  EXPECT_THROW({ cudf::io::kvikio_manager::set_num_io_threads(0u); }, cudf::logic_error);
}
