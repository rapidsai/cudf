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

#include <kvikio/defaults.hpp>

TEST(KvikIOManagerTest, ChangeSettings)
{
  auto initial_setting = cudf::io::kvikio_manager::num_io_threads();

  // Use a new setting
  {
    unsigned int new_setting{8u};
    cudf::io::kvikio_manager::set_num_io_threads(new_setting);
    auto actual_setting   = kvikio::defaults::thread_pool_nthreads();
    auto expected_setting = cudf::io::kvikio_manager::num_io_threads();
    EXPECT_EQ(actual_setting, expected_setting);
  }

  // Revert to the initial setting
  {
    cudf::io::kvikio_manager::set_num_io_threads(initial_setting);
    auto actual_setting = kvikio::defaults::thread_pool_nthreads();
    EXPECT_EQ(actual_setting, initial_setting);
  }
}
