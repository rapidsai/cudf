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

#pragma once

#include "cudf_gtest.hpp"
#include "cudf_test_utils.cuh"

#include <rmm/mr/default_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>

namespace cudf {
namespace test {

/**---------------------------------------------------------------------------*
 * @brief Base test fixture class from which all libcudf tests should inherit.
 *
 * Example:
 * ```
 * class MyTestFixture : public cudf::test::BaseFixture {};
 * ```
 *---------------------------------------------------------------------------**/
class BaseFixture : public ::testing::Test {
  rmm::mr::device_memory_resource* _mr{rmm::mr::get_default_resource()};

 public:
  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to `device_memory_resource` that should be used for
   * all tests inheritng from this fixture
   *---------------------------------------------------------------------------**/
  rmm::mr::device_memory_resource* mr() { return _mr; }

  static void SetUpTestCase() {
    ASSERT_RMM_SUCCEEDED( rmmInitialize(nullptr) );
  }

  static void TearDownTestCase() {
    ASSERT_RMM_SUCCEEDED( rmmFinalize() );
  }
};

}  // namespace test
}  // namespace cudf
