/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "parquet_common.hpp"

#include <cudf_test/testing_main.hpp>

// Global environment for temporary files
cudf::test::TempDirTestEnvironment* const temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

CUDF_TEST_PROGRAM_MAIN()
