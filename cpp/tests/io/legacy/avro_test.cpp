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

#include <cudf/cudf.h>
#include <cuda_runtime.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <tests/io/legacy/io_test_utils.hpp>
#include <tests/utilities/legacy/cudf_test_fixtures.h>

TempDirTestEnvironment *const temp_env = static_cast<TempDirTestEnvironment *>(
    ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));
struct avro_test : GdfTest {};

TEST_F(avro_test, DISABLED_Basic) {
  std::string data = "\n";

  cudf::avro_read_arg args(cudf::source_info(data.c_str(), data.size()));

  const auto df = cudf::read_avro(args);
}
