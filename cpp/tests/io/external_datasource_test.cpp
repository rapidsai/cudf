/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cudf/cudf.h>
#include <cudf/legacy/unary.hpp>
#include <nvstrings/NVStrings.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tests/io/legacy/io_test_utils.hpp>
#include <tests/utilities/legacy/cudf_test_fixtures.h>

#include <arrow/io/api.h>

#include <io/utilities/datasource_factory.hpp>

TempDirTestEnvironment* const temp_env = static_cast<TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));

/**
 * @brief Base test fixture for CSV reader/writer tests
 **/
struct CsvTest : public GdfTest {};

TEST_F(CsvTest, Basic)
{
    cudf::io::datasource_factory dfs;
}
