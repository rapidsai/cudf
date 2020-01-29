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
#include <map>

#include <cudf/cudf.h>
#include <cudf/legacy/unary.hpp>
#include <nvstrings/NVStrings.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tests/io/legacy/io_test_utils.hpp>
#include <tests/utilities/legacy/cudf_test_fixtures.h>

#include <arrow/io/api.h>

#include <io/utilities/datasource_factory.hpp>
#include <cudf/io/readers.hpp>

TempDirTestEnvironment* const temp_env = static_cast<TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));

/**
 * @brief Base test fixture for CSV reader/writer tests
 **/
struct ExternalDatasource : public GdfTest {};

TEST_F(ExternalDatasource, Basic)
{
    std::map<std::string, std::string> datasource_confs;

    cudf::io::external::datasource_factory dfs("/home/jdyer/Development/cudf/external/build");
    cudf::io::external::external_datasource* ex_datasource = dfs.external_datasource_by_id("librdkafka-1.2.2", datasource_confs);
    printf("External Datasource ID is '%s'\n", ex_datasource->libcudf_datasource_identifier().c_str());
}

TEST_F(ExternalDatasource, csv_read)
{
    std::map<std::string, std::string> datasource_confs;
    
    // Create the reader.
    //cudf::experimental::io::detail::csv::reader csv_reader = cudf::experimental::io::detail::csv::reader("librdkafka-1.2.2", datasource_confs, );
}
