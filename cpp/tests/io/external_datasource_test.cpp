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
 * @brief Base test fixture external tests
 **/
struct ExternalDatasource : public GdfTest {};

TEST_F(ExternalDatasource, Basic)
{
    std::map<std::string, std::string> datasource_confs;

    //Topic
    datasource_confs.insert({"ex_ds.kafka.topic", "libcudf-test"});

    //General Conf
    datasource_confs.insert({"bootstrap.servers", "localhost:9092"});
    datasource_confs.insert({"group.id", "libcudf_consumer"});
    datasource_confs.insert({"auto.offset.reset", "beginning"});

    //cudf::io::external::datasource_factory dfs("/home/jdyer/anaconda3/envs/cudf_dev/lib/external");
    //cudf::io::external::external_datasource* ex_datasource = dfs.external_datasource_by_id("librdkafka-1.2.2", datasource_confs);
    //ex_datasource->configure_datasource(datasource_confs);

    // TODO: All of this seems more like integration testing to me ... unsure best approach to take for the tests?
}
