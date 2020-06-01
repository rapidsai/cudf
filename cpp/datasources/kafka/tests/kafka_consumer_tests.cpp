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

#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <string>
#include "kafka_consumer.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/functions.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#define CUDF_DATASOURCE_TEST_PROGRAM_MAIN() \
  int main(int argc, char** argv)           \
  {                                         \
    ::testing::InitGoogleTest(&argc, argv); \
    return RUN_ALL_TESTS();                 \
  }

struct KafkaDatasourceTest : public ::testing::Test {
};

TEST_F(KafkaDatasourceTest, UserImplementedSource)
{
  namespace kafka = cudf::io::external::kafka;

  std::map<std::string, std::string> kafka_configs;
  kafka_configs.insert({"bootstrap.servers", "localhost:9092"});
  kafka_configs.insert({"group.id", "libcudf_consumer"});
  kafka_configs.insert({"auto.offset.reset", "beginning"});

  kafka::kafka_consumer kc(kafka_configs, "csv-topic", 0, 0, 3, 5000, "\n");

  cudf::io::read_csv_args in_args{cudf::io::source_info{&kc}};
  in_args.dtype   = {"int8", "int16", "int32"};
  in_args.header  = -1;
  auto result     = cudf::io::read_csv(in_args);
  auto const view = result.tbl->view();
}

CUDF_DATASOURCE_TEST_PROGRAM_MAIN()