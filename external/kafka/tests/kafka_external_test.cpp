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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sys/stat.h>
#include <string>
#include <vector>
#include <map>

#include <kafka_datasource.hpp>

// TEST(ExternalDatasource, Basic)
// {
//     std::map<std::string, std::string> datasource_confs;

//     //Topic
//     datasource_confs.insert({"ex_ds.kafka.topic", "libcudf-test"});

//     //General Conf
//     datasource_confs.insert({"bootstrap.servers", "localhost:9092"});
//     datasource_confs.insert({"group.id", "jeremy_test_last_59"});
//     datasource_confs.insert({"auto.offset.reset", "beginning"});

//     cudf::io::external::kafka_datasource ex_datasource = cudf::io::external::kafka_datasource(datasource_confs);
//     std::string json_str = ex_datasource.consume_range(datasource_confs, 0, 15, 5000);
// }

TEST(ExternalDatasource, WaterMark)
{
    std::string topic = "libcudf-test";
    int partition = 0;

    std::map<std::string, std::string> datasource_confs;
    std::vector<std::string> topics;
    topics.push_back("libcudf-test");

    std::vector<int> partitions;
    partitions.push_back(0);

    //Topic
    datasource_confs.insert({"ex_ds.kafka.topic", topic});

    //General Conf
    datasource_confs.insert({"bootstrap.servers", "localhost:9092"});
    datasource_confs.insert({"group.id", "jeremy_test_last"});
    datasource_confs.insert({"auto.offset.reset", "beginning"});
    datasource_confs.insert({"enable.partition.eof", "true"});

    // cudf::io::external::kafka_datasource ex_datasource = std::move(cudf::io::external::kafka_datasource(datasource_confs, topics, partitions));
    // ex_datasource.print_consumer_metadata();
    // ex_datasource.dump_configs();
    // ex_datasource.consume_range(0, 1, 6000, "\n");
    // std::map<std::string, int64_t> offsets = ex_datasource.get_watermark_offset(topic, partition);
}
