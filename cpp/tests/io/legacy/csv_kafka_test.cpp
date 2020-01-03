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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cstdlib>

#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <io/utilities/legacy/parsing_utils.cuh>
#include <tests/io/legacy/io_test_utils.hpp>

#include <arrow/io/api.h>

#include <librdkafka/rdkafkacpp.h>
#include <rmm/device_buffer.hpp>

TempDirTestEnvironment *const temp_env = static_cast<TempDirTestEnvironment *>(
    ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));

/**
 * @brief Base test fixture for CSV Kafka reader tests
 **/
class KafkaTest : public GdfTest {
 protected:
  void SetUp() override {
    // Create a baseline default configuration so each test will not need to do so
    kafka_conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);

    conf_result = kafka_conf->set("metadata.broker.list", "localhost:9092", _errstr);
    ASSERT_EQ(conf_result, RdKafka::Conf::ConfResult::CONF_OK);
    conf_result = kafka_conf->set("client.id", "libcudf_unit_test", _errstr);
    ASSERT_EQ(conf_result, RdKafka::Conf::ConfResult::CONF_OK);
    conf_result = kafka_conf->set("group.id", "libcudf", _errstr);
    ASSERT_EQ(conf_result, RdKafka::Conf::ConfResult::CONF_OK);
    conf_result = kafka_conf->set("enable.partition.eof", "true", _errstr);
    ASSERT_EQ(conf_result, RdKafka::Conf::ConfResult::CONF_OK);
    conf_result = kafka_conf->set("auto.offset.reset", "earliest", _errstr);
    ASSERT_EQ(conf_result, RdKafka::Conf::ConfResult::CONF_OK);
    conf_result = kafka_conf->set("enable.auto.commit", "false", _errstr);
    ASSERT_EQ(conf_result, RdKafka::Conf::ConfResult::CONF_OK);

    // Populate the topics list with the test default "libcudf-test"
    topics.push_back(default_test_topic);

    // Push some sample messages to Kafka if that feature is enabled.
    publish_kafka_messages();
  }

  void TearDown() override {
    topics.clear();
    populate_kafka = false;
  }

  void publish_kafka_messages() {
    if (populate_kafka) {
      std::string err_str;
      RdKafka::Producer *producer =
          RdKafka::Producer::create(kafka_conf, err_str);

      for (int i = 0; i < messages_to_produce; i++) {
        producer->produce(default_test_topic, RdKafka::Topic::PARTITION_UA,
                          RdKafka::Producer::RK_MSG_COPY,
                          const_cast<char *>(default_message.c_str()),
                          default_message.size(), NULL, 0, 0, NULL, NULL);
      }

      producer->flush(10 * 1000);
      delete producer;
    }
  }

  std::string default_test_topic = "libcudf-csv-test";
  std::string _errstr;
  RdKafka::Conf *kafka_conf;
  RdKafka::Conf::ConfResult conf_result;
  std::vector<std::string> topics;

  bool populate_kafka = false;
  std::string default_message = "fname,lname,50";
  int messages_to_produce = 10;
};

// Test behavior when read_csv is invoked with Kafka configurations that include
// a Kafka broker that either does not exist or offline.
TEST_F(KafkaTest, NonExistantBroker) {
  cudf::csv_read_arg args(cudf::source_info{kafka_conf, topics, 0, 10});

  // Manipulate the default configurations to use a non existent hostname
  conf_result = kafka_conf->set("metadata.broker.list",
                                "deadend-doesnotexist:9092", _errstr);

  // Create the cudf arguments that would be passed to libcudf from cython
  args.source.kafka_conf = kafka_conf;
  args.source.kafka_topics = topics;

  EXPECT_THROW(cudf::read_csv(args), cudf::logic_error);
}

// Test reading from an empty Kafka Broker
TEST_F(KafkaTest, EmptyTopic) {
  cudf::csv_read_arg args(cudf::source_info{kafka_conf, topics, 0, 10});

  topics.clear();
  topics.push_back("nonexistent-kafka-topic");

  // Create the cudf arguments that would be passed to libcudf from cython
  args.source.kafka_conf = kafka_conf;
  args.source.kafka_topics = topics;

  EXPECT_THROW(cudf::read_csv(args), cudf::logic_error);
}

TEST_F(KafkaTest, TopicConnection) {
  cudf::csv_read_arg args(cudf::source_info{kafka_conf, topics, 0, 10});

  args.names = {"fname", "lname", "age"};
  args.source.kafka_conf = kafka_conf;
  args.source.kafka_topics = topics;
  args.source.kafka_start_offset = 0;
  args.source.kafka_batch_size = 5;

  const auto df = cudf::read_csv(args);
  EXPECT_EQ(df.num_columns(), 3);
  EXPECT_EQ(df.num_rows(), args.source.kafka_batch_size);
}
