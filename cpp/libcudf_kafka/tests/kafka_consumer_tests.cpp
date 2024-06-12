/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>

#include <cudf_kafka/kafka_consumer.hpp>

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>

namespace kafka = cudf::io::external::kafka;

struct KafkaDatasourceTest : public ::testing::Test {};

TEST_F(KafkaDatasourceTest, MissingGroupID)
{
  // group.id is a required configuration.
  std::map<std::string, std::string> kafka_configs;
  kafka_configs["bootstrap.servers"] = "localhost:9092";

  kafka::python_callable_type python_callable;
  kafka::kafka_oauth_callback_wrapper_type callback_wrapper;

  EXPECT_THROW(
    kafka::kafka_consumer kc(
      kafka_configs, python_callable, callback_wrapper, "csv-topic", 0, 0, 3, 5000, "\n"),
    cudf::logic_error);
}

TEST_F(KafkaDatasourceTest, InvalidConfigValues)
{
  // Give a made up configuration value
  std::map<std::string, std::string> kafka_configs;
  kafka_configs["completely_made_up_config"] = "wrong";

  kafka::python_callable_type python_callable;
  kafka::kafka_oauth_callback_wrapper_type callback_wrapper;

  EXPECT_THROW(
    kafka::kafka_consumer kc(
      kafka_configs, python_callable, callback_wrapper, "csv-topic", 0, 0, 3, 5000, "\n"),
    cudf::logic_error);

  // Give a good config property with a bad value
  kafka_configs.clear();
  kafka_configs["message.max.bytes"] = "this should be a number not text";

  EXPECT_THROW(
    kafka::kafka_consumer kc(
      kafka_configs, python_callable, callback_wrapper, "csv-topic", 0, 0, 3, 5000, "\n"),
    cudf::logic_error);
}
