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

#include "kafka_datasource.hpp"

namespace cudf {
namespace io {
namespace external {

  kafka_datasource::kafka_datasource() {
    DATASOURCE_ID = "librdkafka-1.2.2";

    // Create an empty RdKafka::Conf instance. The configurations will be constructed later
    kafka_conf_ = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
  }

  kafka_datasource::kafka_datasource(std::map<std::string, std::string> configs) {
    DATASOURCE_ID = "librdkafka-1.2.2";

    // Construct the RdKafka::Conf object
    kafka_conf_ = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    configure_datasource(configs);
  }

  std::string kafka_datasource::libcudf_datasource_identifier() {
    return DATASOURCE_ID;
  }

  bool kafka_datasource::configure_datasource(std::map<std::string, std::string> configs) {
    // Populate the RdKafka::Conf using the user supplied configs
    printf("Entering configure_datasource\n");
    std::map<std::string, std::string>::iterator it = configs.begin();
    while (it != configs.end())
    {
      std::string name = it->first;
      std::string value = it->second;
      printf("Configuring '%s' - '%s' -> '%s'\n", DATASOURCE_ID.c_str(), name.c_str(), value.c_str());

      conf_res_ = kafka_conf_.get()->set(name, value, errstr_);
  
      // Increment the Iterator to point to next entry
      it++;
    }
    printf("configurations set\n");

    std::map<std::string, std::string>::iterator conf_it;
    conf_it = configs.find("ex_ds.kafka.topic");
    if (conf_it != configs.end()) {
      printf("Setting topic name to '%s'\n", conf_it->second.c_str());
      topics_.push_back(conf_it->second);
    } else {
      printf("Unable to find topic configuration value\n");
    }

    // Kafka 0.9 > requires at least a group.id in the configuration so lets
    // make sure that is present.
    conf_res_ = kafka_conf_.get()->get("group.id", conf_val);

    printf("After getting group id\n");

    // Create the Rebalance callback so Partition Offsets can be assigned.
    KafkaRebalanceCB rebalance_cb(kafka_start_offset_);
    kafka_conf_->set("rebalance_cb", &rebalance_cb, errstr_);

    printf("Before creating consumer\n");
    consumer_ = std::unique_ptr<RdKafka::KafkaConsumer>(RdKafka::KafkaConsumer::create(kafka_conf_.get(), errstr_));

    printf("Before subscribing to topics\n");
    err_ = consumer_.get()->subscribe(topics_);

    printf("Before consuming messages\n");
    consume_messages(kafka_conf_);

    return true;
  }

  std::map<std::string, int64_t> kafka_datasource::get_watermark_offset(std::string topic, int partition) {
    std::vector<RdKafka::TopicPartition *> topic_parts;

    err_ = consumer_.get()->position(topic_parts);

    std::map<std::string, int64_t> results;
    return results;
  }

  bool kafka_datasource::commit(std::string topic, int partition, int64_t offset) {
    return true;
  }

}  // namespace external
}  // namespace io
}  // namespace cudf
