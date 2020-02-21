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
    //kafka_conf_ = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    kafka_conf_ = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
  }

  kafka_datasource::kafka_datasource(std::map<std::string, std::string> configs) {
    DATASOURCE_ID = "librdkafka-1.2.2";

    // Construct the RdKafka::Conf object
    //kafka_conf_ = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    kafka_conf_ = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
    configure_datasource(configs);
  }

  std::string kafka_datasource::libcudf_datasource_identifier() {
    return DATASOURCE_ID;
  }

  bool kafka_datasource::configure_datasource(std::map<std::string, std::string> configs) {
    std::map<std::string, std::string>::iterator it = configs.begin();
    while (it != configs.end())
    {
      std::string name = it->first;
      std::string value = it->second;
      printf("Configuring '%s' - '%s' -> '%s'\n", DATASOURCE_ID.c_str(), name.c_str(), value.c_str());

      conf_res_ = kafka_conf_->set(name, value, errstr_);
  
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
    conf_res_ = kafka_conf_->get("group.id", conf_val);

    // Create the Rebalance callback so Partition Offsets can be assigned.
    //KafkaRebalanceCB rebalance_cb(kafka_start_offset_);
    //kafka_conf_->set("rebalance_cb", &rebalance_cb, errstr_);

    //consumer_ = std::unique_ptr<RdKafka::KafkaConsumer>(RdKafka::KafkaConsumer::create(kafka_conf_.get(), errstr_));
    consumer_ = RdKafka::KafkaConsumer::create(kafka_conf_, errstr_);

    err_ = consumer_->subscribe(topics_);
    if (err_) {
      printf("Error Subscribing to Topics: '%s'\n", err2str(err_).c_str());
    }

    //consume_messages(kafka_conf_);

    return true;
  }

  std::string kafka_datasource::consume_range(std::map<std::string, std::string> configs,
                                              int64_t start_offset,
                                              int64_t end_offset,
                                              int batch_timeout) {
    std::string json_str = "";
    int64_t messages_read = 0;
    int64_t batch_size = end_offset - start_offset;

    printf("Start Offset: '%lu' End Offset: '%lu' Batch Size: '%lu'\n", start_offset, end_offset, batch_size);

    int64_t end = now() + batch_timeout;
    int remaining_timeout = batch_timeout;

    RdKafka::Message *msg;
    while (messages_read < batch_size) {
      printf("Entering while loop ...\n");
      if (consumer_) {
        printf("Consumer_ seems to point to a valid unique_ptr instance...\n");
      } else {
        printf("Consumer_ is NULL\n");
      }
      msg = consumer_->consume(default_timeout_);
      printf("After consume in the while loop .....\n");

      switch (msg->err()) {
        case RdKafka::ErrorCode::ERR__TIMED_OUT:
          printf("Inside timeout .... \n");
          delete msg;
          break;

        case RdKafka::ErrorCode::ERR_NO_ERROR:
          printf("Inside no error ... \n");
          json_str.append(static_cast<char *>(msg->payload()));
          printf("After appending messages.... \n");
          messages_read++;
          printf("Message Read\n");
          break;

        default:
          printf("Inside default .... \n");
          printf("'%s' Consumer error\n", msg->errstr().c_str());
          delete msg;
      }

      remaining_timeout = end - now();
      if (remaining_timeout < 0) {
        break;
      }
    }

    return json_str;
  }

  std::map<std::string, int64_t> kafka_datasource::get_watermark_offset(std::string topic, int32_t partition) {
    int64_t *low;
    int64_t *high;
    std::vector<RdKafka::TopicPartition *> topic_parts;
    std::map<std::string, int64_t> results;

    err_ = consumer_->assignment(topic_parts);
    if (err_ != RdKafka::ErrorCode::ERR_NO_ERROR) {
      printf("Error: '%s'\n", err2str(err_).c_str());
    }
    printf("TopicPartition Size: '%lu'\n", topic_parts.size());
    printf("Topic: '%s' Partition: '%d'\n", topic_parts[0]->topic().c_str(), topic_parts[0]->partition());
    err_ = consumer_->get_watermark_offsets(topic_parts[0]->topic().c_str(), topic_parts[0]->partition(), low, high);
    printf("Before\n");

    if (err_ != RdKafka::ErrorCode::ERR_NO_ERROR) {
      printf("Error: '%s'\n", err2str(err_).c_str());
    } else {
      printf("Low Offset: '%ld' High Offset: '%ld'\n", *low, *high);
      results.insert(std::pair<std::string, int64_t>("low", *low));
      results.insert(std::pair<std::string, int64_t>("high", *high));
    }

    return results;
  }

  bool kafka_datasource::commit(std::string topic, int partition, int64_t offset) {
    return true;
  }

}  // namespace external
}  // namespace io
}  // namespace cudf
