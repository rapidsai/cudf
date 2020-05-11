/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include "kafka_consumer.hpp"

namespace cudf {
namespace io {
namespace external {

  kafka_consumer::kafka_consumer() {
    DATASOURCE_ID = "librdkafka-";
    DATASOURCE_ID.append(RdKafka::version_str());

    // Create an empty RdKafka::Conf instance. The configurations will be applied later.
    kafka_conf_ = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
  }

  kafka_consumer::kafka_consumer(std::map<std::string, std::string> configs) {
    DATASOURCE_ID = "librdkafka-";
    DATASOURCE_ID.append(RdKafka::version_str());

    // Construct the RdKafka::Conf object
    kafka_conf_ = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    configure_datasource(configs);
  }

  bool kafka_consumer::configure_datasource(std::map<std::string, std::string> configs) {

    //Set Kafka global configurations
    for (auto const& x : configs) {
      conf_res_ = kafka_conf_->set(x.first, x.second, errstr_);
      if (conf_res_ != RdKafka::Conf::ConfResult::CONF_OK) {
        if (conf_res_ == RdKafka::Conf::ConfResult::CONF_INVALID) {
          //TODO: I don't think failing is needed here? Just warning maybe?
        } else if (conf_res_ == RdKafka::Conf::ConfResult::CONF_UNKNOWN) {
          //TODO: I don't think failing is needed here? Just warning maybe?
        }
      }
    }

    // Kafka 0.9 > requires at least a group.id in the configuration so lets
    // make sure that is present.
    conf_res_ = kafka_conf_->get("group.id", conf_val);
    if (conf_res_ == RdKafka::Conf::ConfResult::CONF_UNKNOWN) {
      CUDF_FAIL("Kafka `group.id` was not supplied in its configuration and is required for operation");
    }

    consumer_ = std::unique_ptr<RdKafka::KafkaConsumer>(RdKafka::KafkaConsumer::create(kafka_conf_.get(), errstr_));

    return true;
  }

  std::map<std::string, std::string> kafka_consumer::current_configs() {
    std::map<std::string, std::string> configs;
    std::list<std::string> *dump = kafka_conf_->dump();
    std::string key;
    std::string val;
    for (std::list<std::string>::iterator it = dump->begin(); it != dump->end(); ) {
      key = (*it);
      it++;
      val = (*it);
      it++;
      configs.insert(std::pair<std::string, std::string>{key, val});
    }
    return configs;
  }

 int64_t kafka_consumer::get_committed_offset(std::string topic, int partition) {
    std::vector<RdKafka::TopicPartition*> toppar_list;

    // vector of always size 1. Required by underlying library
    toppar_list.push_back(RdKafka::TopicPartition::create(topic, partition));

    // Query Kafka to populate the TopicPartitions with the desired offsets
    err_ = consumer_->committed(toppar_list, default_timeout_);

    return toppar_list[0]->offset();
  }

  std::string kafka_consumer::consume_range(std::string topic,
                                              int partition,
                                              int64_t start_offset,
                                              int64_t end_offset,
                                              int batch_timeout,
                                              std::string delimiter) {
    std::string str_buffer;
    int64_t messages_read = 0;
    int64_t batch_size = end_offset - start_offset;
    int64_t end = now() + batch_timeout;
    int remaining_timeout = batch_timeout;

    update_consumer_toppar_assignment(topic, partition, start_offset);

    while (messages_read < batch_size) {
      RdKafka::Message *msg = consumer_->consume(remaining_timeout);

      if (msg->err() == RdKafka::ErrorCode::ERR_NO_ERROR) {
        str_buffer.append(static_cast<char *>(msg->payload()));
        str_buffer.append(delimiter);
        messages_read++;
      }

      remaining_timeout = end - now();
      if (remaining_timeout < 0) {
        break;
      }

      delete msg;
    }

    return str_buffer;
  }

  std::map<std::string, int64_t> kafka_consumer::get_watermark_offset(std::string topic, int partition, int timeout, bool cached) {
    int64_t low;
    int64_t high;
    std::map<std::string, int64_t> results;

    if (cached == true) {
      err_ = consumer_->get_watermark_offsets(topic, partition, &low, &high);
    } else {
      err_ = consumer_->query_watermark_offsets(topic, partition, &low, &high, timeout);
    }

    if (err_ != RdKafka::ErrorCode::ERR_NO_ERROR) {
      if (err_ == RdKafka::ErrorCode::ERR__PARTITION_EOF) {
        results.insert(std::pair<std::string, int64_t>("low", low));
        results.insert(std::pair<std::string, int64_t>("high", high));
      } else {
        throw std::runtime_error(std::string(err2str(err_).c_str()));
      }
    } else {
      results.insert(std::pair<std::string, int64_t>("low", low));
      results.insert(std::pair<std::string, int64_t>("high", high));
    }

    return results;
  }

  bool kafka_consumer::commit_offset(std::string topic, int partition, int64_t offset) {
    std::vector<RdKafka::TopicPartition*> partitions_;
    RdKafka::TopicPartition* toppar = RdKafka::TopicPartition::create(topic, partition, offset);
    if (toppar != NULL) {
      toppar->set_offset(offset);
      partitions_.push_back(toppar);
      err_ = consumer_->commitSync(partitions_);
      return true;
    } else {
      return false;
    }
  }

  bool kafka_consumer::unsubscribe() {
    err_ = consumer_.get()->unassign();
    if (err_ != RdKafka::ERR_NO_ERROR) {
      //TODO: CUDF_FAIL here or??
      printf("Timeout occurred while unsubscribing from Kafka Consumer assignments.\n");
      return false;
    } else {
      return true;
    }
  }

  bool kafka_consumer::close(int timeout) {
    err_ = consumer_.get()->close();

    if (err_ != RdKafka::ERR_NO_ERROR) {
      //TODO: CUDF_FAIL here or??
      printf("Timeout occurred while closing Kafka Consumer\n");
      return false;
    } else {
      return true;
    }

    delete consumer_.get();
    delete kafka_conf_.get();
  }

}  // namespace external
}  // namespace io
}  // namespace cudf
