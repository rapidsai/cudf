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

#include "kafka_producer.hpp"

namespace cudf {
namespace io {
namespace external {

  kafka_producer::kafka_producer() {
    DATASOURCE_ID = "librdkafka-";
    DATASOURCE_ID.append(RdKafka::version_str());

    // Create an empty RdKafka::Conf instance. The configurations will be constructed later
    kafka_conf_ = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
  }

  kafka_producer::kafka_producer(std::map<std::string, std::string> configs) {
    DATASOURCE_ID = "librdkafka-";
    DATASOURCE_ID.append(RdKafka::version_str());

    // Construct the RdKafka::Conf object
    kafka_conf_ = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    configure_datasource(configs);
  }


  bool kafka_producer::configure_datasource(std::map<std::string, std::string> configs) {

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
      //CUDF_FAIL("Kafka `group.id` was not supplied in its configuration and is required for operation");
      //TODO: What are options here? I would like to not include CUDA RT just for error handling logic alone.
    }

    producer_ = std::unique_ptr<RdKafka::Producer>(RdKafka::Producer::create(kafka_conf_.get(), errstr_));

    return true;
  }

  bool kafka_producer::produce_message(std::string topic, std::string message_val, std::string message_key) {
    err_ = producer_->produce(topic,
                       RdKafka::Topic::PARTITION_UA,
                       RdKafka::Producer::RK_MSG_COPY,
                       const_cast<char *>(message_val.c_str()),
                       message_val.size(),
                       const_cast<char *>(message_key.c_str()),
                       message_key.size(),
                       0,
                       NULL,
                       NULL);
    if (err_ != RdKafka::ERR_NO_ERROR) {
      return false;
    } else {
      return true;
    }
  }

  bool kafka_producer::flush(int timeout) {
    err_ = producer_.get()->flush(timeout);
    if (err_ != RdKafka::ERR_NO_ERROR) {
      return false;
    } else {
      return true;
    }
  }

  bool kafka_producer::close(int timeout) {
    err_ = producer_.get()->flush(timeout);
    delete producer_.get();
    delete kafka_conf_.get();
  }

}  // namespace external
}  // namespace io
}  // namespace cudf
