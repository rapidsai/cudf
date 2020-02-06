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

#include "external_datasource.hpp"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <map>

#include <chrono>
#include <thread>

#include <cudf/cudf.h>
#include <librdkafka/rdkafkacpp.h>

namespace cudf {
namespace io {
namespace external {

/**
 * @brief External Datasource for Apache Kafka
 **/
class kafka_datasource : public external_datasource {
 public:

  kafka_datasource() {
    DATASOURCE_ID = "librdkafka-1.2.2";

    // Create an empty RdKafka::Conf instance. The configurations will be constructed later
    kafka_conf_ = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
  }

  kafka_datasource(std::map<std::string, std::string> configs) {
    DATASOURCE_ID = "librdkafka-1.2.2";

    // Construct the RdKafka::Conf object
    kafka_conf_ = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    configure_datasource(configs);
  }

  std::string libcudf_datasource_identifier() {
    return DATASOURCE_ID;
  }

  bool configure_datasource(std::map<std::string, std::string> configs) {
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

  const std::shared_ptr<arrow::Buffer> get_buffer(size_t offset,
                                                  size_t size) override {
    return arrow::Buffer::Wrap(buffer_.c_str(), buffer_.size());
  }

  size_t size() const override { return buffer_.size(); }

  /**
   * @brief Base class destructor
   **/
  virtual ~kafka_datasource(){};

  private:

    void consume_messages(std::unique_ptr<RdKafka::Conf> const &kafka_conf) {
      // Kafka messages are already stored in a queue outside of libcudf. Here the
      // messages will be transferred from the external queue directly to the
      // arrow::Buffer.
      RdKafka::Message *msg;

      for (int i = 0; i < kafka_batch_size_; i++) {
        printf("\tMessage read\n");
        msg = consumer_->consume(default_timeout_);
        if (msg->err() == RdKafka::ErrorCode::ERR_NO_ERROR) {
          buffer_.append(static_cast<char *>(msg->payload()));
          buffer_.append("\n");
          msg_count_++;
        } else {
          handle_error(msg, kafka_conf);

          // handle_error handles specific errors. Any coded logic error case will
          // generate an exception and cease execution. Kafka has hundreds of
          // possible exceptions however. To be safe its best break the consumer loop.
          break;
        }
      }

      printf("Buffer: '%s'\n", buffer_.c_str());

      delete msg;
    }

    class KafkaRebalanceCB : public RdKafka::RebalanceCb {
      public:
        KafkaRebalanceCB(int64_t start_offset) : start_offset_(start_offset) {}

        void rebalance_cb(RdKafka::KafkaConsumer *consumer, RdKafka::ErrorCode err,
                          std::vector<RdKafka::TopicPartition *> &partitions) {
          if (err == RdKafka::ERR__ASSIGN_PARTITIONS) {
            // NOTICE: We currently purposely only support a single partition. Enhancement PR to be opened later.
            partitions.at(0)->set_offset(start_offset_);
            err = consumer->assign(partitions);
            //CUDF_EXPECTS(err == RdKafka::ErrorCode::ERR_NO_ERROR,
              //          "Error occured while reassigning the topic partition offset");
          } else {
            consumer->unassign();
          }
        }

      private:
        int64_t start_offset_;
    };

    void handle_error(RdKafka::Message *msg, std::unique_ptr<RdKafka::Conf> const &kafka_conf) {
      err_ = msg->err();
      const std::string err_str = msg->errstr();
      std::string error_msg;

      if (msg_count_ == 0 &&
          err_ == RdKafka::ErrorCode::ERR__PARTITION_EOF) {
        // The topic was empty and had no data in it. Most likely best to error
        // here since the most likely cause of this would be a user entering the
        // wrong topic name.
        error_msg.append("Kafka Topic '");
        error_msg.append(topics_.at(0).c_str());
        error_msg.append("' is empty or does not exist on broker(s)");
        //CUDF_FAIL(error_msg);
      } else if (msg_count_ == 0 &&
                err_ == RdKafka::ErrorCode::ERR__TIMED_OUT) {
        // unable to connect to the specified Kafka Broker(s)
        std::string brokers_val;
        conf_res_ = kafka_conf->get("metadata.broker.list", brokers_val);
        if (brokers_val.empty()) {
          // 'bootstrap.servers' is an alias configuration so its valid that
          // either 'metadata.broker.list' or 'bootstrap.servers' is set
          conf_res_ = kafka_conf->get("bootstrap.servers", brokers_val);
        }

        if (conf_res_ == RdKafka::Conf::ConfResult::CONF_OK) {
          error_msg.append("Connection attempt to Kafka broker(s) '");
          error_msg.append(brokers_val);
          error_msg.append("' timed out.");
          //CUDF_FAIL(error_msg);
        } else {
          //CUDF_FAIL(
          //    "No Kafka broker(s) were specified for connection. Connection "
          //    "Failed.");
        }
      } else if (err_ == RdKafka::ErrorCode::ERR__PARTITION_EOF) {
        // Kafka treats PARTITION_EOF as an "error". In our Rapids use case it is
        // not however and just means all messages have been read.
        // Just print imformative message and break consume loop.
        printf("%ld messages read from Kafka\n", msg_count_);
      }
    }

  private:
    std::unique_ptr<RdKafka::Conf> kafka_conf_;
    std::unique_ptr<RdKafka::KafkaConsumer> consumer_;
    RdKafka::Conf::ConfResult conf_res_;
    RdKafka::ErrorCode err_;

    std::vector<std::string> topics_;
    std::string errstr_;
    
    std::string conf_val;
    int64_t kafka_start_offset_ = 0;
    int32_t kafka_batch_size_ = 10000;  // 10K is the Kafka standard. Max is 999,999
    int32_t default_timeout_ = 10000;  // 10 seconds
    int64_t msg_count_ = 0;  // Running tally of the messages consumed. Useful for retry logic.

    std::string buffer_;
};

extern "C" external_datasource* libcudf_external_datasource_load() {
  return new kafka_datasource;
}

extern "C" external_datasource* libcudf_external_datasource_load_from_conf(std::map<std::string, std::string>& configs) {
  return new kafka_datasource(configs);
}

extern "C" void libcudf_external_datasource_destroy(external_datasource* eds) {
  delete eds;
}

}  // namespace external
}  // namespace io
}  // namespace cudf
