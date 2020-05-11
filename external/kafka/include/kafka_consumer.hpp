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

/**
 * @brief libcudf external datasource for Apache Kafka
 **/
class kafka_consumer : public kafka_datasource {

 protected:
    std::unique_ptr<RdKafka::KafkaConsumer> consumer_ = NULL;
    int64_t kafka_start_offset_ = 0;
    int32_t kafka_batch_size_ = 10000;  // 10K is the Kafka standard. Max is 999,999
    int64_t msg_count_ = 0;  // Running tally of the messages consumed. Useful for retry logic.
    //volatile sig_atomic_t consumer_connected = 1;
    std::string buffer_;

 public:

  kafka_consumer();

  kafka_consumer(std::map<std::string, std::string> configs);

  bool commit_offset(std::string topic, int partition, int64_t offset);

  std::map<std::string, int64_t> get_watermark_offset(std::string topic, int partition, int timeout, bool cached);

  bool configure_datasource(std::map<std::string, std::string> configs);

  std::map<std::string, std::string> current_configs();

  int64_t get_committed_offset(std::string topic, int partition);

  std::string consume_range(std::string topic, int partition, int64_t start_offset, int64_t end_offset, int batch_timeout, std::string delimiter);

  bool unsubscribe();

  bool close(int timeout);

  const std::shared_ptr<arrow::Buffer> get_buffer(size_t offset,
                                                  size_t size) override {
    return arrow::Buffer::Wrap(buffer_.c_str(), buffer_.size());
  }

  size_t size() const override { return buffer_.size(); }

  /**
   * @brief Base class destructor
   **/
  virtual ~kafka_consumer(){};

  private:

    RdKafka::ErrorCode update_consumer_toppar_assignment(std::string topic, int partition, int64_t offset) {
      std::vector<RdKafka::TopicPartition*> _toppars;
      _toppars.push_back(RdKafka::TopicPartition::create(topic, partition, offset));
      consumer_.get()->assign(_toppars);
    }

    void handle_error(RdKafka::Message *msg) {
      err_ = msg->err();
      const std::string err_str = msg->errstr();
      std::string error_msg;

      if (msg_count_ == 0 &&
                err_ == RdKafka::ErrorCode::ERR__TIMED_OUT) {
        // unable to connect to the specified Kafka Broker(s)
        std::string brokers_val;
        conf_res_ = kafka_conf_->get("metadata.broker.list", brokers_val);
        if (brokers_val.empty()) {
          // 'bootstrap.servers' is an alias configuration so its valid that
          // either 'metadata.broker.list' or 'bootstrap.servers' is set
          conf_res_ = kafka_conf_->get("bootstrap.servers", brokers_val);
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
};

extern "C" external_datasource* libcudf_external_datasource_load() {
  return new kafka_consumer;
}

extern "C" external_datasource* libcudf_external_datasource_load_from_conf(std::map<std::string, std::string>& configs) {
  return new kafka_consumer(configs);
}

extern "C" void libcudf_external_datasource_destroy(external_datasource* eds) {
  delete eds;
}

}  // namespace external
}  // namespace io
}  // namespace cudf
