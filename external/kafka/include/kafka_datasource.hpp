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
#include <sys/time.h>

//#include <cudf/cudf.h>
#include <librdkafka/rdkafkacpp.h>

namespace cudf {
namespace io {
namespace external {

/**
 * @brief External Datasource for Apache Kafka
 **/
class kafka_datasource : public external_datasource {

 private:
    std::unique_ptr<RdKafka::Conf> kafka_conf_;
    std::unique_ptr<RdKafka::KafkaConsumer> consumer_ = NULL;
    std::unique_ptr<RdKafka::Producer> producer_ = NULL;


    
    std::vector<RdKafka::TopicPartition*> partitions_;
    RdKafka::Conf::ConfResult conf_res_;
    RdKafka::ErrorCode err_;

    std::string errstr_;
    
    std::string conf_val;
    int64_t kafka_start_offset_ = 0;
    int32_t kafka_batch_size_ = 10000;  // 10K is the Kafka standard. Max is 999,999
    int32_t default_timeout_ = 6000;  // 10 seconds
    int64_t msg_count_ = 0;  // Running tally of the messages consumed. Useful for retry logic.

    std::string buffer_;

 public:

  kafka_datasource();

  kafka_datasource(std::map<std::string, std::string> configs, std::vector<std::string> topics, std::vector<int> partitions);

  std::string libcudf_datasource_identifier();

  bool commit_offset(std::string topic, int partition, int64_t offset);

  std::map<std::string, int64_t> get_watermark_offset(std::string topic, int partition);

  bool configure_datasource(std::map<std::string, std::string> configs, std::vector<std::string> topics, std::vector<int> partitions);

  void print_consumer_metadata();

  void dump_configs();

  std::map<int, int64_t> get_committed_offset(std::string topic, std::vector<int> partitions);

  std::string consume_range(int64_t start_offset, int64_t end_offset, int batch_timeout, std::string delimiter);

  bool produce_message(std::string topic, std::string message_val, std::string message_key);

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

    RdKafka::TopicPartition* find_toppar(std::string topic, int partition) {
      for (int i = 0; i < partitions_.size(); i++) {
        if ((partitions_[i]->topic().compare(topic) == 0) && partitions_[i]->partition() == partition) {
          return partitions_[i];
        }
      }
      return NULL;
    }

    int64_t now() {
      struct timeval tv;
      gettimeofday(&tv, NULL);
      return ((int64_t)tv.tv_sec * 1000) + (tv.tv_usec / 1000);
    }

    class ExampleRebalanceCb : public RdKafka::RebalanceCb {
      private:
        static void part_list_print (const std::vector<RdKafka::TopicPartition*>&partitions){
          for (unsigned int i = 0 ; i < partitions.size() ; i++)
            std::cerr << partitions[i]->topic() <<
        "[" << partitions[i]->partition() << "], ";
          std::cerr << "\n";
        }

      public:
        void rebalance_cb (RdKafka::KafkaConsumer *consumer,
              RdKafka::ErrorCode err,
                          std::vector<RdKafka::TopicPartition*> &partitions) {
          std::cerr << "RebalanceCb: " << RdKafka::err2str(err) << ": ";

          part_list_print(partitions);

          if (err == RdKafka::ERR__ASSIGN_PARTITIONS) {
            consumer->assign(partitions);
          } else {
            consumer->unassign();
          }
        }
      };

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
  return new kafka_datasource;
}

extern "C" external_datasource* libcudf_external_datasource_load_from_conf(std::map<std::string, std::string>& configs, std::vector<std::string>& topics, std::vector<int>& partitions) {
  return new kafka_datasource(configs, topics, partitions);
}

extern "C" void libcudf_external_datasource_destroy(external_datasource* eds) {
  delete eds;
}

}  // namespace external
}  // namespace io
}  // namespace cudf
