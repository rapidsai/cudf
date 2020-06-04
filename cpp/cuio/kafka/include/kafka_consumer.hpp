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
#pragma once

#include <librdkafka/rdkafkacpp.h>
#include <sys/time.h>
#include <algorithm>
#include <cudf/io/datasource.hpp>
#include <map>
#include <memory>
#include <string>

namespace cudf {
namespace io {
namespace external {
namespace kafka {

/**
 * @brief libcudf external datasource for Apache Kafka
 **/
class kafka_consumer : public cudf::io::datasource {
  /**
   * @brief Implementation for holding kafka messages
   **/
  class message_buffer : public buffer {
   public:
    message_buffer(uint8_t *data, size_t size) : _data(data), _size(size) {}

    size_t size() const { return _size; }

    const uint8_t *data() const { return _data; }

   private:
    uint8_t *const _data;
    size_t const _size;
  };

 public:
  /**
   * @brief Instantiate a Kafka consumer object
   *
   * @param configs key/value pairs of librdkafka configurations that will be
   *                passed to the librdkafka client
   * @param topic_name name of the Kafka topic to consume from
   * @param partition partition to consume from; 0 - (TOPIC_NUM_PARTITIONS - 1)
   * @param start_offset seek position for the specified TOPPAR (Topic/Partition combo)
   * @param end_offset position in the specified TOPPAR to read to
   * @param batch_timeout maximum read time allowed. If end_offset is not reached before
   *                      batch_timeout is a smaller subset will be returned
   * @param delimiter optional delimiter that should be placed between kafka messages, Ex: "\n"
   **/
  kafka_consumer(std::map<std::string, std::string> configs,
                 std::string topic_name,
                 int partition,
                 int64_t start_offset,
                 int64_t end_offset,
                 int batch_timeout,
                 std::string delimiter);

  std::unique_ptr<cudf::io::datasource::buffer> host_read(size_t offset, size_t size);

  size_t size() const;

  size_t host_read(size_t offset, size_t size, uint8_t *dst);

  /**
   * @brief Base class destructor
   **/
  virtual ~kafka_consumer(){};

 protected:
  std::unique_ptr<RdKafka::Conf> kafka_conf_;  // RDKafka configuration object
  std::unique_ptr<RdKafka::KafkaConsumer> consumer_ = NULL;

  RdKafka::Conf::ConfResult conf_res_;  // Result from configuration update operation
  std::string errstr_;                  // Textual representation of Error
  std::string conf_val;                 // String value of a RDKafka configuration request

  std::string topic_name_;
  int partition_;
  int64_t start_offset_;
  int64_t end_offset_;
  int batch_timeout_;
  std::string delimiter_;

  std::string buffer_;

 private:
  /**
   * Change the TOPPAR assignment for this consumer instance
   **/
  RdKafka::ErrorCode update_consumer_toppar_assignment(std::string topic,
                                                       int partition,
                                                       int64_t offset)
  {
    std::vector<RdKafka::TopicPartition *> _toppars;
    _toppars.push_back(RdKafka::TopicPartition::create(topic, partition, offset));
    return consumer_.get()->assign(_toppars);
  }

  /**
   * Convenience method for getting "now()" in Kafka's standard format
   **/
  int64_t now()
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((int64_t)tv.tv_sec * 1000) + (tv.tv_usec / 1000);
  }

  void consume_to_buffer()
  {
    int64_t messages_read = 0;
    int64_t batch_size    = end_offset_ - start_offset_;
    int64_t end           = now() + batch_timeout_;
    int remaining_timeout = batch_timeout_;

    update_consumer_toppar_assignment(topic_name_, partition_, start_offset_);

    while (messages_read < batch_size) {
      RdKafka::Message *msg = consumer_->consume(remaining_timeout);

      if (msg->err() == RdKafka::ErrorCode::ERR_NO_ERROR) {
        buffer_.append(static_cast<char *>(msg->payload()));
        buffer_.append(delimiter_);
        messages_read++;
      }

      remaining_timeout = end - now();
      if (remaining_timeout < 0) {
        delete msg;
        break;
      }

      delete msg;
    }
  }
};

}  // namespace kafka
}  // namespace external
}  // namespace io
}  // namespace cudf
