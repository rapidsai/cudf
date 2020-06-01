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
#include <cudf/io/datasource.hpp>
#include <map>
#include <memory>
#include <string>

namespace cudf {
namespace io {
namespace external {
namespace kafka {

/**
 * @brief implementation for holding kafka messages
 **/
class message_buffer : public cudf::io::datasource::buffer {
 public:
  message_buffer(const char *message_delimter) : _message_delimiter(message_delimter) {}

  size_t size() const { return _buffer.size(); }

  const uint8_t *data() const { return 0; }

  bool add_message(RdKafka::Message *msg)
  {
    _buffer.append(static_cast<char *>(msg->payload()));
    _buffer.append(_message_delimiter);
  }

 private:
  const char *_message_delimiter;
  std::string _buffer;
};

/**
 * @brief libcudf external datasource for Apache Kafka
 **/
class kafka_consumer : public cudf::io::datasource {
 public:
  /**
   * @brief Create a fully capable Kafka Consumer instance that can
   *consume/produce
   *
   * @param configs key/value pairs of librdkafka configurations that will be
   *passed to the librdkafka client
   **/
  kafka_consumer(std::map<std::string, std::string> configs);

  std::unique_ptr<cudf::io::datasource::buffer> host_read(size_t offset, size_t size);

  /**
   * @brief Acknowledge messages have been successfully read to the Kafka
   *cluster
   *
   * @param topic Name of the topic the offset should be set for
   * @param partition Topic partition for the offset
   * @param offset The offset value that should be applied as the last read
   *message
   *
   * @return True on success and False otherwise
   **/
  bool commit_offset(std::string topic, int partition, int64_t offset);

  /**
   * @brief Retrieves the earliest and latest message offsets for the specified
   *TOPPAR
   *
   * @param topic Name of the topic the offset should be set for
   * @param partition Topic partition for the offset
   * @param timeout how long the operation should wait for a response from the
   *Kafka server before throwing error
   * @param cached True query Kafka server, False use the last response received
   *cache value
   *
   * @return Map containing keys "low" & "high" along with the int64_t offset
   *for the specified TOPPAR instance
   **/
  std::map<std::string, int64_t> get_watermark_offset(std::string topic,
                                                      int partition,
                                                      int timeout,
                                                      bool cached);

  /**
   * @brief Retrieves the latest committed offset for a TOPPAR instance
   *
   * @param topic Kafka Topic name
   * @param partition Associated Topic partition number
   *
   * @return Offset of the latest commiited offset
   **/
  int64_t get_committed_offset(std::string topic, int partition);

  /**
   * @brief Read messages from a Kafka TOPPAR based on parameters
   *
   * @param topic Name of Kafka topic to read from
   * @param partition Partition in the Topic to read from
   * @param start_offset Beginning offset for the read operation
   * @param end_offset Last message that should be read from the TOPPAR
   * @param timeout Millisecond timeout before the read operation should fail
   * @param delimiter The delimiter that should be applied to the concatenated
   *messages before being sent to cuDF
   *
   * @return String with all of the individual messages from Kafka concatenated
   *together ready for handoff to cuDF
   **/
  std::string consume_range(std::string topic,
                            int partition,
                            int64_t start_offset,
                            int64_t end_offset,
                            int batch_timeout,
                            std::string delimiter);

  /**
   * @brief Invoke librdkafka unsubscribe from the Kafka TOPPAR instance
   *
   * @return True if success or False otherwise
   **/
  bool unsubscribe();

  /**
   * @brief Close and free all socket, memory, and filesystem resources used by
   *this consumer
   *
   * @return True on success or False otherwise
   **/
  bool close(int timeout);

  /**
   * @brief Base class destructor
   **/
  virtual ~kafka_consumer(){};

 protected:
  std::unique_ptr<RdKafka::Conf> kafka_conf_;  // RDKafka configuration object
  std::unique_ptr<RdKafka::KafkaConsumer> consumer_ = NULL;

  RdKafka::Conf::ConfResult conf_res_;  // Result from configuration update operation
  RdKafka::ErrorCode err_;              // RDKafka ErrorCode from operation
  std::string errstr_;                  // Textual representation of Error
  std::string conf_val;                 // String value of a RDKafka configuration request
  int32_t default_timeout_ = 10000;     // Default timeout for server bound operations - 10 seconds

  int64_t kafka_start_offset_ = 0;
  int32_t kafka_batch_size_   = 10000;  // 10K is the Kafka standard. Max is 999,999
  int64_t msg_count_          = 0;      // Running tally of the messages consumed
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
};

}  // namespace kafka
}  // namespace external
}  // namespace io
}  // namespace cudf
