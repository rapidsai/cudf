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
 * @brief Apache Kafka Producer for libcudf
 **/
class kafka_producer : public kafka_datasource {

 public:

   /**
    * @brief Create Kafka Producer instance that is unable to consume/produce
   * but is able to assist with configurations
   **/
   kafka_producer();

   /**
   * @brief Create a fully capable Kafka Producer instance
   * 
   * @param configs key/value pairs of librdkafka configurations that will be passed to the librdkafka client
   **/
   kafka_producer(std::map<std::string, std::string> configs);

   /**
   * @brief Applies the specified configurations to the underlying librdkafka client
   * 
   * @param configs Map of key/value pairs that represent the librdkafka configurations to be applied
   * 
   * @return True on success or False otherwise
   **/
  bool configure_datasource(std::map<std::string, std::string> configs);

  /**
   * @brief Sends a message to the specified Kafka topic using the message_key provided
   * 
   * @param topic Name of the Apache Kafka topic to produce the message to
   * @param message_val The message payload that will be sent to Kafka
   * @param message_key Key used to determine which partition in the Kafka cluster the message will be written to
   * 
   * @return True on success or False otherwise
   **/
  bool produce_message(std::string topic, std::string message_val, std::string message_key);

  /**
   * @brief Flush/write any locally pending messages to Kafka now
   * 
   * @param timeout Milliseconds to wait for failing the operation.
   * 
   * @return True on success or False otherwise
   **/
  bool flush(int timeout);

  /**
   * @brief Free up resources, flush, and close the underlying connection and configuration instances
   * 
   * @param timeout How long to wait on flush before failing
   * 
   * @return True on success or False otherwise
   **/
  bool close(int timeout);

 private:

    std::unique_ptr<RdKafka::Producer> producer_ = NULL;

};

}  // namespace external
}  // namespace io
}  // namespace cudf
