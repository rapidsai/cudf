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
#include <algorithm>
#include <chrono>
#include <cudf/io/datasource.hpp>
#include <map>
#include <memory>
#include <string>

namespace cudf {
namespace io {
namespace external {
namespace kafka {

/**
 * @brief libcudf datasource for Apache Kafka
 *
 * @ingroup io_datasources
 **/
class kafka_consumer : public cudf::io::datasource {
 public:
  /**
   * @brief Instantiate a Kafka consumer object. Documentation for librdkafka configurations can be
   * found at https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
   *
   * @param configs key/value pairs of librdkafka configurations that will be
   *                passed to the librdkafka client
   * @param topic_name name of the Kafka topic to consume from
   * @param partition partition index to consume from between `0` and `TOPIC_NUM_PARTITIONS - 1`
   * inclusive
   * @param start_offset seek position for the specified TOPPAR (Topic/Partition combo)
   * @param end_offset position in the specified TOPPAR to read to
   * @param batch_timeout maximum (millisecond) read time allowed. If end_offset is not reached
   * before batch_timeout, a smaller subset will be returned
   * @param delimiter optional delimiter to insert into the output between kafka messages, Ex: "\n"
   **/
  kafka_consumer(std::map<std::string, std::string> configs,
                 std::string topic_name,
                 int partition,
                 int64_t start_offset,
                 int64_t end_offset,
                 int batch_timeout,
                 std::string delimiter);

  /**
   * @brief Returns a buffer with a subset of data from Kafka Topic
   *
   * @param[in] offset Bytes from the start
   * @param[in] size Bytes to read
   *
   * @return The data buffer
   */
  std::unique_ptr<cudf::io::datasource::buffer> host_read(size_t offset, size_t size) override;

  /**
   * @brief Returns the size of the data in Kafka buffer
   *
   * @return size_t The size of the source data in bytes
   */
  size_t size() const override;

  /**
   * @brief Reads a selected range into a preallocated buffer.
   *
   * @param[in] offset Bytes from the start
   * @param[in] size Bytes to read
   * @param[in] dst Address of the existing host memory
   *
   * @return The number of bytes read (can be smaller than size)
   */
  size_t host_read(size_t offset, size_t size, uint8_t *dst) override;

  virtual ~kafka_consumer(){};

 private:
  std::unique_ptr<RdKafka::Conf> kafka_conf;  // RDKafka configuration object
  std::unique_ptr<RdKafka::KafkaConsumer> consumer;

  std::string topic_name;
  int partition;
  int64_t start_offset;
  int64_t end_offset;
  int batch_timeout;
  std::string delimiter;

  std::string buffer;

 private:
  RdKafka::ErrorCode update_consumer_topic_partition_assignment(std::string const &topic,
                                                                int partition,
                                                                int64_t offset);

  /**
   * Convenience method for getting "now()" in Kafka's standard format
   **/
  int64_t now();

  void consume_to_buffer();
};

}  // namespace kafka
}  // namespace external
}  // namespace io
}  // namespace cudf
