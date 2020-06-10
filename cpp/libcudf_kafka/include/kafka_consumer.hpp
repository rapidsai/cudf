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

  std::unique_ptr<cudf::io::datasource::buffer> host_read(size_t offset, size_t size) override;

  size_t size() const override;

  size_t host_read(size_t offset, size_t size, uint8_t *dst) override;

  /**
   * @brief Base class destructor
   **/
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
  RdKafka::ErrorCode update_consumer_toppar_assignment(std::string const &topic,
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
