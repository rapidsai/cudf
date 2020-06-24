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

#include "cudf_kafka/kafka_consumer.hpp"
#include <librdkafka/rdkafkacpp.h>
#include <chrono>
#include <memory>

namespace cudf {
namespace io {
namespace external {
namespace kafka {

kafka_consumer::kafka_consumer(std::map<std::string, std::string> configs,
                               std::string topic_name,
                               int partition,
                               int64_t start_offset,
                               int64_t end_offset,
                               int batch_timeout,
                               std::string delimiter)
  : topic_name(topic_name),
    partition(partition),
    start_offset(start_offset),
    end_offset(end_offset),
    batch_timeout(batch_timeout),
    delimiter(delimiter)
{
  kafka_conf = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));

  for (auto const &key_value : configs) {
    std::string error_string;
    CUDF_EXPECTS(RdKafka::Conf::ConfResult::CONF_OK ==
                   kafka_conf->set(key_value.first, key_value.second, error_string),
                 "Invalid Kafka configuration");
  }

  // Kafka 0.9 > requires group.id in the configuration
  std::string conf_val;
  CUDF_EXPECTS(RdKafka::Conf::ConfResult::CONF_OK == kafka_conf->get("group_id", conf_val),
               "Kafka group.id must be configured");

  std::string errstr;
  consumer = std::unique_ptr<RdKafka::KafkaConsumer>(
    RdKafka::KafkaConsumer::create(kafka_conf.get(), errstr));

  // Pre fill the local buffer with messages so the datasource->size() invocation
  // will return a valid size.
  consume_to_buffer();
}

std::unique_ptr<cudf::io::datasource::buffer> kafka_consumer::host_read(size_t offset, size_t size)
{
  if (offset > buffer.size()) { return 0; }
  size = std::min(size, buffer.size() - offset);
  return std::make_unique<non_owning_buffer>((uint8_t *)buffer.data() + offset, size);
}

size_t kafka_consumer::host_read(size_t offset, size_t size, uint8_t *dst)
{
  if (offset > buffer.size()) { return 0; }
  auto const read_size = std::min(size, buffer.size() - offset);
  memcpy(dst, buffer.data() + offset, size);
  return read_size;
}

size_t kafka_consumer::size() const { return buffer.size(); }

/**
 * Change the TOPPAR assignment for this consumer instance
 **/
RdKafka::ErrorCode kafka_consumer::update_consumer_topic_partition_assignment(
  std::string const &topic, int partition, int64_t offset)
{
  std::vector<RdKafka::TopicPartition *> topic_partitions;
  topic_partitions.push_back(RdKafka::TopicPartition::create(topic, partition, offset));
  return consumer.get()->assign(topic_partitions);
}

void kafka_consumer::consume_to_buffer()
{
  update_consumer_topic_partition_assignment(topic_name, partition, start_offset);

  int64_t messages_read = 0;
  auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(batch_timeout);

  while (messages_read < end_offset - start_offset && end > std::chrono::steady_clock::now()) {
    std::unique_ptr<RdKafka::Message> msg{
      consumer->consume((end - std::chrono::steady_clock::now()).count())};

    if (msg->err() == RdKafka::ErrorCode::ERR_NO_ERROR) {
      buffer.append(static_cast<char *>(msg->payload()));
      buffer.append(delimiter);
      messages_read++;
    }
  }
}

}  // namespace kafka
}  // namespace external
}  // namespace io
}  // namespace cudf
