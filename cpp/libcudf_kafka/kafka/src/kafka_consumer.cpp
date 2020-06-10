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

#include "kafka_consumer.hpp"
#include <librdkafka/rdkafkacpp.h>

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

  // Set configs and check for invalids
  RdKafka::Conf::ConfResult conf_res;
  std::string errstr;    // Textual representation of Error
  std::string conf_val;  // String value of a RDKafka configuration request

  for (auto const &x : configs) {
    conf_res = kafka_conf->set(x.first, x.second, errstr);
    if (conf_res == RdKafka::Conf::ConfResult::CONF_UNKNOWN) {
      CUDF_FAIL("'" + x.first + ", is an invalid librdkafka configuration property");
    } else if (conf_res == RdKafka::Conf::ConfResult::CONF_INVALID) {
      CUDF_FAIL("'" + x.second +
                "' contains an invalid configuration value for librdkafka property '" + x.first +
                "'");
    }
  }

  // Kafka 0.9 > requires group.id in the configuration
  conf_res = kafka_conf->get("group.id", conf_val);
  if (conf_res == RdKafka::Conf::ConfResult::CONF_UNKNOWN) {
    CUDF_FAIL(
      "Kafka `group.id` was not supplied in its configuration and is required for operation");
  }

  consumer = std::unique_ptr<RdKafka::KafkaConsumer>(
    RdKafka::KafkaConsumer::create(kafka_conf.get(), errstr));

  // Pre fill the local buffer with messages so the datasource->size() invocation
  // will return a valid size.
  consume_to_buffer();
}

std::unique_ptr<cudf::io::datasource::buffer> kafka_consumer::host_read(size_t offset, size_t size)
{
  if (offset > buffer.size()) { return std::make_unique<non_owning_buffer>(); }
  size = std::min(size, buffer.size() - offset);
  return std::make_unique<non_owning_buffer>((uint8_t *)buffer.data() + offset, size);
}

size_t kafka_consumer::host_read(size_t offset, size_t size, uint8_t *dst)
{
  auto const read_size = std::min(size, buffer.size() - offset);
  memcpy(dst, buffer.data() + offset, size);
  return read_size;
}

size_t kafka_consumer::size() const { return buffer.size(); }

/**
 * Change the TOPPAR assignment for this consumer instance
 **/
RdKafka::ErrorCode kafka_consumer::update_consumer_toppar_assignment(std::string const &topic,
                                                                     int partition,
                                                                     int64_t offset)
{
  std::vector<RdKafka::TopicPartition *> _toppars;
  _toppars.push_back(RdKafka::TopicPartition::create(topic, partition, offset));
  return consumer.get()->assign(_toppars);
}

/**
 * Convenience method for getting "now()" in Kafka's standard format
 **/
int64_t kafka_consumer::now()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return ((int64_t)tv.tv_sec * 1000) + (tv.tv_usec / 1000);
}

void kafka_consumer::consume_to_buffer()
{
  int64_t messages_read = 0;
  int64_t batch_size    = end_offset - start_offset;
  int64_t end           = now() + batch_timeout;
  int remaining_timeout = batch_timeout;

  update_consumer_toppar_assignment(topic_name, partition, start_offset);

  while (messages_read < batch_size) {
    std::unique_ptr<RdKafka::Message> msg{consumer->consume(remaining_timeout)};

    if (msg->err() == RdKafka::ErrorCode::ERR_NO_ERROR) {
      buffer.append(static_cast<char *>(msg->payload()));
      buffer.append(delimiter);
      messages_read++;
    }

    remaining_timeout = end - now();
    if (remaining_timeout < 0) { break; }
  }
}

}  // namespace kafka
}  // namespace external
}  // namespace io
}  // namespace cudf
