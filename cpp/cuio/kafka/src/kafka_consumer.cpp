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
  : topic_name_(topic_name),
    partition_(partition),
    start_offset_(start_offset),
    end_offset_(end_offset),
    batch_timeout_(batch_timeout),
    delimiter_(delimiter)
{
  kafka_conf_ = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));

  // Set configs and check for invalids
  for (auto const &x : configs) {
    conf_res_ = kafka_conf_->set(x.first, x.second, errstr_);
    if (conf_res_ == RdKafka::Conf::ConfResult::CONF_UNKNOWN) {
      CUDF_FAIL("'" + x.first + ", is an invalid librdkafka configuration property");
    } else if (conf_res_ == RdKafka::Conf::ConfResult::CONF_INVALID) {
      CUDF_FAIL("'" + x.second +
                "' contains an invalid configuration value for librdkafka property '" + x.first +
                "'");
    }
  }

  // Kafka 0.9 > requires group.id in the configuration
  conf_res_ = kafka_conf_->get("group.id", conf_val);
  if (conf_res_ == RdKafka::Conf::ConfResult::CONF_UNKNOWN) {
    CUDF_FAIL(
      "Kafka `group.id` was not supplied in its configuration and is required for operation");
  }

  consumer_ = std::unique_ptr<RdKafka::KafkaConsumer>(
    RdKafka::KafkaConsumer::create(kafka_conf_.get(), errstr_));

  // Pre fill the local buffer with messages so the datasource->size() invocation
  // will return a valid size.
  consume_to_buffer();
}

std::unique_ptr<cudf::io::datasource::buffer> kafka_consumer::host_read(size_t offset, size_t size)
{
  size = std::min(size, buffer_.size() - offset);
  return std::make_unique<message_buffer>((uint8_t *)buffer_.data() + offset, size);
}

size_t kafka_consumer::host_read(size_t offset, size_t size, uint8_t *dst)
{
  auto const read_size = std::min(size, buffer_.size() - offset);
  memcpy(dst, buffer_.data() + offset, size);
  return read_size;
}

size_t kafka_consumer::size() const { return buffer_.size(); }

}  // namespace kafka
}  // namespace external
}  // namespace io
}  // namespace cudf
