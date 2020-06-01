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
{
  kafka_conf_ = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));

  // Ignore 'errstr_' values. librdkafka guards against "invalid" values. 'errstr_' is only warning
  // if improper key is provided, we ignore those messages from the consumer
  for (auto const &x : configs) { kafka_conf_->set(x.first, x.second, errstr_); }

  // Kafka 0.9 > requires at least a group.id in the configuration so lets
  // make sure that is present.
  conf_res_ = kafka_conf_->get("group.id", conf_val);
  if (conf_res_ == RdKafka::Conf::ConfResult::CONF_UNKNOWN) {
    CUDF_FAIL(
      "Kafka `group.id` was not supplied in its configuration and is required for operation");
  }

  consumer_ = std::unique_ptr<RdKafka::KafkaConsumer>(
    RdKafka::KafkaConsumer::create(kafka_conf_.get(), errstr_));

  // We read fill the local buffer with messages so the datasource->size() invocation
  // will return a valid size.
  int64_t messages_read = 0;
  int64_t batch_size    = end_offset - start_offset;
  int64_t end           = now() + batch_timeout;
  int remaining_timeout = batch_timeout;

  update_consumer_toppar_assignment(topic_name, partition, start_offset);

  while (messages_read < batch_size) {
    RdKafka::Message *msg = consumer_->consume(remaining_timeout);

    if (msg->err() == RdKafka::ErrorCode::ERR_NO_ERROR) {
      buffer_.append(static_cast<char *>(msg->payload()));
      buffer_.append(delimiter);
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
