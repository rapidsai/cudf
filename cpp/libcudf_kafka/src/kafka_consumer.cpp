/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_kafka/kafka_consumer.hpp>

#include <librdkafka/rdkafkacpp.h>

#include <chrono>
#include <memory>

namespace cudf {
namespace io {
namespace external {
namespace kafka {

kafka_consumer::kafka_consumer(std::map<std::string, std::string> configs,
                               python_callable_type python_callable,
                               kafka_oauth_callback_wrapper_type callable_wrapper)
  : configs(configs),
    python_callable_(python_callable),
    callable_wrapper_(callable_wrapper),
    kafka_conf(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL))
{
  for (auto const& key_value : configs) {
    std::string error_string;
    CUDF_EXPECTS(RdKafka::Conf::ConfResult::CONF_OK ==
                   kafka_conf->set(key_value.first, key_value.second, error_string),
                 "Invalid Kafka configuration");
  }

  if (python_callable_ != nullptr) {
    std::string error_string;
    python_oauth_refresh_callback cb(callable_wrapper_, python_callable_);
    CUDF_EXPECTS(RdKafka::Conf::ConfResult::CONF_OK ==
                   kafka_conf->set("oauthbearer_token_refresh_cb", &cb, error_string),
                 "Failed to set Kafka oauth callback");
  }

  // Kafka 0.9 > requires group.id in the configuration
  std::string conf_val;
  CUDF_EXPECTS(RdKafka::Conf::ConfResult::CONF_OK == kafka_conf->get("group.id", conf_val),
               "Kafka group.id must be configured");

  std::string errstr;
  consumer = std::unique_ptr<RdKafka::KafkaConsumer>(
    RdKafka::KafkaConsumer::create(kafka_conf.get(), errstr));
}

kafka_consumer::kafka_consumer(std::map<std::string, std::string> configs,
                               python_callable_type python_callable,
                               kafka_oauth_callback_wrapper_type callback_wrapper,
                               std::string const& topic_name,
                               int partition,
                               int64_t start_offset,
                               int64_t end_offset,
                               int batch_timeout,
                               std::string const& delimiter)
  : configs(configs),
    python_callable_(python_callable),
    callable_wrapper_(callback_wrapper),
    topic_name(topic_name),
    partition(partition),
    start_offset(start_offset),
    end_offset(end_offset),
    batch_timeout(batch_timeout),
    delimiter(delimiter),
    kafka_conf(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL))
{
  for (auto const& key_value : configs) {
    std::string error_string;
    CUDF_EXPECTS(RdKafka::Conf::ConfResult::CONF_OK ==
                   kafka_conf->set(key_value.first, key_value.second, error_string),
                 "Invalid Kafka configuration");
  }

  if (python_callable_ != nullptr) {
    std::string error_string;
    python_oauth_refresh_callback cb(callable_wrapper_, python_callable_);
    CUDF_EXPECTS(RdKafka::Conf::ConfResult::CONF_OK ==
                   kafka_conf->set("oauthbearer_token_refresh_cb", &cb, error_string),
                 "Failed to set Kafka oauth callback");
  }

  // Kafka 0.9 > requires group.id in the configuration
  std::string conf_val;
  CUDF_EXPECTS(RdKafka::Conf::ConfResult::CONF_OK == kafka_conf->get("group.id", conf_val),
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
  return std::make_unique<non_owning_buffer>((uint8_t*)buffer.data() + offset, size);
}

size_t kafka_consumer::host_read(size_t offset, size_t size, uint8_t* dst)
{
  if (offset > buffer.size()) { return 0; }
  auto const read_size = std::min(size, buffer.size() - offset);
  memcpy(dst, buffer.data() + offset, size);
  return read_size;
}

size_t kafka_consumer::size() const { return buffer.size(); }

/**
 * Change the TOPPAR assignment for this consumer instance
 */
RdKafka::ErrorCode kafka_consumer::update_consumer_topic_partition_assignment(
  std::string const& topic, int partition, int64_t offset)
{
  std::vector<RdKafka::TopicPartition*> topic_partitions;
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
      buffer.append(static_cast<char*>(msg->payload()));
      buffer.append(delimiter);
      messages_read++;
    } else if (msg->err() == RdKafka::ErrorCode::ERR__PARTITION_EOF) {
      // If there are no more messages return
      break;
    }
  }
}

std::map<std::string, std::string> kafka_consumer::current_configs()
{
  std::map<std::string, std::string> configs;
  std::list<std::string>* dump = kafka_conf->dump();
  for (auto it = dump->begin(); it != dump->end(); std::advance(it, 2))
    configs.insert({*it, *std::next(it)});
  return configs;
}

int64_t kafka_consumer::get_committed_offset(std::string const& topic, int partition)
{
  std::vector<RdKafka::TopicPartition*> toppar_list;
  toppar_list.push_back(RdKafka::TopicPartition::create(topic, partition));

  // Query Kafka to populate the TopicPartitions with the desired offsets
  CUDF_EXPECTS(RdKafka::ERR_NO_ERROR == consumer->committed(toppar_list, default_timeout),
               "Failed retrieve Kafka committed offsets");

  int64_t offset = toppar_list[0]->offset();
  return offset > 0 ? offset : -1001;
}

std::map<std::string, std::vector<int32_t>> kafka_consumer::list_topics(std::string specific_topic)
{
  auto const metadata = [&]() {
    std::string errstr;
    auto spec_topic = std::unique_ptr<RdKafka::Topic>(
      RdKafka::Topic::create(consumer.get(), specific_topic, nullptr, errstr));

    RdKafka::Metadata* md;
    CUDF_EXPECTS(
      RdKafka::ERR_NO_ERROR ==
        consumer->metadata(spec_topic == nullptr, spec_topic.get(), &md, default_timeout),
      "Failed to list_topics in Kafka broker");
    return std::unique_ptr<RdKafka::Metadata>{md};
  }();
  std::map<std::string, std::vector<int32_t>> topic_parts;

  for (auto const& topic : *(metadata->topics())) {
    auto& part_ids    = topic_parts[topic->topic()];
    auto const& parts = *(topic->partitions());
    std::transform(
      parts.cbegin(), parts.cend(), std::back_inserter(part_ids), [](auto const& part) {
        return part->id();
      });
  }

  return topic_parts;
}

std::map<std::string, int64_t> kafka_consumer::get_watermark_offset(std::string const& topic,
                                                                    int partition,
                                                                    int timeout,
                                                                    bool cached)
{
  int64_t low;
  int64_t high;
  std::map<std::string, int64_t> results;
  RdKafka::ErrorCode err;

  if (cached) {
    err = consumer->get_watermark_offsets(topic, partition, &low, &high);
  } else {
    err = consumer->query_watermark_offsets(topic, partition, &low, &high, timeout);
  }

  if (err != RdKafka::ErrorCode::ERR_NO_ERROR) {
    if (err == RdKafka::ErrorCode::ERR__PARTITION_EOF) {
      results.insert(std::pair<std::string, int64_t>("low", low));
      results.insert(std::pair<std::string, int64_t>("high", high));
    } else {
      CUDF_FAIL("Error retrieving Kafka watermark offset from broker");
    }
  } else {
    results.insert(std::pair<std::string, int64_t>("low", low));
    results.insert(std::pair<std::string, int64_t>("high", high));
  }

  return results;
}

void kafka_consumer::commit_offset(std::string const& topic, int partition, int64_t offset)
{
  std::vector<RdKafka::TopicPartition*> partitions_;
  RdKafka::TopicPartition* toppar = RdKafka::TopicPartition::create(topic, partition, offset);
  CUDF_EXPECTS(toppar != nullptr, "RdKafka failed to create TopicPartition");
  toppar->set_offset(offset);
  partitions_.push_back(toppar);
  CUDF_EXPECTS(RdKafka::ERR_NO_ERROR == consumer->commitSync(partitions_),
               "Failed to commit consumer offsets");
}

void kafka_consumer::unsubscribe()
{
  CUDF_EXPECTS(RdKafka::ErrorCode::ERR_NO_ERROR == consumer.get()->unassign(),
               "Failed to unsubscribe from Kafka Consumer");
}

void kafka_consumer::close(int timeout)
{
  CUDF_EXPECTS(RdKafka::ERR_NO_ERROR == consumer->close(), "Failed to close Kafka consumer");
  consumer.reset(nullptr);
  kafka_conf.reset(nullptr);
}

}  // namespace kafka
}  // namespace external
}  // namespace io
}  // namespace cudf
