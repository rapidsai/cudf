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

  kafka_datasource::kafka_datasource() {
    DATASOURCE_ID = "librdkafka-";
    DATASOURCE_ID.append(RdKafka::version_str());

    // Create an empty RdKafka::Conf instance. The configurations will be constructed later
    kafka_conf_ = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
  }

  kafka_datasource::kafka_datasource(std::map<std::string, std::string> configs, std::vector<std::string> topics, std::vector<int> partitions) {
    DATASOURCE_ID = "librdkafka-";
    DATASOURCE_ID.append(RdKafka::version_str());

    // Construct the RdKafka::Conf object
    kafka_conf_ = std::unique_ptr<RdKafka::Conf>(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    configure_datasource(configs, topics, partitions);
  }

  std::string kafka_datasource::libcudf_datasource_identifier() {
    return DATASOURCE_ID;
  }

  bool kafka_datasource::configure_datasource(std::map<std::string, std::string> configs,
                                              std::vector<std::string> topics,
                                              std::vector<int> partitions) {

    //Set Kafka global configurations
    for (auto const& x : configs) {
      conf_res_ = kafka_conf_->set(x.first, x.second, errstr_);
      if (conf_res_ != RdKafka::Conf::ConfResult::CONF_OK) {
        if (conf_res_ == RdKafka::Conf::ConfResult::CONF_INVALID) {
          //TODO
          printf("Invalid configuration supplied ... what to do here?\n");
        } else if (conf_res_ == RdKafka::Conf::ConfResult::CONF_UNKNOWN) {
          //TODO
          printf("Invalid configuration property supplied ... what to do here? Likely just ignore???\n");
        }
      }
    }

    // Create Toppar instances
    for (const auto& x : topics) {
      for (const auto& y : partitions) {
        partitions_.push_back(RdKafka::TopicPartition::create(x, y));
      }
    }

    // Kafka 0.9 > requires at least a group.id in the configuration so lets
    // make sure that is present.
    conf_res_ = kafka_conf_->get("group.id", conf_val);

    // Create the Rebalance callback so Partition Offsets can be assigned.
    ExampleRebalanceCb ex_rebalance_cb;
    kafka_conf_->set("rebalance_cb", &ex_rebalance_cb, errstr_);

    consumer_ = std::unique_ptr<RdKafka::KafkaConsumer>(RdKafka::KafkaConsumer::create(kafka_conf_.get(), errstr_));
    consumer_.get()->assign(partitions_);

    producer_ = std::unique_ptr<RdKafka::Producer>(RdKafka::Producer::create(kafka_conf_.get(), errstr_));

    return true;
  }

  void kafka_datasource::print_consumer_metadata() {

    printf("\n====== START - LIBRDKAFKA CONSUMER METADATA ======\n");

    RdKafka::Topic *topic = NULL;
    class RdKafka::Metadata *metadata;

    /* Fetch metadata */
    err_ = consumer_->metadata(1, topic,
                              &metadata, default_timeout_);
    if (err_ != RdKafka::ERR_NO_ERROR) {
      printf("Failed to acquire metadata: '%s', returning.\n", RdKafka::err2str(err_).c_str());
      return;
    }

    printf("Metadata for topic(s) (for broker '%d:%s')\n", metadata->orig_broker_id(), metadata->orig_broker_name().c_str());

    /* Iterate brokers */
    printf("'%lu' broker(s)\n", metadata->brokers()->size());
    RdKafka::Metadata::BrokerMetadataIterator ib;
    for (ib = metadata->brokers()->begin(); ib != metadata->brokers()->end(); ++ib) {
      printf("\tBroker ID:'%d' at '%s:%d'\n", (*ib)->id(), (*ib)->host().c_str(), (*ib)->port());
    }

    /* Iterate topics */
    printf("'%lu' topic(s)", metadata->topics()->size());
    RdKafka::Metadata::TopicMetadataIterator it;
    for (it = metadata->topics()->begin(); it != metadata->topics()->end(); ++it) {
      printf("\n\tTopic '%s' has '%lu' partitions ->\n", (*it)->topic().c_str(), (*it)->partitions()->size());

      if ((*it)->err() != RdKafka::ERR_NO_ERROR) {
        printf("'%s'\n", RdKafka::err2str((*it)->err()).c_str());
        if ((*it)->err() == RdKafka::ERR_LEADER_NOT_AVAILABLE) {
          printf("Leader not available, try again.\n");
        }
      }

      /* Iterate Topic's partitions */
      RdKafka::TopicMetadata::PartitionMetadataIterator ip;
      for (ip = (*it)->partitions()->begin(); ip != (*it)->partitions()->end(); ++ip) {
        printf("\t\tPartition '%d', leader: '%d', replicas: ", (*ip)->id(), (*ip)->leader());

        /* Iterate partition's replicas */
        RdKafka::PartitionMetadata::ReplicasIterator ir;
        for (ir = (*ip)->replicas()->begin(); ir != (*ip)->replicas()->end(); ++ir) {
          std::cout << (ir == (*ip)->replicas()->begin() ? "":",") << *ir;
        }

        /* Iterate partition's ISRs */
        printf(" isrs: ");
        RdKafka::PartitionMetadata::ISRSIterator iis;
        for (iis = (*ip)->isrs()->begin(); iis != (*ip)->isrs()->end() ; ++iis) {
          std::cout << (iis == (*ip)->isrs()->begin() ? "":",") << *iis;
        }

        if ((*ip)->err() != RdKafka::ERR_NO_ERROR) {
          std::cout << ", " << RdKafka::err2str((*ip)->err()) << std::endl;
        } else {
          std::cout << std::endl;
        }
      }
    }

     printf("\n====== END - LIBRDKAFKA CONSUMER METADATA ======\n");
  }

  void kafka_datasource::dump_configs() {
    printf("\n====== START - LIBRDKAFKA GLOBAL CONFIGS ======\n");

    std::list<std::string> *dump = kafka_conf_->dump();
    for (std::list<std::string>::iterator it = dump->begin(); it != dump->end(); ) {
      printf("'%s' = ", (*it).c_str());
      it++;
      printf("'%s'\n", (*it).c_str());
      it++;
    }
  
    printf("\n====== END - LIBRDKAFKA GLOBAL CONFIGS ======\n");
  }

  std::map<int, int64_t> kafka_datasource::get_committed_offset(std::string topic, std::vector<int> partitions) {
    std::vector<RdKafka::TopicPartition*> toppar_list;
    std::map<int, int64_t> offsets;

    std::vector<int>::iterator it = partitions.begin();
    while (it != partitions.end()) {
      toppar_list.push_back(find_toppar(topic, *it));
      it++;
    }

    // Query Kafka to populate the TopicPartitions with the desired offsets
    err_ = consumer_->committed(toppar_list, default_timeout_);

    std::vector<RdKafka::TopicPartition*>::iterator top_it = toppar_list.begin();
    while (top_it != toppar_list.end()) {
      offsets.insert({(*top_it)->partition(), (*top_it)->offset()});
      top_it++;
    }

    return offsets;
  }

  std::string kafka_datasource::consume_range(int64_t start_offset,
                                              int64_t end_offset,
                                              int batch_timeout,
                                              std::string delimiter) {
    std::string json_str;
    int64_t messages_read = 0;
    int64_t batch_size = end_offset - start_offset;
    int64_t end = now() + batch_timeout;
    int remaining_timeout = batch_timeout;
    RdKafka::Message *msg;

    printf("Start Offset: '%lu' End Offset: '%lu' Batch Size: '%lu'\n", start_offset, end_offset, batch_size);

    while (messages_read < batch_size) {
      msg = consumer_->consume(remaining_timeout);

      if (msg->err() == RdKafka::ErrorCode::ERR_NO_ERROR) {
        json_str.append(static_cast<char *>(msg->payload()));
        json_str.append("\n");
        messages_read++;
      } else {
        handle_error(msg);
        break;
      }

      remaining_timeout = end - now();
      if (remaining_timeout < 0) {
        break;
      }
    }

    delete msg;

    return json_str;
  }

  bool kafka_datasource::produce_message(std::string topic, std::string message_val, std::string message_key) {
    err_ = producer_->produce(topic,
                       RdKafka::Topic::PARTITION_UA,
                       RdKafka::Producer::RK_MSG_COPY,
                       const_cast<char *>(message_val.c_str()),
                       message_val.size(),
                       const_cast<char *>(message_key.c_str()),
                       message_key.size(),
                       0,
                       NULL,
                       NULL);
    if (err_ != RdKafka::ERR_NO_ERROR) {
      printf("Failed to produce to topic '%s' : '%s'\n", topic.c_str(), RdKafka::err2str(err_).c_str());
      return false;
    } else {
      return true;
    }
  }

  std::map<std::string, int64_t> kafka_datasource::get_watermark_offset(std::string topic, int32_t partition) {
    int64_t low;
    int64_t high;
    std::vector<RdKafka::TopicPartition *> topic_parts;
    std::map<std::string, int64_t> results;

    err_ = consumer_->assignment(topic_parts);
    if (err_ != RdKafka::ErrorCode::ERR_NO_ERROR) {
      printf("Error: '%s'\n", err2str(err_).c_str());
    }
    err_ = consumer_->get_watermark_offsets(topic_parts[0]->topic().c_str(), topic_parts[0]->partition(), &low, &high);

    if (err_ != RdKafka::ErrorCode::ERR_NO_ERROR) {
      printf("Error: '%s'\n", err2str(err_).c_str());
    } else {
      results.insert(std::pair<std::string, int64_t>("low", low));
      results.insert(std::pair<std::string, int64_t>("high", high));
    }

    return results;
  }

  bool kafka_datasource::commit_offset(std::string topic, int partition, int64_t offset) {
    RdKafka::TopicPartition* toppar = find_toppar(topic, partition);
    if (toppar != NULL) {
      toppar->set_offset(offset);
      err_ = consumer_->commitSync(partitions_);
      return true;
    } else {
      return false;
    }
  }

}  // namespace external
}  // namespace io
}  // namespace cudf
