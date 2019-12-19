/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "datasource.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <thread>

#include <cudf/cudf.h>
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace io {

class kafka_io_source : public datasource {
 public:
  explicit kafka_io_source(RdKafka::Conf *kafka_configs,
                           std::vector<std::string> kafka_topics,
                           int64_t start_offset,
                           int16_t batch_size)
      : kafka_conf_(kafka_configs), topics_(kafka_topics), start_offset_(start_offset), batch_size_(batch_size) {
    // Kafka 0.9 > requires at least a group.id in the configuration so lets
    // make sure that is present.
    conf_res = kafka_conf_->get("group.id", conf_val);
    CUDF_EXPECTS(
        (conf_res == RdKafka::Conf::ConfResult::CONF_OK && !conf_val.empty()),
        "Kafka requires 'group.id' configuration value be present. Please "
        "ensure Kafka configuration contains 'group.id'");

    // Create the Rebalance callback so Partition Offsets can be assigned.
    KafkaRebalanceCB rebalance_cb(start_offset_);
    kafka_conf_->set("rebalance_cb", &rebalance_cb, errstr_);

    consumer_ = RdKafka::KafkaConsumer::create(kafka_conf_, errstr_);
    CUDF_EXPECTS(consumer_, "Failed to create Kafka consumer");

    err = consumer_->subscribe(topics_);
    CUDF_EXPECTS(err == RdKafka::ErrorCode::ERR_NO_ERROR,
                 "Failed to subscribe to Kafka Topics");

    // The csv_reader implementation will call 'empty()' to determine how maby
    // bytes are available. With files this works, with Kafka we don't yet have
    // the messages at this point so we need to get those messages now.
    consume_messages();
  }

  virtual ~kafka_io_source() {
    delete kafka_conf_;
    delete consumer_;
  }

  const std::shared_ptr<arrow::Buffer> get_buffer(size_t offset,
                                                  size_t size) override {
    return arrow::Buffer::Wrap(buffer_.c_str(), buffer_.size());
  }

  size_t size() const override { return buffer_.size(); }

 private:
  class KafkaRebalanceCB : public RdKafka::RebalanceCb {
   public:

    int64_t start_offset_;

    KafkaRebalanceCB(int64_t start_offset) : start_offset_(start_offset) {}

    void rebalance_cb(RdKafka::KafkaConsumer *consumer, RdKafka::ErrorCode err,
                      std::vector<RdKafka::TopicPartition *> &partitions) {
      if (err == RdKafka::ERR__ASSIGN_PARTITIONS) {
        // NOTICE: We currently purposely only support a single partition. Enhancement PR to be opened later.
        partitions.at(0)->set_offset(start_offset_);
        err = consumer->assign(partitions);
        CUDF_EXPECTS(err == RdKafka::ErrorCode::ERR_NO_ERROR,
                    "Error occured while reassigning the topic partition offset");
      } else {
        consumer->unassign();
      }
    }
  };

  void consume_messages() {
    // Kafka messages are already stored in a queue outside of libcudf. Here the
    // messages will be transferred from the external queue directly to the
    // arrow::Buffer.
    RdKafka::Message *msg;

    for (int i = 0; i <= batch_size_; i++) {
      msg = consumer_->consume(default_timeout_);
      if (msg->err() == RdKafka::ErrorCode::ERR_NO_ERROR) {
        buffer_.append(static_cast<char *>(msg->payload()));
        buffer_.append("\n");
        msg_count_++;
      } else {
        handle_error(msg);

        // handle_error handles specific errors. Any coded logic error case will
        // generate an exception and cease execution. Kafka has hundreds of
        // possible exceptions however. To be safe its best to print the generic
        // error message here and break the consumer loop.
        break;
      }
    }

    delete msg;
  }

  void handle_error(RdKafka::Message *msg) {
    err = msg->err();
    const std::string err_str = msg->errstr();
    std::string error_msg;

    if (msg_count_ == 0 &&
        err == RdKafka::ErrorCode::ERR__PARTITION_EOF) {
      // The topic was empty and had no data in it. Most likely best to error
      // here since the most likely cause of this would be a user entering the
      // wrong topic name.
      error_msg.append("Kafka Topic '");
      error_msg.append(topics_.at(0).c_str());
      error_msg.append("' is empty or does not exist on broker(s)");
      CUDF_FAIL(error_msg);
    } else if (msg_count_ == 0 &&
               err == RdKafka::ErrorCode::ERR__TIMED_OUT) {
      // unable to connect to the specified Kafka Broker(s)
      std::string brokers_val;
      conf_res = kafka_conf_->get("metadata.broker.list", brokers_val);
      if (brokers_val.empty()) {
        // 'bootstrap.servers' is an alias configuration so its valid that
        // either 'metadata.broker.list' or 'bootstrap.servers' is set
        conf_res = kafka_conf_->get("bootstrap.servers", brokers_val);
      }

      if (conf_res == RdKafka::Conf::ConfResult::CONF_OK) {
        error_msg.append("Connection attempt to Kafka broker(s) '");
        error_msg.append(brokers_val);
        error_msg.append("' timed out.");
        CUDF_FAIL(error_msg);
      } else {
        CUDF_FAIL(
            "No Kafka broker(s) were specified for connection. Connection "
            "Failed.");
      }
    } else if (err == RdKafka::ErrorCode::ERR__PARTITION_EOF) {
      // Kafka treats PARTITION_EOF as an "error". In our Rapids use case it is
      // not however and just means all messages have been read.
      // Just print imformative message and break consume loop.
      printf("%ld messages read from Kafka\n", msg_count_);
    }
  }

 private:
  RdKafka::Conf *kafka_conf_;
  RdKafka::KafkaConsumer *consumer_;
  RdKafka::ErrorCode err;

  std::vector<std::string> topics_;
  std::string errstr_;
  RdKafka::Conf::ConfResult conf_res;
  std::string conf_val;
  int64_t start_offset_ = -1;
  int16_t batch_size_ = 10000;  // 10K is the Kafka standard. Max is 999,999
  int32_t default_timeout_ = 10000;  // 1 second
  int64_t msg_count_ = 0;  // Running tally of the messages consumed. Useful for retry logic.

  std::string buffer_;
};  // namespace io

/**
 * @brief Implementation class for reading from an Apache Arrow file. The file
 * could be a memory-mapped file or other implementation supported by Arrow.
 **/
class arrow_io_source : public datasource {
 public:
  explicit arrow_io_source(std::shared_ptr<arrow::io::RandomAccessFile> file)
      : arrow_file(file) {}

  const std::shared_ptr<arrow::Buffer> get_buffer(size_t position,
                                                  size_t length) override {
    std::shared_ptr<arrow::Buffer> out;
    CUDF_EXPECTS(arrow_file->ReadAt(position, length, &out).ok(),
                 "Cannot read file data");
    return out;
  }

  size_t size() const override {
    int64_t size;
    CUDF_EXPECTS(arrow_file->GetSize(&size).ok(), "Cannot get file size");
    return size;
  }

 private:
  std::shared_ptr<arrow::io::RandomAccessFile> arrow_file;
};

/**
 * @brief Implementation class for reading from a file or memory source using
 * memory mapped access.
 *
 * Unlike Arrow's memory mapped IO class, this implementation allows memory
 * mapping a subset of the file where the starting offset may not be zero.
 **/
class memory_mapped_source : public datasource {
  struct file_wrapper {
    const int fd = -1;
    explicit file_wrapper(const char *filepath)
        : fd(open(filepath, O_RDONLY)) {}
    ~file_wrapper() { close(fd); }
  };

 public:
  explicit memory_mapped_source(const char *filepath, size_t offset,
                                size_t size) {
    auto file = file_wrapper(filepath);
    CUDF_EXPECTS(file.fd != -1, "Cannot open file");

    struct stat st {};
    CUDF_EXPECTS(fstat(file.fd, &st) != -1, "Cannot query file size");
    file_size_ = static_cast<size_t>(st.st_size);

    if (file_size_ != 0) {
      map(file.fd, offset, size);
    }
  }

  virtual ~memory_mapped_source() {
    if (map_addr_ != nullptr) {
      munmap(map_addr_, map_size_);
    }
  }

  const std::shared_ptr<arrow::Buffer> get_buffer(size_t offset,
                                                  size_t size) override {
    // Clamp length to available data in the mapped region
    CUDF_EXPECTS(offset >= map_offset_, "Requested offset is outside mapping");
    size = std::min(size, map_size_ - (offset - map_offset_));

    return arrow::Buffer::Wrap(
        static_cast<uint8_t *>(map_addr_) + (offset - map_offset_), size);
  }

  size_t size() const override { return file_size_; }

 private:
  void map(int fd, size_t offset, size_t size) {
    // Offset for `mmap()` must be page aligned
    const auto map_offset = offset & ~(sysconf(_SC_PAGESIZE) - 1);
    CUDF_EXPECTS(offset < file_size_, "Offset is past end of file");

    // Clamp length to available data in the file
    if (size == 0) {
      size = file_size_ - offset;
    } else {
      if ((offset + size) > file_size_) {
        size = file_size_ - offset;
      }
    }

    // Size for `mmap()` needs to include the page padding
    const auto map_size = size + (offset - map_offset);

    // Check if accessing a region within already mapped area
    map_addr_ = mmap(NULL, map_size, PROT_READ, MAP_PRIVATE, fd, map_offset);
    CUDF_EXPECTS(map_addr_ != MAP_FAILED, "Cannot create memory mapping");
    map_offset_ = map_offset;
    map_size_ = map_size;
  }

 private:
  size_t file_size_ = 0;
  void *map_addr_ = nullptr;
  size_t map_size_ = 0;
  size_t map_offset_ = 0;
};

std::unique_ptr<datasource> datasource::create(
    RdKafka::Conf *global_configs, std::vector<std::string> kafka_topics, 
    int64_t start_offset, int16_t batch_size) {
  return std::make_unique<kafka_io_source>(global_configs, kafka_topics, start_offset, batch_size);
}

std::unique_ptr<datasource> datasource::create(const std::string filepath,
                                               size_t offset, size_t size) {
  // Use our own memory mapping implementation for direct file reads
  return std::make_unique<memory_mapped_source>(filepath.c_str(), offset, size);
}

std::unique_ptr<datasource> datasource::create(const char *data,
                                               size_t length) {
  // Use Arrow IO buffer class for zero-copy reads of host memory
  return std::make_unique<arrow_io_source>(
      std::make_shared<arrow::io::BufferReader>(
          reinterpret_cast<const uint8_t *>(data), length));
}

std::unique_ptr<datasource> datasource::create(
    std::shared_ptr<arrow::io::RandomAccessFile> file) {
  // Support derived classes of the top-level Arrow IO interface
  return std::make_unique<arrow_io_source>(file);
}

}  // namespace io
}  // namespace cudf
