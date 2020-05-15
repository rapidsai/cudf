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
#include <map>
#include <memory>
#include <string>

namespace cudf {
namespace io {
namespace external {

/**
 * @brief libcudf external datasource for Apache Kafka
 **/
class kafka_datasource {
 public:
  /**
   * @brief Retrieves the current configurations of the underlying librdkafka
   *client instance
   *
   * @return Map of key/value pairs representing the librdkafka current
   *configurations
   **/
  std::map<std::string, std::string> current_configs()
  {
    std::map<std::string, std::string> configs;
    std::list<std::string> *dump = kafka_conf_->dump();
    std::string key;
    std::string val;
    for (std::list<std::string>::iterator it = dump->begin(); it != dump->end();) {
      key = (*it);
      it++;
      val = (*it);
      it++;
      configs.insert(std::pair<std::string, std::string>{key, val});
    }
    return configs;
  };

  /**
   * @brief Applies the specified configurations to the underlying librdkafka
   *client
   *
   * @param configs Map of key/value pairs that represent the librdkafka
   *configurations to be applied
   *
   * @return True on success or False otherwise
   **/
  virtual bool configure_datasource(std::map<std::string, std::string> configs) = 0;

  /**
   * @brief Base class destructor
   **/
  virtual ~kafka_datasource(){};

 protected:
  /**
   * Convenience method for getting "now()" in Kafka standard format
   **/
  int64_t now()
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((int64_t)tv.tv_sec * 1000) + (tv.tv_usec / 1000);
  }

 protected:
  std::unique_ptr<RdKafka::Conf> kafka_conf_;  // RDKafka configuration object
  RdKafka::Conf::ConfResult conf_res_;         // Result from configuration update operation
  RdKafka::ErrorCode err_;                     // RDKafka ErrorCode from operation
  std::string errstr_;                         // Textual representation of Error
  std::string conf_val;                        // String value of a RDKafka configuration request
  int32_t default_timeout_ = 10000;  // Default timeout for server bound operations - 10 seconds
};

}  // namespace external
}  // namespace io
}  // namespace cudf
