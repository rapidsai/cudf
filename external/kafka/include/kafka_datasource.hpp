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

#include "external_datasource.hpp"
#include <map>
#include <sys/time.h>
#include <librdkafka/rdkafkacpp.h>

namespace cudf {
namespace io {
namespace external {

/**
 * @brief libcudf external datasource for Apache Kafka
 **/
class kafka_datasource : public external_datasource {

 public:

  /**
   * Returns the Kafka datasource identifier for a datsource instance.
   * Example: 'librdkafka-1.3.1'
   **/
  std::string libcudf_datasource_identifier();

  /**
   * Apply user supplied configurations to the current datasource object.
   * 
   * It often makes sense to create a librdkafka instant that has not yet
   * been configured. This method provides a way for those objects to later 
   * be configured or altered from their current configuration.
   **/ 
  bool configure_datasource(std::map<std::string, std::string> configs);

  /**
   * Queries the librdkafka instance and returns its current configurations.
   **/
  std::map<std::string, std::string> current_configs();

  /**
   * @brief Base class destructor
   **/
  virtual ~kafka_datasource(){};


  protected:

    /**
     * Protected since kafka_datasource should never be instantiated directly.
     * 
     * Creates a none configured kafka_datasource instance. Before consuming/publishing
     * 'configure_datasource(...)' should be called.
     **/
    kafka_datasource();

    /**
     * Creates a kafka_datasource instance that is immediately ready for consuming/publishing
     */
    kafka_datasource(std::map<std::string, std::string> configs);

    /**
     * Convenience method for getting "now()" in Kafka standard format
     **/
    int64_t now() {
      struct timeval tv;
      gettimeofday(&tv, NULL);
      return ((int64_t)tv.tv_sec * 1000) + (tv.tv_usec / 1000);
    }

    /**
     * librdkafka errors are contextually and usage sensitive. Often errors are set that can be
     * ignored if certain usage patterns or configurations are set. This method provides a 
     * common place for a subset of those errors to be handled in the context of libcudf usage/
     **/
    void handle_error(RdKafka::Message *msg, int msg_count) {
      err_ = msg->err();
      errstr_ = msg->errstr();
      std::string error_msg;

      if (msg_count == 0 &&
                err_ == RdKafka::ErrorCode::ERR__TIMED_OUT) {
        // unable to connect to the specified Kafka Broker(s)
        std::string brokers_val;
        conf_res_ = kafka_conf_->get("metadata.broker.list", brokers_val);
        if (brokers_val.empty()) {
          // 'bootstrap.servers' is an alias configuration so its valid that
          // either 'metadata.broker.list' or 'bootstrap.servers' is set
          conf_res_ = kafka_conf_->get("bootstrap.servers", brokers_val);
        }

        if (conf_res_ == RdKafka::Conf::ConfResult::CONF_OK) {
          error_msg.append("Connection attempt to Kafka broker(s) '");
          error_msg.append(brokers_val);
          error_msg.append("' timed out.");
        }
      }
    }

  protected:
    std::unique_ptr<RdKafka::Conf> kafka_conf_; // RDKafka configuration object
    RdKafka::Conf::ConfResult conf_res_;        // Result from configuration update operation
    RdKafka::ErrorCode err_;                    // RDKafka ErrorCode from operation
    std::string errstr_;                        // Textual representation of Error
    std::string conf_val;                       // String value of a RDKafka configuration request
    int32_t default_timeout_ = 10000;            // Default timeout for server bound operations - 10 seconds

};

}  // namespace external
}  // namespace io
}  // namespace cudf
