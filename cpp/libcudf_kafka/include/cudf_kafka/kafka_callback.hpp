/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/io/datasource.hpp>

#include <librdkafka/rdkafkacpp.h>

#include <map>
#include <memory>
#include <string>

namespace cudf {
namespace io {
namespace external {
namespace kafka {

/**
 * @brief Python Callback function wrapper type used for Kafka OAuth events
 *
 * The KafkaConsumer calls the `kafka_oauth_callback_wrapper_type` when the existing
 * oauth token is considered expired by the KafkaConsumer. Typically that
 * means this will be invoked a single time when the KafkaConsumer is created
 * to get the initial token and then intermediately as the token becomes
 * expired.
 *
 * The callback function signature is:
 *     `std::map<std::string, std::string> kafka_oauth_callback_wrapper_type(void*)`
 *
 * The callback function returns a std::map<std::string, std::string>,
 * where the std::map consists of the Oauth token and its
 * linux epoch expiration time. Generally the token and expiration
 * time is retrieved from an external service by the callback.
 * Ex: [token, token_expiration_in_epoch]
 */
using kafka_oauth_callback_wrapper_type = std::map<std::string, std::string> (*)(void*);
using python_callable_type              = void*;

/**
 * @brief Callback to retrieve OAuth token from external source. Invoked when
 * token refresh is required.
 */
class python_oauth_refresh_callback : public RdKafka::OAuthBearerTokenRefreshCb {
 public:
  /**
   * @brief Construct a new python oauth refresh callback object
   *
   * @param callback_wrapper Cython wrapper that will
   *                 be used to invoke the `python_callable`. This wrapper serves the purpose
   *                 of preventing us from having to link against the Python development library
   *                 in libcudf_kafka.
   * @param python_callable pointer to a Python `functools.partial` object
   */
  python_oauth_refresh_callback(kafka_oauth_callback_wrapper_type callback_wrapper,
                                python_callable_type python_callable);

  /**
   * @brief Invoke the Python callback function to get the OAuth token and its expiration time
   *
   * @param handle
   * @param oauthbearer_config pointer to the OAuthBearerConfig object
   */
  void oauthbearer_token_refresh_cb(RdKafka::Handle* handle, std::string const& oauthbearer_config);

 private:
  kafka_oauth_callback_wrapper_type callback_wrapper_;
  python_callable_type python_callable_;
};

}  // namespace kafka
}  // namespace external
}  // namespace io
}  // namespace cudf
