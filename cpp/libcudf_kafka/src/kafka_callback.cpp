/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include "cudf_kafka/kafka_callback.hpp"

#include <librdkafka/rdkafkacpp.h>

namespace cudf {
namespace io {
namespace external {
namespace kafka {

python_oauth_refresh_callback::python_oauth_refresh_callback(kafka_oauth_callback_type cb,
                                                             void* python_callable)
  : oauth_callback_(cb), python_callable_(python_callable){};

void python_oauth_refresh_callback::oauthbearer_token_refresh_cb(
  RdKafka::Handle* handle, const std::string& oauthbearer_config)
{
  printf("!!!!Invoking the python_oauth_callback!!!!\n");

  std::map<std::string, std::string> resp = oauth_callback_(python_callable_);

  // Build parameters to pass to librdkafka
  std::string token         = resp["token"];
  int64_t token_lifetime_ms = std::stoll(resp["token_expiration_in_epoch"]);
  std::list<std::string> extensions;  // currently not supported
  std::string errstr;
  CUDF_EXPECTS(
    RdKafka::ErrorCode::ERR_NO_ERROR ==
      handle->oauthbearer_set_token(token, token_lifetime_ms, "kafka", extensions, errstr),
    "Error occurred while setting the oauthbearer token");
}

}  // namespace kafka
}  // namespace external
}  // namespace io
}  // namespace cudf
