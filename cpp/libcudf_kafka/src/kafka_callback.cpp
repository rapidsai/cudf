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

namespace cudf {
namespace io {
namespace external {
namespace kafka {

PythonOAuthRefreshCb::PythonOAuthRefreshCb(void* callback) : callback(callback){};

void PythonOAuthRefreshCb::oauthbearer_token_refresh_cb(RdKafka::Handle* handle,
                                                        const std::string& oauthbearer_config)
{
  printf("oauthbearer_token_refresh_cb... I want this called so bad!!!\n");

  // Since I need to get the results of the invoked Python function (PyObject) here
  // I don't see how this avoids importing Python dependencies?
  PyObject result = callback();

  // Need to get 3 dict elements and set them here ....
}

}  // namespace kafka
}  // namespace external
}  // namespace io
}  // namespace cudf
