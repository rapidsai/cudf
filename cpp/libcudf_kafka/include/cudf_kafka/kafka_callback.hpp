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
#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <librdkafka/rdkafkacpp.h>
#include <cudf/io/datasource.hpp>
#include <map>
#include <memory>
#include <string>

namespace cudf {
namespace io {
namespace external {
namespace kafka {

/**
 * @brief Callback to retrieve OAuth token from external source. Invoked when
 * token refresh is required.
 */
class OAuthRefreshCb : public RdKafka::OAuthBearerTokenRefreshCb {
 public:
  OAuthRefreshCb(PyObject* callback, PyObject* args);

  void oauthbearer_token_refresh_cb(RdKafka::Handle* handle, const std::string& oauthbearer_config);

 private:
  PyObject* callback;
  PyObject* args;
};

}  // namespace kafka
}  // namespace external
}  // namespace io
}  // namespace cudf
