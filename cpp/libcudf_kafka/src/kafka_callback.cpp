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

OAuthRefreshCb::OAuthRefreshCb(PyObject* callback, PyObject* args)
  : callback(callback), args(args){};

void OAuthRefreshCb::oauthbearer_token_refresh_cb(RdKafka::Handle* handle,
                                                  const std::string& oauthbearer_config)
{
  CUDF_EXPECTS(PyCallable_Check(callback), "A Python callable is required");

  // Make sure that we own the GIL
  PyGILState_STATE state = PyGILState_Ensure();
  PyObject* result       = PyObject_CallObject(callback, args);
  Py_XINCREF(result);

  // Set the token in the Kafka context
  if (result) {
    CUDF_EXPECTS(PyDict_Check(result),
                 "cudf_kafka requires a Dictionary response from the Python OAuthRefreshCb with "
                 "dictionary keys (token, token_lifetime_ms, principal, extensions)");

    // Ensure that expected keys are present from the Python callback response.
    std::string token = PyUnicode_AsUTF8(PyDict_GetItemString(result, "token"));
    int64_t token_lifetime_ms =
      PyLong_AsLongLong(PyDict_GetItemString(result, "token_lifetime_ms"));
    std::string principal = PyUnicode_AsUTF8(PyDict_GetItemString(result, "principal"));
    std::list<std::string> extensions;
    std::string errstr;

    handle->oauthbearer_set_token(token, token_lifetime_ms, principal, extensions, errstr);
  } else {
    handle->oauthbearer_set_token_failure("");
  }

  Py_XDECREF(result);
  PyGILState_Release(state);
}

}  // namespace kafka
}  // namespace external
}  // namespace io
}  // namespace cudf
