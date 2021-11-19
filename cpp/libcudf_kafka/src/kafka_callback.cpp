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

python_oauth_refresh_callback::python_oauth_refresh_callback(kafka_oauth_callback_type cb)
  : oauth_callback_(cb){};

void python_oauth_refresh_callback::oauthbearer_token_refresh_cb(
  RdKafka::Handle* handle, const std::string& oauthbearer_config)
{
  // Make sure that we own the GIL
  PyGILState_STATE state = PyGILState_Ensure();
  PyObject* result       = oauth_callback_();
  Py_XINCREF(result);

  // Set the token in the Kafka context
  if (result) {
    CUDF_EXPECTS(PyTuple_Check(result) && PyTuple_Size(result) == 2,
                 "cudf_kafka requires a Tuple response with 2 entries from the "
                 "PythonOAuthRefreshCb containing [token, token_expiration_ms_in_epoch");

    // Ensure that expected keys are present from the Python callback response.
    std::string token         = PyUnicode_AsUTF8(PyTuple_GetItem(result, 0));
    int64_t token_lifetime_ms = PyLong_AsLongLong(PyTuple_GetItem(result, 1));
    std::list<std::string> extensions;  // currently not supported
    std::string errstr;

    CUDF_EXPECTS(
      RdKafka::ErrorCode::ERR_NO_ERROR ==
        handle->oauthbearer_set_token(token, token_lifetime_ms, "kafka", extensions, errstr),
      "Error occurred while setting the oauthbearer token");
  } else {
    handle->oauthbearer_set_token_failure("Unable to acquire oauth bearer token");
  }

  Py_XDECREF(result);
  PyGILState_Release(state);
}

}  // namespace kafka
}  // namespace external
}  // namespace io
}  // namespace cudf
