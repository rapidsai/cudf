/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <arrow/result.h>
#include <cudf/ipc.hpp>

CudaMessageReader::CudaMessageReader(arrow::cuda::CudaBufferReader* stream,
                                     arrow::io::BufferReader* schema)
  : stream_(stream), host_schema_reader_(schema){};

arrow::Result<std::unique_ptr<arrow::ipc::Message>> CudaMessageReader::ReadNextMessage()
{
  if (host_schema_reader_ != nullptr) {
    auto message        = arrow::ipc::ReadMessage(host_schema_reader_);
    host_schema_reader_ = nullptr;
    if (message.ok() && *message != nullptr) { return message; }
  }
  return arrow::ipc::ReadMessage(stream_, arrow::default_memory_pool());
}

std::unique_ptr<arrow::ipc::MessageReader> CudaMessageReader::Open(
  arrow::cuda::CudaBufferReader* stream, arrow::io::BufferReader* schema)
{
  return std::unique_ptr<arrow::ipc::MessageReader>(new CudaMessageReader(stream, schema));
}
