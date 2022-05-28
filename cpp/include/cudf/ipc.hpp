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

#pragma once

#include <arrow/api.h>
#include <arrow/gpu/cuda_api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>

/**
 * @brief Reads Message objects from cuda buffer source
 *
 */
class CudaMessageReader : arrow::ipc::MessageReader {
 public:
  /**
   * @brief Construct a new Cuda Message Reader object from a cuda buffer stream
   *
   * @param stream The cuda buffer reader stream
   * @param schema The schema of the stream
   */
  CudaMessageReader(arrow::cuda::CudaBufferReader* stream, arrow::io::BufferReader* schema);

  /**
   * @brief Open stream from source.
   *
   * @param stream The cuda buffer reader stream
   * @param schema The schema of the stream
   * @return arrow::ipc::MessageReader object
   */
  static std::unique_ptr<arrow::ipc::MessageReader> Open(arrow::cuda::CudaBufferReader* stream,
                                                         arrow::io::BufferReader* schema);

  /**
   * @brief Read next Message from the stream.
   *
   * @return arrow::ipc::Message object
   */
  arrow::Result<std::unique_ptr<arrow::ipc::Message>> ReadNextMessage() override;

 private:
  arrow::cuda::CudaBufferReader* stream_;
  arrow::io::BufferReader* host_schema_reader_ = nullptr;
  std::shared_ptr<arrow::cuda::CudaBufferReader> owned_stream_;
};
