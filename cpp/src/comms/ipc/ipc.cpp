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
    if (message.ok() && *message != nullptr) { return std::move(message); }
  }
  std::unique_ptr<arrow::ipc::Message> message;
  auto status = arrow::cuda::ReadMessage(stream_, arrow::default_memory_pool(), &message);
  if (!status.ok()) { return status; }
  return std::move(message);
}

std::unique_ptr<arrow::ipc::MessageReader> CudaMessageReader::Open(
  arrow::cuda::CudaBufferReader* stream, arrow::io::BufferReader* schema)
{
  return std::unique_ptr<arrow::ipc::MessageReader>(new CudaMessageReader(stream, schema));
}
