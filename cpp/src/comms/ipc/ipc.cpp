#include <cudf/ipc.hpp>

CudaMessageReader::CudaMessageReader(arrow::cuda::CudaBufferReader* stream,
                                     arrow::io::BufferReader* schema)
  : stream_(stream), host_schema_reader_(schema){};

arrow::Status CudaMessageReader::ReadNextMessage(std::unique_ptr<arrow::ipc::Message>* message)
{
  if (host_schema_reader_ != nullptr) {
    arrow::Status status(arrow::ipc::ReadMessage(host_schema_reader_, message));
    if (status.ok() && *message != nullptr) { return status; }
    host_schema_reader_ = nullptr;
  }
  return arrow::cuda::ReadMessage(stream_, arrow::default_memory_pool(), message);
}

std::unique_ptr<arrow::ipc::MessageReader> CudaMessageReader::Open(
  arrow::cuda::CudaBufferReader* stream, arrow::io::BufferReader* schema)
{
  return std::unique_ptr<arrow::ipc::MessageReader>(new CudaMessageReader(stream, schema));
}
