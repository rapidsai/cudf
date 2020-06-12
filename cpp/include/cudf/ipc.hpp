#include <arrow/api.h>
#include <arrow/gpu/cuda_api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>

class CudaMessageReader : arrow::ipc::MessageReader {
 public:
  CudaMessageReader(arrow::cuda::CudaBufferReader* stream, arrow::io::BufferReader* schema);

  static std::unique_ptr<arrow::ipc::MessageReader> Open(arrow::cuda::CudaBufferReader* stream,
                                                         arrow::io::BufferReader* schema);

  arrow::Result<std::unique_ptr<arrow::ipc::Message>> ReadNextMessage() override;

 private:
  arrow::cuda::CudaBufferReader* stream_;
  arrow::io::BufferReader* host_schema_reader_ = nullptr;
  std::shared_ptr<arrow::cuda::CudaBufferReader> owned_stream_;
};
