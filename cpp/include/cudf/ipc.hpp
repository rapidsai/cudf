#include <arrow/api.h>
#include <arrow/ipc/api.h>
#include <arrow/gpu/cuda_api.h>

class CudaMessageReader : arrow::ipc::MessageReader {
    public:

        CudaMessageReader(arrow::cuda::CudaBufferReader* stream, arrow::io::BufferReader* schema);

        static std::unique_ptr<arrow::ipc::MessageReader> Open(arrow::cuda::CudaBufferReader* stream, arrow::io::BufferReader* schema);

        arrow::Status ReadNextMessage(std::unique_ptr<arrow::ipc::Message>* message);

    private:
        arrow::cuda::CudaBufferReader* stream_;
        arrow::io::BufferReader* host_schema_reader_ = nullptr;
        std::shared_ptr<arrow::cuda::CudaBufferReader> owned_stream_;
};
