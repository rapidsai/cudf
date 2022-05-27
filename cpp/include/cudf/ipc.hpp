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
