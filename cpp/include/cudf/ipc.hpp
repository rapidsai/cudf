#include <cudf/interop.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <memory>
#include <string>
#include <vector>

namespace arrow {
class Buffer;
namespace cuda {
class CudaContext;
class CudaBuffer;
}  // namespace cuda
}  // namespace arrow

namespace cudf {
std::shared_ptr<arrow::Buffer> export_ipc(std::shared_ptr<arrow::cuda::CudaContext> ctx,
                                          table_view input,
                                          std::vector<column_metadata> const& metadata);

std::pair<std::unique_ptr<table>, std::vector<std::string>> import_ipc(
  std::shared_ptr<arrow::cuda::CudaContext> ctx, std::shared_ptr<arrow::Buffer> ipc_handles);
}  // namespace cudf
