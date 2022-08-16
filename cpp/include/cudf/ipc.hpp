#include <cudf/interop.hpp>
#include <cudf/table/table_view.hpp>
#include <memory>
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

std::pair<table_view, std::unique_ptr<std::vector<std::shared_ptr<arrow::cuda::CudaBuffer>>>>
import_ipc(std::shared_ptr<arrow::cuda::CudaContext> ctx, std::vector<char> const& ipc_handles);
}  // namespace cudf
