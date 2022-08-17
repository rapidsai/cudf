#include <cudf/interop.hpp>           // column_metadata
#include <cudf/table/table_view.hpp>  // table_view
#include <memory>                     // std::unique_ptr
#include <string>
#include <utility>  // std::pair
#include <vector>

namespace arrow {
class Buffer;
}  // namespace arrow

namespace cudf {
std::shared_ptr<arrow::Buffer> export_ipc(table_view input,
                                          std::vector<column_metadata> const& metadata);

std::pair<std::unique_ptr<table>, std::vector<std::string>> import_ipc(
  std::shared_ptr<arrow::Buffer> ipc_handles);
}  // namespace cudf
