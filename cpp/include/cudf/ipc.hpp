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
class ipc_imported_ptr {
  void* base_ptr{nullptr};

 public:
  ipc_imported_ptr() = default;
  explicit ipc_imported_ptr(cudaIpcMemHandle_t handle)
  {
    CUDF_CUDA_TRY(cudaIpcOpenMemHandle(&base_ptr, handle, cudaIpcMemLazyEnablePeerAccess));
  }
  ~ipc_imported_ptr() noexcept(false)
  {
    if (base_ptr) { CUDF_CUDA_TRY(cudaIpcCloseMemHandle(base_ptr)) };
  }

  ipc_imported_ptr(ipc_imported_ptr const& that) = delete;
  ipc_imported_ptr(ipc_imported_ptr&& that) { std::swap(that.base_ptr, this->base_ptr); }

  template <typename T>
  auto get() const
  {
    return reinterpret_cast<T const*>(base_ptr);
  }
  template <typename T>
  auto get()
  {
    return reinterpret_cast<T*>(base_ptr);
  }
};

struct ipc_imported_column {
  std::string name;

  ipc_imported_ptr data;
  ipc_imported_ptr mask;

  ipc_imported_column(ipc_imported_column const& that) = delete;

  ipc_imported_column(std::string n, ipc_imported_ptr&& d)
    : name{std::move(n)}, data{std::forward<ipc_imported_ptr>(d)}
  {
  }

  ipc_imported_column(std::string n, ipc_imported_ptr&& d, ipc_imported_ptr&& m)
    : name{std::move(n)},
      data{std::forward<ipc_imported_ptr>(d)},
      mask{std::forward<ipc_imported_ptr>(m)}
  {
  }
};

std::shared_ptr<arrow::Buffer> export_ipc(table_view input,
                                          std::vector<column_metadata> const& metadata);

std::pair<table_view, std::vector<std::shared_ptr<ipc_imported_column>>> import_ipc(
  std::shared_ptr<arrow::Buffer> ipc_handles);
}  // namespace cudf
