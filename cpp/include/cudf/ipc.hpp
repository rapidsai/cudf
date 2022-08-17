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

#include <cudf/interop.hpp>           // column_metadata
#include <cudf/table/table_view.hpp>  // table_view
#include <memory>                     // std::shared_ptr
#include <string>
#include <utility>  // std::pair
#include <vector>

namespace arrow {
class Buffer;
}  // namespace arrow

namespace cudf {
namespace ipc {
class ipc_imported_ptr;
}

class ipc_imported_column {
 public:
  std::string name;

 private:
  struct impl;
  std::unique_ptr<impl> _pimpl;

 public:
  ipc_imported_column(ipc_imported_column const& that) = delete;
  ipc_imported_column(std::string n, ipc::ipc_imported_ptr&& d, ipc::ipc_imported_ptr&& m);
  ipc_imported_column(std::string n, ipc::ipc_imported_ptr&& d);
  ~ipc_imported_column();
};

std::shared_ptr<arrow::Buffer> export_ipc(table_view input,
                                          std::vector<column_metadata> const& metadata);

std::pair<table_view, std::vector<std::shared_ptr<ipc_imported_column>>> import_ipc(
  std::shared_ptr<arrow::Buffer> ipc_handles);
}  // namespace cudf
