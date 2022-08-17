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

#include "ipc.hpp"
#include <cudf/ipc.hpp>

namespace cudf {
struct ipc_imported_column::impl {
  ipc::ipc_imported_ptr data;
  ipc::ipc_imported_ptr mask;

  impl(ipc::ipc_imported_ptr&& d, ipc::ipc_imported_ptr&& m)
    : data{std::forward<ipc::ipc_imported_ptr>(d)}, mask{std::forward<ipc::ipc_imported_ptr>(m)}
  {
  }
};

ipc_imported_column::ipc_imported_column(std::string n, ipc::ipc_imported_ptr&& d, ipc::ipc_imported_ptr&& m)
  : name{std::move(n)},
    _pimpl{
      std::make_unique<impl>(std::forward<ipc::ipc_imported_ptr>(d), std::forward<ipc::ipc_imported_ptr>(m))}
{
}

ipc_imported_column::ipc_imported_column(std::string n, ipc::ipc_imported_ptr&& d)
  : name{std::move(n)},
    _pimpl{std::make_unique<impl>(std::forward<ipc::ipc_imported_ptr>(d), ipc::ipc_imported_ptr{})}
{
}

ipc_imported_column::~ipc_imported_column() = default;
}  // namespace cudf
