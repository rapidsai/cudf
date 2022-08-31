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
struct imported_column::impl {
  ipc::imported_ptr data;
  ipc::imported_ptr mask;
  std::vector<std::shared_ptr<imported_column>> children;

  impl(ipc::imported_ptr&& d, ipc::imported_ptr&& m)
    : data{std::forward<ipc::imported_ptr>(d)}, mask{std::forward<ipc::imported_ptr>(m)}
  {
  }

  impl(ipc::imported_ptr&& d,
       ipc::imported_ptr&& m,
       std::vector<std::shared_ptr<imported_column>>&& children)
    : data{std::forward<ipc::imported_ptr>(d)},
      mask{std::forward<ipc::imported_ptr>(m)},
      children{std::forward<std::vector<std::shared_ptr<imported_column>>>(children)}
  {
  }
};

imported_column::imported_column(std::string n, ipc::imported_ptr&& d, ipc::imported_ptr&& m)
  : name{std::move(n)},
    _pimpl{std::make_unique<impl>(std::forward<ipc::imported_ptr>(d),
                                  std::forward<ipc::imported_ptr>(m))}
{
}

imported_column::imported_column(std::string n,
                                 ipc::imported_ptr&& d,
                                 ipc::imported_ptr&& m,
                                 std::vector<std::shared_ptr<imported_column>>&& children)
  : name{std::move(n)},
    _pimpl{
      std::make_unique<impl>(std::forward<ipc::imported_ptr>(d),
                             std::forward<ipc::imported_ptr>(m),
                             std::forward<std::vector<std::shared_ptr<imported_column>>>(children))}
{
}

imported_column::imported_column(std::string n, ipc::imported_ptr&& d)
  : name{std::move(n)},
    _pimpl{std::make_unique<impl>(std::forward<ipc::imported_ptr>(d), ipc::imported_ptr{})}
{
}

imported_column::~imported_column() = default;

namespace ipc {
void exported_column::serialize(std::string* p_bytes) const
{
  std::string& bytes = *p_bytes;
  int64_t n_children = children.size();
  size_t orig_size   = p_bytes->size();

  // n_children
  bytes.resize(orig_size + sizeof(n_children));
  auto ptr = bytes.data() + orig_size;
  std::memcpy(ptr, &n_children, sizeof(n_children));
  // children
  if (!children.empty()) {
    for (auto const& child : children) {
      child.serialize(p_bytes);
    }
  }

  orig_size = p_bytes->size();
  // has_nulls
  auto hn = has_nulls();
  bytes.resize(orig_size + sizeof(hn));
  ptr = bytes.data() + orig_size;
  std::memcpy(ptr, &hn, sizeof(hn));
  // size
  orig_size = bytes.size();
  bytes.resize(orig_size + sizeof(this->size));
  ptr = bytes.data() + orig_size;
  std::memcpy(ptr, &this->size, sizeof(this->size));
  // data
  data.serialize(p_bytes);
  // mask
  if (has_nulls()) { mask.serialize(p_bytes); }
}

uint8_t const* exported_column::from_buffer(uint8_t const* ptr, exported_column* out)
{
  exported_column& column = *out;
  // n_children
  int64_t n_children{0};
  std::memcpy(&n_children, ptr, sizeof(n_children));
  ptr += sizeof(n_children);
  column.children.resize(n_children);
  // children
  for (int64_t i = 0; i < n_children; ++i) {
    exported_column& child = column.children[i];
    ptr                    = exported_column::from_buffer(ptr, &child);
  }
  // has_nulls
  bool hn;
  std::memcpy(&hn, ptr, sizeof(hn));
  ptr += sizeof(hn);
  // size
  size_type size;
  std::memcpy(&size, ptr, sizeof(size));
  ptr += sizeof(size);
  column.size = size;
  // data
  ptr = exported_ptr::from_buffer(ptr, &column.data);
  // mask
  if (hn) { ptr = exported_ptr::from_buffer(ptr, &column.mask); }
  return ptr;
}
}  // namespace ipc
}  // namespace cudf
