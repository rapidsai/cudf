/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#pragma once

#include <string.h>
#include "orc.h"

namespace cudf {
namespace io {
namespace orc {

struct ProtobufWriter::ProtobufFieldWriter {
  int struct_size;
  ProtobufWriter *p;

  ProtobufFieldWriter(ProtobufWriter *pbw) : struct_size(0), p(pbw) {}

  template <typename T>
  void field_uint(int id, const T &v)
  {
    struct_size += p->put_uint(id * 8 + PB_TYPE_VARINT);
    struct_size += p->put_uint(static_cast<uint64_t>(v));
  }

  template <typename T>
  void field_packed_uint(int id, const std::vector<T> &v)
  {
    size_t cnt = v.size(), sz = 0, lpos;
    struct_size += p->put_uint(id * 8 + PB_TYPE_FIXEDLEN);
    lpos = p->m_buf->size();
    p->putb(0);
    for (size_t i = 0; i < cnt; i++) sz += p->put_uint(v[i]);
    struct_size += sz + 1;
    for (; sz > 0x7f; sz >>= 7, struct_size++)
      p->m_buf->insert(p->m_buf->begin() + (lpos++), static_cast<uint8_t>((sz & 0x7f) | 0x80));
    (*(p->m_buf))[lpos] = static_cast<uint8_t>(sz);
  }

  void field_string(int id, const std::string &v)
  {
    size_t len = v.length();
    struct_size += p->put_uint(id * 8 + PB_TYPE_FIXEDLEN);
    struct_size += p->put_uint(len) + len;
    for (size_t i = 0; i < len; i++) p->putb(v[i]);
  }

  template <typename T>
  void field_blob(int id, const std::vector<T> &v)
  {
    size_t len = v.size();
    struct_size += p->put_uint(id * 8 + PB_TYPE_FIXEDLEN);
    struct_size += p->put_uint(len) + len;
    for (size_t i = 0; i < len; i++) p->putb(v[i]);
  }

  template <typename T>
  void field_struct(int id, const T &v)
  {
    size_t sz, lpos;
    struct_size += p->put_uint((id)*8 + PB_TYPE_FIXEDLEN);
    lpos = p->m_buf->size();
    p->putb(0);
    sz = p->write(v);
    struct_size += sz + 1;
    for (; sz > 0x7f; sz >>= 7, struct_size++)
      p->m_buf->insert(p->m_buf->begin() + (lpos++), static_cast<uint8_t>((sz & 0x7f) | 0x80));
    (*(p->m_buf))[lpos] = static_cast<uint8_t>(sz);
  }

  void field_repeated_string(int id, const std::vector<std::string> &v)
  {
    for (const auto &elem : v) field_string(id, elem);
  }

  template <typename T>
  void field_repeated_struct(int id, const std::vector<T> &v)
  {
    for (const auto &elem : v) field_struct(id, elem);
  }

  template <typename T>
  void field_repeated_struct_blob(int id, const std::vector<T> &v)
  {
    for (const auto &elem : v) field_blob(id, elem);
  }

  size_t value(void) { return struct_size; }
};

}  // namespace orc
}  // namespace io
}  // namespace cudf
