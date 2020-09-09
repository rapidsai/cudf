/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <string.h>
#include "orc.h"

namespace cudf {
namespace io {
namespace orc {

template <int index>
struct FunctionSwitchImpl {
  template <typename... Operator>
  static inline bool run(ProtobufReader *pbr,
                         const uint8_t *end,
                         const int &fld,
                         std::tuple<Operator...> &ops)
  {
    if (fld == std::get<index>(ops).field) {
      return std::get<index>(ops)(pbr, end);
    } else {
      return FunctionSwitchImpl<index - 1>::run(pbr, end, fld, ops);
    }
  }
};

template <>
struct FunctionSwitchImpl<0> {
  template <typename... Operator>
  static inline bool run(ProtobufReader *pbr,
                         const uint8_t *end,
                         const int &fld,
                         std::tuple<Operator...> &ops)
  {
    if (fld == std::get<0>(ops).field) {
      return std::get<0>(ops)(pbr, end);
    } else {
      pbr->skip_struct_field(fld & 7);
      return false;
    }
  }
};

template <typename T>
inline bool ProtobufReader::function_builder_return(T *s, const uint8_t *end)
{
  return m_cur <= end;
}

template <>
inline bool ProtobufReader::function_builder_return<FileFooter>(FileFooter *s, const uint8_t *end)
{
  return (m_cur <= end) && InitSchema(s);
}

template <typename T, typename... Operator>
inline bool ProtobufReader::function_builder(T *s, size_t maxlen, std::tuple<Operator...> &op)
{
  constexpr int index = std::tuple_size<std::tuple<Operator...>>::value - 1;
  const uint8_t *end  = std::min(m_cur + maxlen, m_end);
  while (m_cur < end) {
    int fld            = get_u32();
    bool exit_function = FunctionSwitchImpl<index>::run(this, end, fld, op);
    if (exit_function) { return false; }
  }
  return function_builder_return(s, end);
}

struct ProtobufReader::FieldInt32 {
  int field;
  int32_t &val;

  FieldInt32(int f, int32_t &v) : field((f * 8) + PB_TYPE_VARINT), val(v) {}

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    val = pbr->get_i32();
    return false;
  }
};

struct ProtobufReader::FieldUInt32 {
  int field;
  uint32_t &val;

  FieldUInt32(int f, uint32_t &v) : field((f * 8) + PB_TYPE_VARINT), val(v) {}

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    val = pbr->get_u32();
    return false;
  }
};

struct ProtobufReader::FieldInt64 {
  int field;
  int64_t &val;

  FieldInt64(int f, int64_t &v) : field((f * 8) + PB_TYPE_VARINT), val(v) {}

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    val = pbr->get_i64();
    return false;
  }
};

struct ProtobufReader::FieldUInt64 {
  int field;
  uint64_t &val;

  FieldUInt64(int f, uint64_t &v) : field((f * 8) + PB_TYPE_VARINT), val(v) {}

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    val = pbr->get_u64();
    return false;
  }
};

template <typename Enum>
struct ProtobufReader::FieldEnum {
  int field;
  Enum &val;

  FieldEnum(int f, Enum &v) : field((f * 8) + PB_TYPE_VARINT), val(v) {}

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    val = static_cast<Enum>(pbr->get_u32());
    return false;
  }
};

struct ProtobufReader::FieldPackedUInt32 {
  int field;
  std::vector<uint32_t> &val;

  FieldPackedUInt32(int f, std::vector<uint32_t> &v) : field((f * 8) + PB_TYPE_FIXEDLEN), val(v) {}

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    uint32_t len           = pbr->get_u32();
    const uint8_t *fld_end = std::min(pbr->m_cur + len, end);
    while (pbr->m_cur < fld_end) val.push_back(pbr->get_u32());
    return false;
  }
};

struct ProtobufReader::FieldString {
  int field;
  std::string &val;

  FieldString(int f, std::string &v) : field((f * 8) + PB_TYPE_FIXEDLEN), val(v) {}

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    uint32_t n = pbr->get_u32();
    if (n > (size_t)(end - pbr->m_cur)) return true;
    val.assign((const char *)(pbr->m_cur), n);
    pbr->m_cur += n;
    return false;
  }
};

struct ProtobufReader::FieldRepeatedString {
  int field;
  std::vector<std::string> &val;

  FieldRepeatedString(int f, std::vector<std::string> &v)
    : field((f * 8) + PB_TYPE_FIXEDLEN), val(v)
  {
  }

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    uint32_t n = pbr->get_u32();
    if (n > (size_t)(end - pbr->m_cur)) return true;
    val.resize(val.size() + 1);
    val.back().assign((const char *)(pbr->m_cur), n);
    pbr->m_cur += n;
    return false;
  }
};

template <typename Enum>
struct ProtobufReader::FieldRepeatedStructFunctor {
  int field;
  std::vector<Enum> &val;

  FieldRepeatedStructFunctor(int f, std::vector<Enum> &v)
    : field((f * 8) + PB_TYPE_FIXEDLEN), val(v)
  {
  }

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    uint32_t n = pbr->get_u32();
    if (n > (size_t)(end - pbr->m_cur)) return true;
    val.resize(val.size() + 1);
    if (!pbr->read(&val.back(), n)) return true;
    return false;
  }
};

template <typename Enum>
struct ProtobufReader::FieldRepeatedStructBlobFunctor {
  int field;
  std::vector<Enum> &val;

  FieldRepeatedStructBlobFunctor(int f, std::vector<Enum> &v)
    : field((f * 8) + PB_TYPE_FIXEDLEN), val(v)
  {
  }

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    uint32_t n = pbr->get_u32();
    if (n > (size_t)(end - pbr->m_cur)) return true;
    val.resize(val.size() + 1);
    val.back().assign(pbr->m_cur, pbr->m_cur + n);
    pbr->m_cur += n;
    return false;
  }
};

struct ProtobufWriter::WriterBuild {
  int struct_size;
  ProtobufWriter *p;

  WriterBuild(ProtobufWriter *pbw) : struct_size(0), p(pbw) {}

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
    sz = p->write(&v);
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
